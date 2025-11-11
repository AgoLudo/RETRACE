
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from datetime import timedelta
from scipy.signal import savgol_filter
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from scipy.stats import kendalltau
import emd
from scipy.signal import hilbert
import pickle
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator, FixedLocator,FuncFormatter
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines

# ---------------------- side plot functions ---------------------------------
from matplotlib.ticker import Locator

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

# ------- evaluate mass conservation error in the simulation -----------
def mass_conservation(df_q,df_ssc,
                      col_values_q,col_values_ssc,
                      timedelta_q,timedelta_ssc,
                      sigma_t_q,sigma_t_ssc,
                      time_step,
                      df_sim,debug):
    
    
    # event extent
    event_init=df_sim.index[0]
    event_end=df_sim.index[-1]
    dt_sim=(df_sim.index[3]-df_sim.index[2]).total_seconds()
    
    # ------ mass simulation-----
   
    q_up_sim=df_sim[(col_values_q+'_up')].values
    ssc_up_sim=df_sim[(col_values_ssc+'_up')].values
    
    vect_sim=q_up_sim*ssc_up_sim/1000
    # calculate mass
    mass_up_sim=np.nansum(dt_sim*vect_sim)
    
    # ----------mass observation-----------------
    # time observation init
    init_obs_q=event_init-timedelta_q+sigma_t_q
    init_obs_ssc=event_init-timedelta_ssc+sigma_t_ssc
    # time observation end
    end_obs_q=event_end-timedelta_q-sigma_t_q
    end_obs_ssc=event_end-timedelta_ssc-sigma_t_ssc
    
    q_obs  =df_q  .loc[init_obs_q:end_obs_q]
    ssc_obs=df_ssc.loc[init_obs_ssc:end_obs_ssc]
    
    joined_df = q_obs.join(ssc_obs, how='inner')
    # 
    if joined_df.empty:
        n = len(df_sim)  # target length for deltas
        nan_arr = np.full(n, np.nan)
        debug and print(event_init,'** no Q and SSC data overlap **')
        return np.nan, np.nan, nan_arr, nan_arr, df_sim.index
    
    q_obs_values=joined_df[col_values_q].values
    ssc_obs_values=joined_df[col_values_ssc].values
    vect_obs=q_obs_values*ssc_obs_values/1000
    
    
    # calculate mass
    mass_obs=np.nansum(dt_sim*vect_obs)

    # calculate the mass difference
    rel_mass = (mass_up_sim-mass_obs)/mass_obs
    
    # estimate the error mass based on the time difference
    vect_obs_clean = pd.Series(vect_obs).dropna()
    vect_sim_clean = pd.Series(vect_sim).dropna()
    
    N_obs=len(vect_obs_clean)
    N_sim=len(vect_sim_clean)
    
    error_mass= (abs(N_obs-N_sim)*dt_sim*
                 np.nanmin(vect_obs_clean)/1000)/mass_obs
    # 
    
    # import the downstream observations ( REAL DATA)
    q_down=df_sim[(col_values_q+'_down')].values
    ssc_down=df_sim[(col_values_ssc+'_down')].values
    
    
    if len(df_sim)<len(joined_df):
        joined_df=joined_df.iloc[0:len(df_sim),:]
        q_obs_values=joined_df[col_values_q].values
        ssc_obs_values=joined_df[col_values_ssc].values
        
        
    elif len(df_sim)>len(joined_df):
        df_sim=df_sim.iloc[0:len(joined_df),:]
        q_down=df_sim[(col_values_q+'_down')].values
        ssc_down=df_sim[(col_values_ssc+'_down')].values
        
        
    elif len(df_sim)==len(joined_df):
        print('same length')
    
    # -------------- calculate the observed delta SSC -----------------
    delta_ssc_obs=ssc_down-ssc_obs_values
    # ------------- calulate the observed delta Q ------------------
    delta_q_obs=q_down-q_obs_values
    # -----------------------------------------------------------
    new_time_index=df_sim.index
    
    return rel_mass,error_mass,delta_ssc_obs,delta_q_obs,new_time_index

def mass_conservation_routine(station_couple,
                              q_dict_data,ssc_dict_data,
                              reach_hydro_df,geom_dict,
                              input_sim_station_difference,
                              input_dict_FPCA,sim_dict,
                              debug):
    
    '''
    IN
    station_couple: list with the two stations upstream - downstream
    q_dict_data,ssc_dict_data: discharge and ssc data t the two stations
    reach_hydro_df: hydraulics parameters of the reach
    geom_dict: dictionary with the reach geom properties

    # dictionary with parameters
    input_sim_station_difference: to calculate the sim station difference
    input_dict_FPCA: to performe the FPCA
    sim_dict: dicionnary with the simulations results

    OUT
    rel_mass_df: relative mass variation and erro for each simualtion
    delta_vars_dict:delta Q and delta SSC for every event
    '''
    
    # time step of the simulation
    time_step=input_sim_station_difference['time_step']
    # dataframe from observation
    q_df=q_dict_data[station_couple['upstream']]
    q_df_10min = q_df.resample(time_step).interpolate(method='linear')
    
    ssc_df_10m=ssc_dict_data[station_couple['upstream']]
    col_values_q  =input_dict_FPCA['col_values_q']
    col_values_ssc=input_dict_FPCA['col_values_ssc']
    
    # distance 
    L=geom_dict['distance']
    method_ssc=input_sim_station_difference['diffusion_method_ssc']
    
    # event list
    event_list=list(sim_dict.keys())
    # lists to save
    rel_mass_list=[]
    err_list=[]
    # 
    delta_vars_dict={} # calculate the delta ssc based just on event routing (no advection diffusion)
    for event in event_list:
        delta_vars_dict[event]={}
        
        # load the simlulation df
        sim_df=sim_dict[event]
        
        # load sim parameters
        lambda_ssc=reach_hydro_df.loc[event,'U']
        lambda_q=reach_hydro_df.loc[event,'dqdD']
        k_ssc=reach_hydro_df.loc[event,method_ssc]
        k_q=reach_hydro_df.loc[event,'K_q']
        
        # advection     time
        timedelta_ssc=timedelta(seconds=L/lambda_ssc)
        timedelta_q  =timedelta(seconds=L/lambda_q)
        # diffusion time
        sigma_t_q  =timedelta(seconds=np.sqrt(2*k_q  *L)/(lambda_q  **3))
        sigma_t_ssc=timedelta(seconds=np.sqrt(2*k_ssc*L)/(lambda_ssc**3))
        # 
        (rel_mass,err,
         delta_ssc_obs,delta_q_obs,
         new_time_index)=mass_conservation(q_df_10min,ssc_df_10m,
                                   col_values_q,col_values_ssc,
                                   timedelta_q,timedelta_ssc,
                                   sigma_t_q,sigma_t_ssc,
                                   time_step,
                                   sim_df,debug)
        # save to list the scalar value for the mass
        rel_mass_list.append(rel_mass)
        err_list.append(err)
        
        # save df differences from data
        df2save=pd.DataFrame(index=new_time_index,
                             columns=[col_values_q,col_values_ssc])
        df2save[col_values_q]=delta_q_obs
        df2save[col_values_ssc]=delta_ssc_obs
        delta_vars_dict[event]=df2save
        
        # save the time step differences from discharge and ssc to datafreme and dictionary
    rel_mass_df = pd.DataFrame({
        'rel_mass': rel_mass_list,
        'error': err_list
            }, index=event_list)
    
    # clean simulations
    rel_mass_thr=input_dict_FPCA['rel_mass_threshold']
    rel_mass_df=rel_mass_df[rel_mass_df['rel_mass']<rel_mass_thr]
    # 
    
    return rel_mass_df,delta_vars_dict

def plot_mass_error(rel_mass_df,input_dict,debug):
    # Ensure the DataFrame is sorted by the index (time)
    rel_mass_df = rel_mass_df.sort_index()

    # Calculate and print the overall relative mass error
    overall_rel_mass_error = np.round(np.nanmean(rel_mass_df['rel_mass'].values),4)
    debug and print(f"relative mass difference: {overall_rel_mass_error:.4f}")
    # Calculate the standard deviation of the rel_mass distribution
    rel_mass_std =np.nanstd( rel_mass_df['rel_mass'].values)

    # Create the upper and lower bounds for the error lines
    upper_bound = overall_rel_mass_error + rel_mass_std
    lower_bound = overall_rel_mass_error - rel_mass_std
     
     # Calculate and print the relative error over time
    rel_error_over_time = np.round(rel_mass_df['error'].median(),4)
    debug and print(f"mass difference error: {rel_error_over_time}")
    
    
    plt.figure(figsize=input_dict['figure_size'])

    plt.errorbar(rel_mass_df.index, rel_mass_df['rel_mass'],
                 yerr=rel_mass_df['error'],
                 fmt=input_dict['marker_style'],
                 color=input_dict['marker_color'],
                 ecolor=input_dict['error_color'],
                 markeredgecolor=input_dict['marker_edge_color'],
                 markersize=input_dict['marker_size'],
                 elinewidth=input_dict['error_line_width'])
# Plot the overall relative mass error line
    plt.hlines(y=overall_rel_mass_error, 
               xmin=rel_mass_df.index.min(), 
               xmax=rel_mass_df.index.max(),
                color=input_dict['avg_line_color'],
                linewidth=input_dict['error_line_width']*2)


    # Fill the area between the upper and lower bounds with a pale color
    plt.fill_between(rel_mass_df.index, 
                     lower_bound, upper_bound, 
                     color=input_dict['error_color'], 
                     alpha=input_dict['alpha'])
    
    plt.xlabel(input_dict['x_label'],fontsize=input_dict['labelfont'])
    plt.ylabel(input_dict['y_label'],fontsize=input_dict['labelfont']+5)
    plt.title(input_dict['title'],fontweight='bold',fontsize=input_dict['labelfont']+4)
    plt.grid(input_dict['grid'], lw=input_dict['grid_line_width'])

    for annotation in input_dict['annotations']:
        plt.annotate(f'mass difference: {overall_rel_mass_error:.4f}' 
                     if annotation['position'] == (0.05, 0.95) else
                     f'average time error: {rel_error_over_time}',
                     
                     xy=annotation['position'],
                     xycoords='axes fraction',
                     fontsize=annotation['fontsize'],
                     ha=annotation['horizontal_alignment'],
                     va=annotation['vertical_alignment'])

    plt.show()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
#              SUSPENDED SEDIMENT VARIATION                       
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# prepare the data from PCA analysis
def smooth_with_savgol_filter(curve, window_length, polyorder):
    return savgol_filter(curve, window_length, polyorder)

def clean_dataframe(df):
        '''Interpolate internal NaNs, drop edge NaNs.'''
        df = df.interpolate(method="linear", limit_direction="both") # fills inside NaNs
        df = df.dropna(how="any")
        return df

def prepare_data_FPCA(df_sim,
                     input_dict_FPCA,
                     slope,width,
                     debug):
    '''
    df_sim: tataframe with the simulated and observed data at downstream station
    input_dict_FPCA: FPCA parameters
    '''
    rho=1000
    g=9.81

    # --- smooth parameters ---
    window_length     =input_dict_FPCA['window_size'    ]
    polyorder         =input_dict_FPCA['polyorder'      ]
    col_values_q      =input_dict_FPCA['col_values_q'   ]
    col_values_ssc    =input_dict_FPCA['col_values_ssc' ]
    columns_FPCA_list =input_dict_FPCA['columns_FPCA'   ].copy()
   
    # ------- load simulation and observations ------------
    df_sim=clean_dataframe(df_sim)
    q_up_sim  =df_sim[(col_values_q  +'_up'  )].values
    q_down    =df_sim[(col_values_q  +'_down')].values
    ssc_up_sim=df_sim[(col_values_ssc+'_up'  )].values
    ssc_down  =df_sim[(col_values_ssc+'_down')].values
    
    #  >>>>>>>>>>>>>> HERE CUSTOM VARIABLES <<<<<<<<<<<<<<<<<<<<<<
    # variable 1: discharge change
    delta_q = q_down-q_up_sim
    
    # variable 2: shear stress (from Normal Flow Depth=Q**3/5)
    depth=(q_up_sim/(33*width*np.sqrt(slope)))**(3/5) # ks=33, but not important, all terms are later normalized for the FPCA.
    q_scale_shear=rho*g*depth*slope # engelund shear

    # # variable 3: shear stress variation
    dt_sec=(df_sim.index[3]-df_sim.index[2]).total_seconds()
    q_scale_derivative=np.gradient(depth,dt_sec)
  
    # variable 4: ssc concentration
    ssc_input = ssc_up_sim /1000 # g/L
    
    # target delta Mass
    target = ((ssc_down*q_down)-(ssc_up_sim*q_up_sim))/1000 #kg/s


    # ---------- save variables to dataframe-------
    df2_FPCA=pd.DataFrame(columns=(columns_FPCA_list+['time']))
    # set the temporal index
    df2_FPCA['time']=df_sim.index
    df2_FPCA = df2_FPCA.set_index('time')

    # set the variables and the target
    df2_FPCA['delta_q']=delta_q
    
    df2_FPCA['q_shear']=q_scale_shear
    df2_FPCA['D_der_sec']=q_scale_derivative
    
    df2_FPCA['ssc_input']=ssc_input
    
    df2_FPCA['target']=target

    df2save=df2_FPCA.copy()

    # ---------------- iterate and prepare variable-----------
    for variable in df2_FPCA.columns:  
        if variable!='target':      
            median_val=df2_FPCA[variable].median()
            iqr_val=(df2_FPCA[variable].quantile(0.75) - df2_FPCA[variable].quantile(0.25))
            df2_FPCA[variable] =(df2_FPCA[variable] - median_val )/iqr_val

        # smooth normalized variables for FPCA 
        df2_FPCA[variable] = smooth_with_savgol_filter(
            df2_FPCA[variable].values, window_length, polyorder)
    
    if df2_FPCA.isna().any().any():
        print(q_down,'\n',q_up_sim)
        raise ValueError("NaN detected in FPCA input variables:\n"
                         f"{df2_FPCA.isna().sum()}")

    if not np.isfinite(df2_FPCA.values).all():
        raise ValueError("Inf/-Inf detected in FPCA input variables")

    return [df2_FPCA,df2save]

# -----------------------------------------------------------------
# function to find the optimal number of components of the FPCA
def choose_optimal_n_components(fd, max_components, threshold
                                ,debug):
    for n_components in range(1, max_components + 1):
        fpca_discretized = FPCA(n_components=n_components)
        fpca_discretized.fit(fd)
        explained_variance = fpca_discretized.explained_variance_ratio_
        total_explained_variance = np.sum(explained_variance)
        
        if total_explained_variance >= threshold:
            # debug and print(f'N_components: {n_components}','\n',
            #                 f'Explained Variance Ratio: {total_explained_variance}')
            return n_components
    return max_components

def r2_adjusted(numb_of_data,n_optimal_components,r2):
    num=(1-r2)*(numb_of_data-1)
    denom=numb_of_data-n_optimal_components-1
    return 1-(num/denom)

# -----------------------------------------------------------------

def FPCA_regression(sim_dict,
                    input_dict_FPCA,
                    geom_dict,
                    debug):
    '''
    IN
    sim_dict: dictionary with each event, the observed and the simulated results
    input_dict_FPCA: FPCA parameters, defined in main_input.py

    OUT
    event_variable_df: FPCA results with R2, init and end date, and weights for each variable (MASS)
     input_FPCA_variables: event inputs and target, real values
    '''
    # geom parameteers
    slope=geom_dict['slope']
    width=geom_dict['width']
    # --- fpca parameters ---
    exp_variance_tr=input_dict_FPCA['exp_variance_threshold' ]
    ortho_tr       =input_dict_FPCA['orthogonality_threshold']
    r2_tr          =input_dict_FPCA['r2_threshold'           ]
    
    # ---- iterate through the events -------
    event_list=list(sim_dict.keys())
    results=[]
    input_FPCA_variables={}
    for event_date in event_list:
        input_FPCA_variables[event_date]={}
        # event data
        sim_df=sim_dict[event_date].copy()
        
        init=sim_df.index[0]
        end=sim_df.index[-1]
        
        # prepare data for the PCA Regression
        df2_FPCA,df2save=prepare_data_FPCA(sim_df,
                         input_dict_FPCA,
                         slope,width,
                         debug)
    
        # save the real variables to then analyse later
        input_FPCA_variables[event_date]=df2save.copy()
        
        # perform the FPCA
        target_vect = df2_FPCA['target'].values

        # 1: Extract all columns from the DataFrame
        columns = df2_FPCA.columns
        columns = columns.drop('target') # remove the target column

        # 2: Stack the normalized arrays vertically
        data=df2_FPCA[columns].values.T
        
        # 3: Perform FPCA on the matrix_data array
        grid = np.linspace(0, 1,data.shape[1])  
        fd = FDataGrid(data_matrix=data, grid_points=grid)
        #  3b: estimate optimal number of FPCA components
        max_components=len(columns)
        optimal_n_components = choose_optimal_n_components(
                                fd, max_components,exp_variance_tr,debug)
        
        # 4: compute the FPCA matrix and normlize
        fpca_discretized = FPCA(n_components=optimal_n_components)
        fpca_discretized.fit(fd)
        FPCA_components=fpca_discretized.fit(fd).components_.data_matrix
        
        # --------------------- Verify Orthogonality of FPCs ---------------------
        # normalize
        if optimal_n_components>=2:
            X_flat = FPCA_components.squeeze()  
            X_flat_normalized = X_flat / np.linalg.norm(X_flat, axis=1, keepdims=True)     
            orthogonality_matrix = np.dot(X_flat_normalized, X_flat_normalized.T) 
            off_diagonal_elements = np.sum(np.abs(np.tril(orthogonality_matrix, -1)))
            diagonal_elements = np.sum(np.diag(orthogonality_matrix))

            ratio=off_diagonal_elements/diagonal_elements
            # debug and print(f'covariance/correlation: {ratio}')
            
            # if overfitting introduces some covariance correct
            if ratio>ortho_tr:
                if optimal_n_components>=2:
                    optimal_n_components-=1 # overfitting can cause orthogonality issues
                    debug and print('ortho reduction: ',optimal_n_components)
                    # Fit FPCA with the optimal number of components
                    fpca_discretized = FPCA(
                    n_components=optimal_n_components)
                    fpca_discretized.fit(fd)
                    FPCA_components=fpca_discretized.fit(fd).components_.data_matrix

        
        # --------- fitting procedure and weight saving -------------------
        # prepare data for dtaframe
        components_2d = np.squeeze(FPCA_components)
        data = components_2d.T
        # prepare data in dataframe
        col_name = [f'FPC_{i+1}' for i in range(optimal_n_components)]
        df_input_fit=pd.DataFrame(index=grid,columns=col_name,
                                  data=data)
        # FPCA_components.T[0]
        # 5: Linear Regression 
        reg=LinearRegression().fit(df_input_fit,target_vect)
        target_predict=reg.predict(df_input_fit)
        
        # 6 : weights
        coefficients=reg.coef_
        scores=fpca_discretized.transform(fd)
        importance_score=np.abs(scores @ coefficients) # matrix dot product
        importance_score=importance_score/np.sum(importance_score) # normalized
        
        # r2 score assessment
        r2 = r2_score(target_vect,target_predict)
        adjusted_r2=r2_adjusted(len(target_vect),
                                optimal_n_components,r2)
        # saving df
        weights_df=pd.DataFrame(index=[event_date],
                                columns=columns)
        
        if adjusted_r2>=r2_tr: #control over fitting
            # save the weights
            weights_df.iloc[0,:]=importance_score
            
        else:
            weights_df.iloc[0,:]=np.nan
            
        # append new data
        weights_df['r2'] = adjusted_r2
        weights_df['start_time'] = init
        weights_df['end_time'] = end
        weights_df['event_date'] = event_date   

        # append dataframe line to the results list
        results.append(weights_df)
    # ----------------------------------------------   
    # save in single dataframe
    event_variable_df=pd.concat(results, ignore_index=True)
    event_variable_df=event_variable_df.set_index('event_date')
    
    debug and print('input',len(sim_dict),
                    '\nclassified',len(event_variable_df.dropna()),
                    '\nnon classified events: ',len(event_variable_df)-len(event_variable_df.dropna())
                    )
    
    return [event_variable_df,input_FPCA_variables]

# -----------------------------------------------------------------

def FPCA_Non_explained_plot(event_variable_df):
    explained_df=event_variable_df.dropna()
    non_explained_df = event_variable_df[event_variable_df.isna().any(axis=1)]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    plt.scatter(explained_df.index,
                explained_df['r2'].values, color='blue',
                label='Explained Events')
    
    plt.scatter(non_explained_df.index,
                non_explained_df['r2'].values, color='red',
                label='Non-Explained Events')

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel(r'$R^2$')
    plt.grid(lw=0.15,which='major',axis='y')
    plt.title(r'$R^2$ threshold for Linear Regression')
    plt.legend(framealpha=1)
    

    # Show the plot
    plt.show()
    
def FPCA_variables_classification(event_variable_df,
                                         input_dict_FPCA,
                                         debug):
    '''
    classifies the event based on the highest scoring variable
    IN
    event_variable_df= dataframe with the scores for each variable
    input_dict_FPCA: parameters from the FPCA

    OUT
    classified_variables_dict: dictionary with the events classified by variable (the weights and the r2)
    '''
    
    col_variables=input_dict_FPCA['columns_FPCA'].copy()
    col_variables.remove('target')
    nan_df = event_variable_df[event_variable_df[col_variables].isna().any(axis=1)]
    
    # ------------------------------

   # Initialize a dictionary with lists for each variable
    classification_dict = {var: [] for var in col_variables}

    event_variable_df_clean = event_variable_df.dropna()
    for index, row in event_variable_df_clean.iterrows():
        weights = row[col_variables].copy()
        max_weight_var = weights.idxmax()

        # Classify the variable based on the column with the maximum value
        classification_dict[max_weight_var].append(row)

    col2save = event_variable_df_clean.columns
    # Convert the lists of classified rows into DataFrames
    classified_variables_dict = {var: pd.DataFrame(
        classification_dict[var], columns=col2save) for var in col_variables}
    classified_variables_dict['non_explained'] = nan_df # non explained events
    
    N_events=len(event_variable_df)
    # Calculate and print the ratio for each category
    for category, df in classified_variables_dict.items():
        ratio = np.round(len(df) / N_events, 3)
        print(f'{category}: {ratio}')

    return classified_variables_dict 

def convert_to_seconds(time_str):
    # Define a dictionary to map time units to their corresponding seconds
    unit_to_seconds = {
        'sec': 1,
        'min': 60,
        'h': 3600,
        'd': 86400
    }

    # Iterate over the dictionary to find a matching unit
    for unit, seconds_per_unit in unit_to_seconds.items():
        if time_str.endswith(unit):
            # Extract the numeric part
            value = int(time_str[:-len(unit)])
            # Convert to seconds
            total_seconds = value * seconds_per_unit
            return total_seconds

def plot_error_SimObs_ssc_season(list_season,
                          col_values_q,col_values_ssc,
                          sim_dict,delta_vars_dict,
                          smoothing_window,polyorder):
    #  classify events by season 
    dict_season={season: {} for season in list_season}
    list_events=list(sim_dict.keys())
    
    for event_idx in list_events:
        # import the observed differences between data at the upstream and dowsntream station
        df_diff_obs=delta_vars_dict[event_idx].copy()    
        # import the simulation data
        df_sim=sim_dict[event_idx].copy()
        # 
        # df joined
        df_joined = df_diff_obs.join( df_sim, how='inner')

        
        # compute the variable differences
        q_up_sim  =df_joined[(col_values_q  +'_up'  )].values
        q_down    =df_joined[(col_values_q  +'_down')].values
        ssc_up_sim=df_joined[(col_values_ssc+'_up'  )].values
        ssc_down  =df_joined[(col_values_ssc+'_down')].values
        
        
        df_diff_obs[(col_values_ssc+'_sim')]=ssc_down-ssc_up_sim
        df_diff_obs[(col_values_q+'_sim')]=q_down-q_up_sim
        # 
        # normalize the time index
        df_diff_obs.index=(df_diff_obs.index-df_diff_obs.index[0])/(
                            df_diff_obs.index[-1]-df_diff_obs.index[0])
        
        # get the season in which the event happened
        month = event_idx.month
        season = get_season(month)
        
        dict_season[season][event_idx]={}
        dict_season[season][event_idx]=df_diff_obs
    
    # -------- plot the average differences for each season ---------
    common_length=100
    # ------- PLOT DISCHARGE ----
    fig_q, axs_q = plt.subplots(2, 2, figsize=(15, 15), sharey=True)
    axs_q = axs_q.flatten()
    fig_ssc, axs_ssc = plt.subplots(2, 2, figsize=(15, 15), sharey=True)
    axs_ssc = axs_ssc.flatten()
    
    # Loop over each season and plot curves.
    for idx, season in enumerate(list_season):
        # Lists to store interpolated curves for discharge and SSC for the current season.
        curves_q = []
        curves_q_obs=[]
        curves_ssc = []
        curves_ssc_obs = []
        
        # For each event in the season, interpolate the curves to a common time axis.
        for event_idx, df in dict_season[season].items():
            # Define common time grid based on normalized index (0 to 1)
            x_common = np.linspace(df.index.min(), df.index.max(), common_length)
            # -----------------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------
            # Interpolate discharge differences simulated
            f_q = interp1d(df.index, df[f"{col_values_q}_sim"],
                           kind='linear', fill_value="extrapolate")
            y_q = f_q(x_common)
            curves_q.append(y_q)
            
            # Interpolate SSC differences simulated
            f_ssc = interp1d(df.index, df[f"{col_values_ssc}_sim"],
                             kind='linear', fill_value="extrapolate")
            y_ssc = f_ssc(x_common)
            curves_ssc.append(y_ssc)
            # -----------------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------
            f_q_obs=interp1d(df.index, df[col_values_q],
                           kind='linear', fill_value="extrapolate")
            y_q_obs=f_q_obs(x_common)
            curves_q_obs.append(y_q_obs)
            # ssc
            f_ssc_obs = interp1d(df.index, df[col_values_ssc],
                             kind='linear', fill_value="extrapolate")
            y_ssc_obs = f_ssc_obs(x_common)
            curves_ssc_obs.append(y_ssc_obs)
            
            
            
        # Convert list of curves to numpy arrays for median computation
        curves_q = np.array(curves_q)
        curves_ssc = np.array(curves_ssc)
        curves_q_obs = np.array(curves_q_obs)
        curves_ssc_obs = np.array(curves_ssc_obs)
        
        # Plot all curves in light grey (if any curves exist)
       
        for curve in curves_q:
                axs_q[idx].plot(x_common, curve, color='lightgrey', lw=0.5)
  
        
        for curve in curves_ssc:
                axs_ssc[idx].plot(x_common, curve/1000, 
                                  color='lightgrey', lw=0.5)

        
        if len(curves_q) > 0:
            curves_q = np.array(curves_q)
            median_q = np.nanmedian(curves_q, axis=0)
            axs_q[idx].plot(x_common, median_q, 
                            color='black', lw=2, label='Median sim')
  
        
        if len(curves_q_obs) > 0:
            curves_q_obs = np.array(curves_q_obs)
            median_q_obs = np.nanmedian(curves_q_obs, axis=0)
            axs_q[idx].plot(x_common, median_q_obs, 
                            color='black', linestyle=':', lw=2, label='Median obs')

        if len(curves_ssc) > 0:
            curves_ssc = np.array(curves_ssc)
            median_ssc = np.nanmedian(curves_ssc, axis=0)
            axs_ssc[idx].plot(x_common, median_ssc / 1000, 
                              color='black', lw=2, label='Median')

        if len(curves_ssc_obs) > 0:
            curves_ssc_obs = np.array(curves_ssc_obs)
            median_ssc_obs = np.nanmedian(curves_ssc_obs, axis=0)
            axs_ssc[idx].plot(x_common, median_ssc_obs / 1000,
                               color='black', linestyle=':', lw=2, label='Median')

        
        # Set titles and labels for each subplot
        axs_q[idx].set_title(f"{season} - Discharge")
        axs_q[idx].set_xlabel("Normalized Time")
        if idx == 0:
            axs_q[idx].set_ylabel(f"{col_values_q} difference")
        axs_q[idx].legend()
        
        axs_ssc[idx].set_title(f"{season} - SSC")
        axs_ssc[idx].set_xlabel("Normalized Time")
        if idx == 0:
            axs_ssc[idx].set_ylabel(f"{col_values_ssc} difference")
        axs_ssc[idx].legend()

    for ax in axs_ssc:
        ax.set_yscale('symlog', linthresh=0.1)
    for ax in axs_q:
        ax.set_yscale('symlog', linthresh=100)
    fig_q.tight_layout()
    fig_ssc.tight_layout()
    plt.show()


# ----------------------------------------------------------------- 
# -----------------------------------------------------------------
#                               MASS VARIATION
# -----------------------------------------------------------------
# -----------------------------------------------------------------

def calc_cumulate_mass(q_values,ssc_values,dt):
    return_mass=np.nansum(q_values * (ssc_values / 1000) * dt)
    return return_mass

def calc_time_mass(q_values,ssc_values):
    mass_time=q_values * (ssc_values / 1000)
    return mass_time

#

def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'
    

def get_month(month,list_month):
    return list_month[month-1]

def total_mass(station_couple,
               col_values_q,col_values_ssc,
               q_dict_data_10m,ssc_dict_data_10m,
               sim_dict,dt,
               event_class_dict,
               debug):
    
    '''
    calulates the mass contribution for the events, and for each single variable
    IN
    station_couple: station couple upstream - downstream
    col_values_q,col_values_ssc: name of the columns for dishcrage and ssc
    q_dict_data_10m,ssc_dict_data_10m: data for two station, at 10 minutes resolution
    sim_dict: dicitonary with event obs and sim
    dt: seconds of the time interval
    event_class_dict: dictionary with the events classified by variable ( the weights and the r2)

    OUT
    mass_station_dict: dictionary for each var classified event. Each had a dataframe with
        'mass_kg': event mass 
        'theor_mass_kg':event mass (corrected by the timesteps missing)
        'event_mass_kg': mass of all events
        'time_events': time of all the events

    '''


    #--------  Total mass events vs timeseiries (at the station) ------
    mass_station_dict={}
    for loc in station_couple:
        station=station_couple[loc]
        mass_station_dict[station]={}
        
        # data
        q_df  =q_dict_data_10m  [station].copy()
        ssc_df=ssc_dict_data_10m[station].copy()  
        
        # Perform inner join on the time index
        merged_df = pd.merge(q_df, ssc_df, 
                             left_index=True, 
                             right_index=True, how='inner')
        # 

        merged_df=clean_dataframe(merged_df)
        init_data=merged_df.index[0]
        end_data=merged_df.index[-1]
        timesteps=(end_data-init_data).total_seconds() / dt
        real_timesteps=len(merged_df)
        
        total_data_time=real_timesteps*dt
        # Extract the relevant columns for the calculation
        q_values = merged_df[col_values_q].values
        ssc_values = merged_df[col_values_ssc].values
        # Calculate the mass
        mass = calc_cumulate_mass(q_values,ssc_values,dt)  # in this case, the data are in mg/L
        mass_real=mass*(timesteps/real_timesteps) 
        
        # ---------------- event total mass ------------
        event_list=list(sim_dict.keys())
        if loc=='upstream':
            sfx='_up'
        elif loc=='downstream':
            sfx='_down'
        
        tot_mass=0
        tot_time=0
        for event in event_list:
            df_sim=sim_dict[event].copy()
            df_sim=clean_dataframe(df_sim) # linear interpolate and drop nan

            init=df_sim.index[0]
            end=df_sim.index[-1]
            time_event=(end-init).total_seconds()
            tot_time=tot_time+time_event
            
            q_event  =df_sim[(col_values_q  +sfx  )].values
            ssc_event=df_sim[(col_values_ssc+sfx  )].values
            
            mass_event=calc_cumulate_mass(q_event,ssc_event,dt)
            tot_mass=tot_mass+mass_event
                    
        #-------------------------------------------------------------
        #                event based mass classification 
        # ------------------------------------------------------------  
        list_var=list(event_class_dict.keys())
        mass_dict_var={}
        for var in list_var: # iterate through the studiesd variables
            event_var_df=event_class_dict[var].copy()
           
            mass_var=0 # calculate the mass related to the variable classified events
            for event_idx in event_var_df.index:
                df_sim=sim_dict[event_idx].copy()
                
                q_event  =df_sim[(col_values_q  +sfx  )].values
                ssc_event=df_sim[(col_values_ssc+sfx  )].values
                
                mass_event=calc_cumulate_mass(q_event,ssc_event,dt)
                mass_var=mass_var+mass_event
            mass_dict_var[var]=mass_var
        
        
        # ----------------- save -------------------
        mass_df = pd.DataFrame({
        'mass_kg': [mass],
        'theor_mass_kg': [mass_real],
        'event_mass_kg':[tot_mass],
        'time_events':[tot_time]
            })
        for var in list_var:
            mass_df[var]=[mass_dict_var[var]]/mass_df['event_mass_kg']
            
        # ratio of mass for each cathegorized event
        debug and print(station,'mass_var_event\n',mass_df[list_var], 
              '\nmass ratio event/total:',tot_mass/mass,
              '\ntime ratio event/total',tot_time/total_data_time)
        
        mass_station_dict[station]=mass_df
    return mass_station_dict

def mass_variation(col_values_q,col_values_ssc,dt,
                    rel_mass_df,sim_dict,
                    input_dict_massVar_plot,
                    debug):
    '''
    this function calculates the total mass variation during the event
    ADD THE REAL INPUT MASS DIFFERENCE AND TOGETHER WITH THE SIMULATED
    IN
    col_values_q,col_values_ssc: name of the columsn in the input dataset
    dt: time step in seconds
    rel_mass_df: relative mass difference << from mass_conservation_routine
    sim_dict: sim and obs for each event
    input_dict_massVar_plot: input parameters to plot

    OUT
    - Plot of the cumulative mass and event mass balance
    delta_mass_df , delta_mass_df_obs: dataframe with the mass variation integral per each event
    '''
    rel_mass_df = rel_mass_df.sort_index()
    list_sim=rel_mass_df.index
    col_mass='delta_mass_ton'
    col_mass_cum='delta_mass_cum_ton'
    
    delta_mass_df=pd.DataFrame(index=list_sim,
                               columns=[col_mass,col_mass_cum])
    delta_mass_df_obs=pd.DataFrame(index=list_sim,
                               columns=[col_mass,col_mass_cum])
    #
    
    for event_idx in list_sim:
        event_df=sim_dict[event_idx]
        
        q_in  =event_df[(col_values_q+'_up')].values
        ssc_in=event_df[(col_values_ssc+'_up')].values
        
        q_out  =event_df[(col_values_q+'_down')].values
        ssc_out=event_df[(col_values_ssc+'_down')].values
        
        mass_input=calc_cumulate_mass(q_in,ssc_in,dt)
        mass_output=calc_cumulate_mass(q_out,ssc_out,dt)
        
        delta_mass=mass_output-mass_input
        delta_mass_df.loc[event_idx,col_mass]=delta_mass/1000 # express in [tons]
        # positive for erosion, negative for deposition
        # 
        delta_mass_df_obs.loc[event_idx,col_mass]= (delta_mass/1000)/(
                                        1+rel_mass_df.loc[event_idx,'rel_mass'])
         
    # calculate also the cumulative value  
    delta_mass_df.loc[:,col_mass_cum]=np.cumsum(
        delta_mass_df.loc[:,col_mass].values)
    
    delta_mass_df_obs.loc[:,col_mass_cum]=np.cumsum(
        delta_mass_df_obs.loc[:,col_mass].values)
    
    # ------------------------------------------
    # ------------    PLOT   -------------------
    # ----------------------------------------
    if debug:
        erosion_color        = input_dict_massVar_plot['erosion_color'       ]
        deposition_color     = input_dict_massVar_plot['deposition_color'    ]
        erosion_color_obs    = input_dict_massVar_plot['erosion_color_obs'   ]
        deposition_color_obs = input_dict_massVar_plot['deposition_color_obs']
        figsize              = input_dict_massVar_plot['figsize'             ]
        title                = input_dict_massVar_plot['title'               ]
        xlabel               = input_dict_massVar_plot['xlabel'              ]
        ylabel               = input_dict_massVar_plot['ylabel'              ]
        ylabel_cum           = input_dict_massVar_plot['ylabel2'             ]
        label_sim            = input_dict_massVar_plot['label_sim'           ]
        label_obs            = input_dict_massVar_plot['label_obs'           ]
        linestyle_obs        = input_dict_massVar_plot['linestyle_obs'       ]
        markerstyle_obs      = input_dict_massVar_plot['markerstyle_obs'     ]
        markersize_obs       = input_dict_massVar_plot['markersize_obs'      ]
        label_cum            = input_dict_massVar_plot['label2'              ]
        labelsize            = input_dict_massVar_plot['labelsize'           ]
        cum_linestyle        = input_dict_massVar_plot['cum_linestyle'       ]
        labelpad        = input_dict_massVar_plot['labelpad'       ]

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=figsize)

        colors = [deposition_color if val < 0 else erosion_color
                  for val in delta_mass_df[col_mass]]
        colors_obs= [deposition_color_obs if val < 0 else erosion_color_obs
                  for val in delta_mass_df_obs[col_mass]]
        
        ax1.bar(delta_mass_df.index, delta_mass_df[col_mass],
                color=colors, label=label_sim,width=2)
        
        ax1.scatter(delta_mass_df.index, delta_mass_df_obs[col_mass],
                marker=markerstyle_obs,s=markersize_obs,
                color=colors_obs, label=label_obs)

        # Plot the cumulative mass using a loop to assign colors correctly
        ax2 = ax1.twinx()
        for i in range(len(delta_mass_df) - 1):
            
            if i==0:
                ax2.plot(delta_mass_df.index[i:i+2], delta_mass_df[col_mass_cum][i:i+2],
                        color=deposition_color if delta_mass_df[col_mass_cum].iloc[i] < 0 else erosion_color,
                        linestyle=cum_linestyle,label=label_cum)
                # obs
                ax2.plot(delta_mass_df_obs.index[i:i+2], delta_mass_df_obs[col_mass_cum][i:i+2],
                        color=deposition_color_obs if delta_mass_df_obs[col_mass_cum].iloc[i] < 0 else erosion_color_obs,
                        linestyle=cum_linestyle)
            else:
                # sim
                ax2.plot(delta_mass_df.index[i:i+2], delta_mass_df[col_mass_cum][i:i+2],
                        color=deposition_color if delta_mass_df[col_mass_cum].iloc[i] < 0 else erosion_color,
                        linestyle=cum_linestyle)
                # obs
                ax2.plot(delta_mass_df_obs.index[i:i+2], delta_mass_df_obs[col_mass_cum][i:i+2],
                        color=deposition_color_obs if delta_mass_df_obs[col_mass_cum].iloc[i] < 0 else erosion_color_obs,
                        linestyle=cum_linestyle)
            
        max_abs_cum_mass = max(abs(delta_mass_df[col_mass_cum].min()), 
                            abs(delta_mass_df[col_mass_cum].max()))
        max_abs_mass = max(abs(delta_mass_df[col_mass].min()), 
                        abs(delta_mass_df[col_mass].max()))

        margin_cum = 0.2 * max_abs_cum_mass  # 20% margin for better visualization
        margin_mass = 0.2 * max_abs_mass

        ax2.set_ylim(-max_abs_cum_mass, max_abs_cum_mass+ margin_cum )
        ax1.set_ylim(-max_abs_mass , max_abs_mass+margin_mass )
        
        ax1.axhline(0, color='black', linewidth=1.2)
        
        
        # Set labels and title
        ax1.set_xlabel(xlabel     ,fontsize=labelsize  )
        ax1.set_ylabel(ylabel     ,fontsize=labelsize  )
        ax2.set_ylabel(ylabel_cum ,fontsize=labelsize  )
        # plt.title     (title      ,fontsize=labelsize+4)

        ax1.set_yscale('symlog',
                        linthresh=input_dict_massVar_plot['linthreshold_cum_mass'])
        
        ax2.set_yscale('symlog',
                        linthresh=input_dict_massVar_plot['linthreshold_cum_mass']*1000)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        lines=lines1+lines2
        labels=labels1+labels2
        ax1.legend(lines, labels, loc='lower right',
                   frameon=True, facecolor='white',fontsize=labelsize*0.95)
        ax1.tick_params(axis='both', labelsize=labelsize*0.95)
        ax2.tick_params(axis='y', labelsize=labelsize*0.95)

        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        ax1.grid(lw=0.3,ls=':',which='major',axis='both')


        # ---- Label padding ----

        # Padding for axis labels
        ax1.xaxis.labelpad = labelpad  # x-axis label padding
        ax1.yaxis.labelpad = labelpad-3  # left y-axis label padding
        ax2.yaxis.labelpad = labelpad-3  # right y-axis label padding
        # Show plot
        plt.tight_layout()
        plt.show()
    
    return delta_mass_df,delta_mass_df_obs

def mass_time_variation(col_values_q,col_values_ssc,
                        sim_dict,dt,
                        col_mass,col_mass_exch):
    
    '''
    this function calculates the mass variation during the events
    IN
    col_values_q,col_values_ssc: columns names of the data
    sim_dict,dt: dictionary with the simulated and observed data for each event
    dt: timestep in seconds
    col_mass,col_mass_exch: name of the columns to set up
    window_length_ratio,polyorder_ratio: parameters for Savgol filter

    OUT
    delta_mass_time_dict: for each event the timeseries of the mass differences downsteram-upstream
    df_mass_exchanged: EMER INdex, exchanged/budget
    '''
    list_sim=list(sim_dict.keys())
    df_mass_exchanged=pd.DataFrame(index=list_sim,
                                   columns=[col_mass_exch])
    delta_mass_time_dict={}
    for event_idx in list_sim:
        delta_mass_time_dict[event_idx]={}
        # data from simulation
        event_df=sim_dict[event_idx]
        event_df=clean_dataframe(event_df) # clean the dataframe

        # dataframe to save
        delta_time_mass_df=pd.DataFrame(index=event_df.index,
                               columns=[col_mass])
        # 
        q_in  =event_df[(col_values_q+'_up')].values
        ssc_in=event_df[(col_values_ssc+'_up')].values
        
        q_out  =event_df[(col_values_q+'_down')].values
        ssc_out=event_df[(col_values_ssc+'_down')].values
        
        mass_input =calc_time_mass(q_in ,ssc_in ) #kg/sec 
        mass_output = calc_time_mass(q_out,ssc_out)
        
        delta_mass=mass_output-mass_input
        delta_time_mass_df.loc[:,col_mass]=delta_mass
        #  negative for erosion, positive for deposition
        
        # mass variation/ mass balance
        mass_input_tot =np.sum(mass_input *dt)
        mass_output_tot=np.sum(mass_output*dt)
        
        mass_exchanged=np.sum(abs(delta_mass)*dt)
        ratio_mass_exchanges=abs(mass_output_tot-mass_input_tot)/mass_exchanged
        
        # save to a dataframe
        df_mass_exchanged.loc[event_idx,col_mass_exch]=ratio_mass_exchanges
        
        # --- save in dict---
        delta_mass_time_dict[event_idx]=delta_time_mass_df # kg/ s
    return delta_mass_time_dict,df_mass_exchanged



def plot_mass_time_month(input_dict_massVar_plot,
    delta_mass_time_dict,df_mass_exchange,
    col_mass,col_mass_exch,debug):

    '''
    IN 
    input_dict_massVar_plot: parameters for plotting
    delta_mass_time_dict: time mass variationf for each event
    df_mass_exchange: EMAR Index
    col_mass,col_mass_exch: columnts name

    OUT
    plot the mass exchange flux and the EMER index
    '''

    # --------------- subdivision by month - plot ----------------
    list_month =input_dict_massVar_plot['list_month'     ]
    labelsize   =input_dict_massVar_plot['labelsize'       ]
    labelpad    =input_dict_massVar_plot['labelpad'        ]
    fontsize    =input_dict_massVar_plot['fontsize_month' ]

    month_events = {m: {'flux': {}, 'emer': {}} for m in list_month}
    list_sim=list(delta_mass_time_dict.keys()) # bprepare which events to be iterated
    
    # sort the results by month
    for event_idx in list_sim:
        month = event_idx.month
        month_name = get_month(month,list_month)

        df_delta_mass=delta_mass_time_dict[event_idx].copy()
        flux=df_delta_mass.loc[:,col_mass].astype('float') # this is a vector 
        flux = flux.interpolate(method='linear', limit_direction='both').dropna()

        df_mass_exchange[col_mass_exch] = pd.to_numeric(df_mass_exchange[col_mass_exch], errors='coerce')
        df_mass_exchange=clean_dataframe(df_mass_exchange)
        emer=df_mass_exchange.loc[event_idx,col_mass_exch] # this is a scalar

        # save to the corresponding month in the dictionary
        month_events[month_name]['flux'][event_idx] = flux
        month_events[month_name]['emer'][event_idx] = emer


    df_sum=pd.DataFrame(index=list_month,columns=['flux','flux_p25','flux_p75',
                                                  'emer','emer_p25','emer_p75','n_event'], dtype=float)
    median_flux_year=[]
    for m in list_month:
        month_dict=month_events[m]
        events_list=(month_dict['flux'].keys())
        N_events=len(events_list)

        flux_month=[]
        emer_month=[]
        
        for event_idx in events_list: # iterate through the event list
            flux=abs(month_dict['flux'][event_idx].copy())

            flux_sum=np.nansum(flux)/len(flux)
            flux_month.append(flux_sum) # calculate the total fluxes of the event
            emer_month.append(month_dict['emer'][event_idx]) # save the emer value for the event

        # make it an array
        flux_month=np.array(flux_month)
        emer_month=np.array(emer_month) 

        median_flux_month=np.median(flux_month)
        median_flux_year.append(median_flux_month) # sum up

        df_sum.loc[m,'emer']=np.nanmedian(emer_month)
        df_sum.loc[m,'emer_p25']=np.nanquantile(emer_month,0.25)
        df_sum.loc[m,'emer_p75']=np.nanquantile(emer_month,0.75)
        df_sum.loc[m,'flux']=median_flux_month
        df_sum.loc[m,'flux_p25']=np.nanquantile(flux_month,0.25)
        df_sum.loc[m,'flux_p75']=np.nanquantile(flux_month,0.75)
        df_sum.loc[m,'n_event']=N_events


    # normalize the fluxes for each month
    median_flux_year=np.array(median_flux_year)
    mean_flux_year=np.mean(median_flux_year)
    mean_emer_year=np.mean(df_sum.loc[:,'emer'].values)

    df_sum.loc[:,'flux']=df_sum.loc[:,'flux']
    df_sum.loc[:,'flux_p25'  ]=df_sum.loc[:,'flux_p25'  ]
    df_sum.loc[:,'flux_p75'  ]=df_sum.loc[:,'flux_p75'  ]
    debug and print(df_sum)

    color= input_dict_massVar_plot['color_month_fluxes' ]
    fill_color = mpl.colors.to_rgba(color, alpha=0.2)

    color_emer= input_dict_massVar_plot['color_month_emer' ]
    fill_color_emer = mpl.colors.to_rgba(color_emer, alpha=0.2)

    # standatrd deviation
    std_up  =df_sum['flux_p25'].values
    std_down=df_sum['flux_p75'].values
    std_up = np.asarray(std_up, dtype=float)
    std_down = np.asarray(std_down, dtype=float)

    # standatrd deviation
    std_emer_up  =df_sum['emer_p25'].values
    std_emer_down=df_sum['emer_p75'].values

    std_emer_up   = np.asarray(std_emer_up  , dtype=float)
    std_emer_down = np.asarray(std_emer_down, dtype=float)

    x=np.arange(len(df_sum))
    fig, (ax1,ax2)= plt.subplots(nrows=2,
        figsize=input_dict_massVar_plot['figsize_month'])
   
   
    # Plot median points and connecting line
    
    ax2.plot(x,df_sum['flux'].values,color=color,
            linestyle='-',marker='o', label=r'$|Q_s|$')
    # Fill between the bounds
    ax2.fill_between(x,std_up,std_down,
                color=fill_color ,label = (r'$|Q_s|$ - IQR'))
    ax2.axhline(y=mean_flux_year, color=color, linestyle=':', linewidth=1,
                label=(r'$|Q_s|$ - average') )


    ax1.plot(x,df_sum['emer'].values,color=color_emer,
            linestyle=input_dict_massVar_plot['ls_month_emer'],
            marker='o',label='EMER index')
    ax1.fill_between(x,std_emer_up,std_emer_down,
                color=fill_color_emer,label='EMER - IQR')
    ax1.axhline(y=1, color=color_emer, linestyle='--', linewidth=1,label='EMER - max')
    ax1.axhline(y=mean_emer_year, color=color_emer, linestyle=':', linewidth=1,label='EMER - average')
    
    ax2.set_xticks(x)
    ax1.set_xticks(x)
    ax2.set_xticklabels(df_sum.index)  # your string month names
                
    # ax2.set_xlabel(input_dict_massVar_plot['xlabel_month'],
    #           fontsize=labelsize, labelpad=labelpad)
    ax2.set_ylabel(input_dict_massVar_plot['ylabel_month'],
              fontsize=labelsize, labelpad=labelpad)
    ax1.set_ylabel(input_dict_massVar_plot['ylabel_emer_month'],
              fontsize=labelsize, labelpad=labelpad,color=color_emer)
    
    ax2.set_title(input_dict_massVar_plot['title_month'],
                     fontsize=fontsize,pad=labelpad*0.5)
    ax1.set_title(input_dict_massVar_plot['title_month_emer'],
                     fontsize=fontsize,pad=labelpad*0.5)
       
    ax2.grid(True,lw=0.3,ls=':',which='both',axis='y')
    ax1.grid(True,lw=0.3,ls=':',which='major',axis='y')

    ax1.set_xticklabels([]) 
     
    # ax.legend(fontsize=labelsize*0.95,loc='upper right', framealpha=1)  # Add legend to show the average line
    ax2.tick_params(axis='both', which='major', labelsize=labelsize*0.95)
    ax1.tick_params(axis='y', which='major', labelsize=labelsize*0.95,
                    colors=color_emer)
    ax1.spines['right'].set_color(color_emer)
    ax2.set_yscale('log')
    # ax1.set_yscale('log')
    
    ax1.set_ylim([0.51,1.01])
    ax2.set_ylim([np.min(df_sum['flux_p25'])*0.95,np.max(df_sum['flux_p75'])*1.05])
    ax1.legend(fontsize=labelsize*0.95)
    ax2.legend(fontsize=labelsize*0.95)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()  

def plot_mass_exchange( df_mass_exchanged, col_mass_exch, 
                       input_dict_massVar_plot):
    '''

    IN
    df_mass_exchanged: dataframe with emer index for each event
    col_mass_exch: name of the column
    input_dict_massVar_plot: parameters to use for plotting
    '''
    figsize=input_dict_massVar_plot['figsize_ratio']
    label_x=input_dict_massVar_plot['label_xaxis']
    label_y=input_dict_massVar_plot['label_yaxis']
    title  =input_dict_massVar_plot['title_ratio']
    bins   =input_dict_massVar_plot['bins_mass_exchange']
    labelpad= input_dict_massVar_plot['labelpad']
    labelsize = input_dict_massVar_plot['labelsize']
    # 

    # df_mass_exchanged=df_mass_exchanged.sort_index()
    
    # Plot mass exchange ratio as a bar plot
    mass_exchange_values = df_mass_exchanged[col_mass_exch].values
    
    log_min = np.log10(np.nanmin(mass_exchange_values[mass_exchange_values > 0]))
    log_max = np.log10(np.nanmax(mass_exchange_values))
    log_bins = np.logspace(log_min, log_max, bins)  

    counts, edges = np.histogram(mass_exchange_values, bins=log_bins,
                                 density=False)
    bin_widths = np.diff(edges)
    bin_mids      = edges[:-1] + bin_widths/2
    pdf = counts / (counts.sum() * bin_widths)

    # Plot with log-scaled x-axis
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(edges[:-1], pdf, width=bin_widths, align='edge',
             color=input_dict_massVar_plot['color_marker_ratio'],
             alpha=0.7,label=input_dict_massVar_plot['label_ratio'])
    ax1.set_xscale('log')

    average_ratio =  np.sum(bin_mids * pdf * bin_widths)
    median_ratio = np.nanmedian(mass_exchange_values)

    # Add a horizontal line for the average ratio
    ax1.axvline(x=average_ratio, color=input_dict_massVar_plot['color_mean_ratio'],
                 linestyle=input_dict_massVar_plot['linestyle_mean'],
                 lw=input_dict_massVar_plot['linewidth_ratio']-0.2,
                 label=input_dict_massVar_plot['label_mean']  )
    ax1.axvline(x=median_ratio , color=input_dict_massVar_plot['color_mean_ratio'],
                 linestyle=input_dict_massVar_plot['linestyle_median'],
                 lw=input_dict_massVar_plot['linewidth_ratio'], 
                 label=input_dict_massVar_plot['label_median'] )
    
    ax1.set_xlabel(label_x,labelpad=labelpad,fontsize=labelsize)
    ax1.set_ylabel(label_y,labelpad=labelpad,fontsize=labelsize)
    ax1.tick_params(axis='both', labelsize=labelsize*0.95)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # Grid settings
    ax1.grid(True, linestyle=':',axis='x',which='both', alpha=0.3)
    # Titles and legends
    # plt.title(title)
    ax1.legend(loc='best',framealpha=1,fontsize=labelsize*0.95)

    plt.show()    
     

def plot_mass_classification_yearly(dt, event_class_dict,
                                    input_dict_massVar_plot,
                                    delta_mass_time_dict, col_mass,
                                    df_mass_exchanged, col_mass_exch):
    '''
    IN
    dt: timestep in seconds
    event_class_dict: dictionary with the events and which variable class they belong to
    input_dict_massVar_plot: parameters to use for plotting
    delta_mass_time_dict: dictionary with the mass variations during the events
    col_mass: column name
    df_mass_exchanged: DataFrame with exchanged mass
    col_mass_exch: column name for exchanged mass
    debug: boolean

    OUT
    A single bar plot (aggregated yearly) comparing N_events and mass_budget across variable classes
    '''

    labelsize = input_dict_massVar_plot['labelsize']
    fontsize  = input_dict_massVar_plot['fontsize_month']
    ylim      = input_dict_massVar_plot['ylim_season_var']
    colors_variables = input_dict_massVar_plot['color_variables']
    label_list       = input_dict_massVar_plot['label_list']

    list_sim = list(delta_mass_time_dict.keys())
    list_var = list(event_class_dict.keys())

    # ---- Collect results across all events (no seasonal subdivision) ----
    var_events_collect = {
        v: {'flux': [], 'emer': [], 'mass_budget': [], 'N_events': 0}
        for v in list_var
    }

    totals = {'mass_budget': 0.0, 'N_events': 0}

    # Mapping from event to variable
    event_to_var = {}
    for var_name in list_var:
        events = event_class_dict[var_name].index
        for ev in events:
            if ev in list_sim:
                event_to_var[ev] = var_name

    # ---- Loop through events and accumulate ----
    for event_idx in list_sim:
        flux = delta_mass_time_dict[event_idx].loc[:, col_mass]  # [kg/sec]
        mass_budget = abs(np.nansum(flux * dt))                  # [kg]
        emer = df_mass_exchanged.loc[event_idx, col_mass_exch]
        var_name = event_to_var.get(event_idx)

        var_events_collect[var_name]['flux'].append(np.nansum(abs(flux)))
        var_events_collect[var_name]['emer'].append(emer)
        var_events_collect[var_name]['mass_budget'].append(mass_budget)
        var_events_collect[var_name]['N_events'] += 1

        totals['mass_budget'] += mass_budget
        totals['N_events'] += 1

    # ---- Compute normalized values and medians ----
    df_year = pd.DataFrame(np.nan,
                           index=list_var,
                           columns=['mass_budget', 'flux', 'emer', 'N_events'])

    for v in list_var:
        flux_list = var_events_collect[v]['flux']
        emer_list = var_events_collect[v]['emer']
        mass_list = var_events_collect[v]['mass_budget']
        n_ev = var_events_collect[v]['N_events']

        median_flux = np.nanmedian(flux_list)
        median_emer = np.nanmedian(emer_list)
        sum_mass_budget = np.nansum(mass_list)

        df_year.loc[v, 'flux'] = median_flux
        df_year.loc[v, 'emer'] = median_emer

        if totals['mass_budget'] > 0 and not np.isnan(sum_mass_budget):
            df_year.loc[v, 'mass_budget'] = sum_mass_budget / totals['mass_budget']

        if totals['N_events'] > 0:
            df_year.loc[v, 'N_events'] = n_ev / totals['N_events']


    # ---- PLOT ----
    N = len(list_var)
    fig, ax = plt.subplots(figsize=input_dict_massVar_plot['figsize_season_var'])

    x = np.arange(N)
    width = 0.3

    n_events_vals = df_year['N_events'].values.astype(float)
    mass_budget_vals = df_year['mass_budget'].values.astype(float)

    for j, v in enumerate(list_var):
        color = colors_variables[v]

        # Full color for N_events
        ax.bar(x[j] - width/2, n_events_vals[j],
               width=width, color=color, label='N. event' if j == N-1 else "")

        # Pale color for mass_budget
        ax.bar(x[j] + width/2, mass_budget_vals[j],
               width=width, color=color, alpha=0.5, label=r'$\Delta M$' if j == N-1 else "")

    ax.set_xticks(x)
    xlabels = [label_list[v] for v in list_var]
    ax.set_xticklabels(xlabels, rotation=0, ha='center', fontsize=labelsize-1)

    ax.tick_params(axis='y', labelsize=labelsize-2)
   
    ax.set_title('Yearly ', fontsize=fontsize+1)
    ax.grid(lw=0.3, ls=':', axis='y')
    ax.axhline(0, color='black', linewidth=0.9, linestyle='-')
    ax.set_ylim(ylim[0],ylim[1]-0.1)
    ax.set_yticks([0.1, 0.3, 0.5])
    ax.legend(fontsize=labelsize-1)

    plt.tight_layout(pad=2)
    plt.show()



def plot_mass_classification(dt,event_class_dict,
                        input_dict_massVar_plot,
                        delta_mass_time_dict,col_mass,
                        df_mass_exchanged,col_mass_exch,
                        debug):
    '''
    IN
    dt: timestep in seconds
    event_class_dict: dictionary with the events and to which variable class they are part of
    input_dict_massVar_plot: paramteres to use for plotting
    delta_mass_time_dict: dicitopnary with the mass variations during the even
    col_mass: column name

    OUT
    plot seaosn and classificaiton 

    '''
    # --------------- subdivision by month - plot ----------------
    list_season=input_dict_massVar_plot['list_season']

    labelsize   =input_dict_massVar_plot['labelsize'       ]
    labelpad    =input_dict_massVar_plot['labelpad'        ]
    fontsize    =input_dict_massVar_plot['fontsize_month' ]

    # list to iterate through
    list_sim=list(delta_mass_time_dict.keys())
    list_var=list(event_class_dict.keys())

    # dictionary to collect the events and variable
    var_events_collect = {
        s: {v: {'flux': [], 'emer': [], 'mass_budget': [], 'N_events': 0} 
            for v in list_var} 
        for s in list_season
    }

    # tracking dictionary
    totals_season = {
        s: {'mass_budget': 0.0,
             'N_events': 0,
             'flux':0.0,
             'emer':0.0} for s in list_season
    }

    # mapping dicitonary
    event_to_var = {}
    for var_name in list(event_class_dict.keys()):
        events=event_class_dict[var_name].index
        for ev in events:
            if ev in list_sim: # check the sim is in the cathegorized list
                event_to_var[ev] = var_name

    # sort results by season
    for event_idx in list_sim:

        # select season  
        month = event_idx.month
        season = get_season(month)
        
        # extract flux,emer and event mass budget
        flux=delta_mass_time_dict[event_idx].loc[:,col_mass]# qs [kg/sec]  this is a vector  
        mass_budget=abs(np.nansum(flux*dt)) #  qs [kg/sec] * dt [sec] this is the evenf final budget
        emer=df_mass_exchanged.loc[event_idx,col_mass_exch]
        

        # sort the event based on the FPCA classification
        var_name = event_to_var[event_idx]  

        # accumulate into DataFrame
        var_events_collect[season][var_name]['flux'].append(np.nansum(abs(flux))) # [kg/ sec m ]current definition of flux
        var_events_collect[season][var_name]['emer'].append(emer)
        var_events_collect[season][var_name]['mass_budget'].append(mass_budget)
        var_events_collect[season][var_name]['N_events'] += 1

        # accumulate totals for the whole season
        totals_season[season]['mass_budget'] += mass_budget
        totals_season[season]['N_events'] += 1

    # -------------------- prepare dict to save info --------------------------------------------
    var_events_dict = { s: pd.DataFrame( np.nan,
                                         index=list_var,
                                           columns=['mass_budget', 'flux', 'emer', 'N_events']) 
                                           for s in list_season}
    # compute and normalize
    for s in list_season:
        n_events = totals_season[s]['N_events']
        season_mass_budget = totals_season[s]['mass_budget']

        for v in list_var:
            flux_list = var_events_collect[s][v]['flux']
            emer_list = var_events_collect[s][v]['emer']
            mass_list = var_events_collect[s][v]['mass_budget']
            n_ev = var_events_collect[s][v]['N_events']

            # median flux and emer
            median_flux = np.nanmedian(flux_list) 
            median_emer = np.nanmedian(emer_list) 
            sum_mass_budget = np.nansum(mass_list)


            # write to yDataFrame
            var_events_dict[s].loc[v, 'flux'] = median_flux
            var_events_dict[s].loc[v, 'emer'] = median_emer

            # normalised mass_budget and N_events
            if season_mass_budget > 0 and not np.isnan(sum_mass_budget):
                var_events_dict[s].loc[v, 'mass_budget'] = sum_mass_budget / season_mass_budget

            if n_events > 0:
                var_events_dict[s].loc[v, 'N_events'] = n_ev / n_events
                # 
        # get some useful information
        debug and print(s)
        debug and print('n_events',n_events,
              '\nbudget [ton]',season_mass_budget/1000)
        debug and print(s,var_events_dict[s],'\n')
       
    # ------------------ PLOT --------------------------------------
    colors_variables = input_dict_massVar_plot['color_variables' ]
    label_list       = input_dict_massVar_plot['label_list'      ]
    ylim             = input_dict_massVar_plot['ylim_season_var' ]

    N=len(list_var)
    fig, axes = plt.subplots(2, 2, 
                             figsize=input_dict_massVar_plot['figsize_season_var'])

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Iterate over the seasons and plot each one in its own subplot
    for i, season in enumerate(list_season):
        ax = axes[i]

        df = var_events_dict[season]

        # x-axis: variables
        x = np.arange(len(list_var))

        # heights
        n_events_vals = df['N_events'].values.astype(float)
        mass_budget_vals = df['mass_budget'].values.astype(float)

        # width for grouped bars
        width = 0.3

        for j, v in enumerate(list_var):
            color = colors_variables[v]

            # full color for N_events
            ax.bar(x[j] - width/2,
                n_events_vals[j],
                width=width,
                color=color,
                label='N. event' if j == N-1 else "")

            # pale color for mass_budget
            ax.bar(x[j] + width/2,
                mass_budget_vals[j],
                width=width,
                color=color,
                alpha=0.5,
                label=r'$\Delta M$' if j == N-1 else "")

        ax.set_xticks(x)
        xlabels = [label_list[v] for v in list_var]
        ax.set_xticklabels(xlabels, rotation=0, ha='center', fontsize=labelsize-1)

       
        ax.tick_params(axis='y', labelsize=labelsize-2)
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_title(season, fontsize=fontsize+1)
        ax.grid(lw=0.3,ls=':',axis='y')
        ax.axhline(0, color='black', linewidth=0.9, linestyle='-')
        ax.set_ylim(ylim)
        ax.legend(fontsize=labelsize-1)

    plt.tight_layout(pad=2)
    plt.show()


# --------------------------------------------------------------------
#               emd event scale
# ------------------------------------------------------------------
def save_df(df2save,input_file_dict,debug):
    file_output=(input_file_dict['path_output']
                 +input_file_dict['file_name'])
    if not debug:
        print('save: ',input_file_dict['file_name'],'\n')
        with open(file_output, 'wb') as f:
            pickle.dump(df2save, f)
            
            
def import_df(input_dataframe,debug):
    file_path=input_dataframe['path_input']
    file_name=input_dataframe['file_name']
    print(file_name)
    full_file_path = (file_path+ file_name)
    
    # Load the pickle file
    with open(full_file_path, 'rb') as f:
        import_df = pickle.load(f)
        
    debug and print('import  dataframe: ',file_name,'\n')
    return import_df
            
def prepare_emd_target(col_q,col_ssc,sim_dict,list_events,
                       window_length,polyorder,target_columns,
                       time_step_downscale,method_downscale,
                       debug):
    
    # dict to save targets
    target_dict={}
    
    for event_idx in list_events:
        target_dict[event_idx]={}
        
        df=sim_dict[event_idx].copy()
        
        # smooth
        df_smooth = df.apply(
            lambda x:savgol_filter(x, window_length, polyorder),
            axis=0)
        
        # interpolate
        df_interp=df_smooth.resample(
            time_step_downscale).interpolate(
            method=method_downscale)
    
        # dataframe to save data
        df_target=pd.DataFrame(index=df_interp.index,
                               columns=target_columns)
        
        
        # target1 : calculate the ssc difference
        ssc_in =df_interp[(col_ssc+'_up')].values
        ssc_out=df_interp[(col_ssc+'_down')].values
        delta_ssc=ssc_out-ssc_in        
        
        # target 2: mass
        q_in =df_interp[(col_q+'_up')].values
        q_out=df_interp[(col_q+'_down')].values
        delta_q=q_out-q_in
        
        mass_in=calc_time_mass(q_in,ssc_in) # kg/s
        mass_out=calc_time_mass(q_out,ssc_out)
        delta_mass=(mass_out-mass_in) # kg/s
        
        # save in dataframe
        df_target[target_columns[0]] = delta_ssc
        df_target[target_columns[1]] = delta_q
        df_target[target_columns[2]] = delta_mass
        # save in the dictionary
        target_dict[event_idx]=df_target
        
    return target_dict

def emd_HHT_analysis(target_dict,target_columns,list_events,
    max_timestep,min_timestep,min_resolution,N_bins,scale_freq,
    eps,max_imfs,sift_thresh,return_residual,imf_opts,
    amplitude_threshold,timestep,method_hht,result_columns,
    debug):
    
    # define the grid and the range of possible frequencies
    freq_edges,freq_centres=emd.spectra.define_hist_bins(data_min=1/max_timestep,
                                                         data_max=1/min_timestep,
                                                         nbins=N_bins,
                                                         scale=scale_freq)
    
    # columns of variables to save
    col_power    =result_columns[0]
    col_timescale=result_columns[1]
    # dictionary to save results
    HHT_power_dict={}
    for event_idx in list_events: # iterate through the events
        HHT_power_dict[event_idx]={}
        
        for col in target_columns:
            HHT_power_dict[event_idx][col]={}
            
            # target variables
            df_target=target_dict[event_idx].copy()
            tot_event_duration=(df_target.index[-1]-df_target.index[0]).total_seconds() #total event duration
            target_vect=df_target[col].values
            
            # ------------ extremes boundary handling Stallone,Cicone,Materassi 2020 ----
            # 1. subtract the mean value
            m=np.mean(target_vect)
            target_vect_norm=target_vect-m
            
            # 2. mirroring (signal 5 times longer)
            original_len = len(target_vect_norm)
            init=int(round(original_len*2.1))
            end =int(round(original_len*2.9))
            # to crop to a shorter extend to eliminate mirroring flat ends
            
            mirrored_signal = np.concatenate([target_vect_norm,target_vect_norm[::-1], 
                                              target_vect_norm, 
                                              target_vect_norm[::-1],target_vect_norm])

            # 3. chi characteristic function
            alpha_decay=np.log(eps)
            middle=np.ones(original_len)
            decay_after=np.exp(alpha_decay*np.linspace(0,1,2*original_len))
            increase_before=decay_after[::-1]
            
            chi_weights=np.concatenate([increase_before,middle,decay_after])
            signal2emd=chi_weights*mirrored_signal+m # new signal to emd
            
            # Perform EMD on the extended signal
            imfs = emd.sift.sift(signal2emd,
                             max_imfs=max_imfs,
                             sift_thresh=sift_thresh,
                             return_residual=return_residual,
                             imf_opts=imf_opts)
            if debug:
                init=original_len*2
                end=original_len*3
                # Trim back to original length
                imfs_test = imfs[init:end, :]
                print(
                    f'{event_idx} rmse:', np.sqrt(np.mean((
                        target_vect - np.sum(imfs_test, axis=1)) ** 2)))
                
                fig, axs = plt.subplots(1, 1, figsize=(15, 30))
                # # Create colormap object
                cmap = plt.cm.turbo
                emd.plotting.plot_imfs(imfs_test,df_target.index,X=target_vect,
                       cmap=cmap,sharey=True, xlabel='Time (years)',step=5,ax=axs)
                plt.show()
                
            # ----------------------------------------------------
            #           HILBER-HUANG TRANSFORM
            # ---------------------------------------------------
            # eliminate the trend
            imfs=imfs[:,:-1]
            
            # --- save dataframe ---
            n_imfs=imfs.shape[1]
            list_imf=[f'imf_{i+1}' for i in range(n_imfs)] 
            df2save=pd.DataFrame(index=list_imf,
                                 columns=[col_power, col_timescale]) 
            
            # 1. calculate the relative energy of each imf
            tot_power=0 
            for i in range(n_imfs):
                imf=imfs[:,i]
                imf_name=df2save.index[i]
                
                # analytical signal
                imf_analytic=hilbert(imf)
                imf_analytic_original=imf_analytic[init:end] # compute the power on original length
                # power
                power=np.sum((np.abs(imf_analytic_original))**2) # imf power
                tot_power += power
                
                df2save.loc[imf_name,col_power]=power
         
            # normalize over the total power
            df2save.loc[:,col_power]=df2save.loc[:,col_power]/tot_power
            
            # ----------------------------------------------------------
            # calculate the frequency content
            for i in range(n_imfs):
                imf=imfs[:,i]
                imf_name=df2save.index[i]
                norm_amplitude=df2save.loc[imf_name,col_power]
                
                # --> first check point
                # check that the imf contributes significantly to the 
                # energy of the variable part of the signal
                if norm_amplitude< amplitude_threshold:
                    continue
                
                # compute istantaneous frequency and amplitude
                _, freq_i, _ = emd.spectra.frequency_transform(imf, 
                                                            1/timestep,
                                                            method=method_hht)
                # 
                freq_i=freq_i.ravel() # convert to just 1D array
                
                # crop to original extent
                freq_i_orig=freq_i[init:end]
                
                # filter out frequencies that are zero
                freq_i_orig=freq_i_orig[freq_i_orig>0] 
                median_freq= np.nanmedian(freq_i_orig)
                    
                if debug: # Plot the cumulative distribution function
                    hist,_=np.histogram(freq_i_orig,bins=100)
                    # 
                    plt.figure(figsize=(10, 6))
                    plt.plot(freq_centres, hist,'k', label='cdf norm')
                    plt.axvline(median_freq, color='r', linestyle='-', label='Median Frequency')
                    plt.xlabel('Frequency')
                    plt.ylabel('pdf')
                    plt.title(('CDF of '+imf_name))
                    plt.xscale('log')
                    plt.legend()
                    plt.grid(True,lw=0.15)
                    plt.show()
                
                #
                time_scale_i=1/median_freq
                # --> second check point
                # Nyquist ratio + at least one revolution inside my event duration
                if (time_scale_i< min_resolution*2) or (time_scale_i > tot_event_duration): #<<<<<
                    # debug and print(imf_name,' numerical artifact')
                    # if the imf has a median freq lower than the data reolution
                    # than is an artifact of the downscaling
                    continue
                df2save.loc[imf_name,col_timescale] = time_scale_i
                
            # delete the lines that have not been saved either bc of amplitude or for frequency
            df2save=df2save.dropna()
            # save in dictionary
            HHT_power_dict[event_idx][col] = df2save
            
    # save in folder
    return HHT_power_dict

def plot_HHT_analysis(HHT_power_dict_import,list_events,
                      target_columns,target_col,result_columns,
                      time_scale_centers,title,amplitude_threshold,
                      max_cs,max_pdf_freq                     
                      ):
    
    bin_edges = []
    for i, center in enumerate(time_scale_centers):
        if i == 0:
            # First edge: extend half the gap below the first center
            left_edge = center - 0.5*(time_scale_centers[i+1] - center)
            bin_edges.append(left_edge)
        else:
            # Midpoint between previous center and current center
            midpoint = 0.5*(time_scale_centers[i] + time_scale_centers[i - 1])
            bin_edges.append(midpoint)

    # Finally, extend half the gap above the last center
    last_edge = time_scale_centers[-1] + 0.5*(time_scale_centers[-1] - time_scale_centers[-2])
    bin_edges.append(last_edge)
    bin_edges = np.array(bin_edges)
    
    col_power=result_columns[0]
    col_timescale=result_columns[1]
     # Create a figure with one subplot per variable
    fig, axes = plt.subplots(nrows=len(target_columns), ncols=1,
                             figsize=(8, 4*len(target_columns)),
                             sharex=False, sharey=False)
    
    for i, col in enumerate(target_columns):
        # Gather timescale & power from all events
        all_times = []
        all_powers = []
        for event_idx in list_events:
            df = HHT_power_dict_import[event_idx][col].copy()
            times = df[col_timescale].values
            powers = df[col_power].values
            
            all_times.append(times)
            all_powers.append(powers)
        
        # Concatenate
        all_times = np.concatenate(all_times) if all_times else np.array([])
        all_powers = np.concatenate(all_powers) if all_powers else np.array([])
        
        # 1) Histogram of counts (unweighted)
        
        hist_counts, _ = np.histogram(all_times,
                                      bins=bin_edges, density=False)
        
       
        # 2) Histogram of total power (weighted)
        hist_power, _ = np.histogram(all_times, bins=bin_edges,
                                     weights=all_powers, density=False)
        
      
        # 3) Compute average power in each bin
        
        avg_power = np.zeros_like(hist_counts, dtype=float)
        nonzero_mask = (hist_counts > 2)
        avg_power[nonzero_mask] = hist_power[nonzero_mask] / (hist_counts[nonzero_mask]-1) # unbiased average
        
    
        # 4) Map average power to colors
        
        cmap = target_col[col]
        # Normalize from min to max average power
        vmin, vmax = amplitude_threshold, max_cs
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
  
        # Convert timescale to hours for plotting
        bin_edges_hours = bin_edges / 3600.0
        widths = np.diff(bin_edges_hours)  # bar widths in hours
        
        # Build a list of colors for each bin
        bin_colors = []
        for j in range(len(hist_counts)):
            # If hist_counts=0 => avg_power=0 => color can be gray or something
            if hist_counts[j] == 0:
                bin_colors.append("lightgray")
            else:
                # Scale avg_power to [0,1] via norm, then map to colormap
                color_val = norm(avg_power[j])
                bin_colors.append(cmap(color_val))
                
        total_events = np.sum(hist_counts)
        normalized_hist_counts = hist_counts / total_events
        
        # -----------------------------------------------------------------
        # 5) Plot the unweighted histogram with colored bars
        # -----------------------------------------------------------------
        ax = axes[i]
        left_edges_hours = bin_edges_hours[:-1]
        
        ax.bar(left_edges_hours, normalized_hist_counts,
               width=widths, align='center',
               color=bin_colors, edgecolor=None, alpha=0.8)
        if i==2:
            ax.set_xlabel('Timescale [Hours]')

        ax.set_ylabel('pdf')
        ax.set_title(f'{title[col]}: Event - IMF distribution ')
        ax.grid(True, lw=0.3)
        
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # dummy array for color scale
        cbar = plt.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(r'$\mathbf{S_{rel}}$ ')
        ax.set_ylim(0.0,max_pdf_freq)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xscale('log')
        
    
    plt.tight_layout()
    plt.show()


def plot_HHT_var_season(event_class_dict,
                        list_season,list_events,
                        HHT_power_dict,
                        input_dict_emd,
                        result_columns,
                        debug
                        ):
    
    '''
    plot the timescales for each variable and for each season
    IN
    event_class_dict: for each variable there is a dictionary of the events
    list_season,list_events: list of the seasons list of the events
    HHT_power_dict: dictionary with the EMD decomposition the the Hilbert Huang Transform analysis
    input_dict_emd: parameters for the plot
    result_columns: columns to which apply the results

    OUT
    the plot
    '''
    
    var2plot=input_dict_emd['var2plot']
    # classify for leading variable
    list_var= list(event_class_dict.keys())
   
    hht_season_var_dict={}
    #  prepare dictionary
    for var in list_var:
        hht_season_var_dict[var]={}
        for season in list_season:
            hht_season_var_dict[var][season]=[]
    
    # classify the events
    for var in list_var:
        events_var_list=event_class_dict[var].index
        # 
        for event_idx in list_events:
            month = event_idx.month
            season = get_season(month)
            # check if the event is in the list of 
            # events by whcih the var is the leading
            
            if event_idx in events_var_list:
               df=HHT_power_dict[event_idx][var2plot].copy()
               hht_season_var_dict[var][season].append(df)  
   
    for var in list_var:
        for season in list_season:
            df_list = hht_season_var_dict[var][season]
            if df_list:  # If not empty
                combined_df = pd.concat(df_list)
                hht_season_var_dict[var][season] = combined_df
    
    # --------------  PLOT --------------------------------
    # Unpack the columns for power, timescale
    col_power = result_columns[0]
    col_timescale = result_columns[1]
    min_num=input_dict_emd['min_imf_num']
    quant_high=input_dict_emd['quantile_high']
    quant_low=input_dict_emd['quantile_low']
    # 1) Convert bin centers -> bin edges (same as your code)
    bin_edges = []
    time_scale_centers=input_dict_emd['time_scale_centers_var']
    for i, center in enumerate(time_scale_centers):
        if i == 0:
            left_edge = center - 0.5*(time_scale_centers[i+1] - center)
            bin_edges.append(left_edge)
        else:
            midpoint = 0.5*(time_scale_centers[i] + time_scale_centers[i-1])
            bin_edges.append(midpoint)
            
    last_edge = time_scale_centers[-1] + 0.5*(time_scale_centers[-1] - time_scale_centers[-2])
    bin_edges.append(last_edge)
    bin_edges = np.array(bin_edges)
    
    color_maps = input_dict_emd['colors'             ]
    label_list = input_dict_emd['label'              ]
    ymax       = input_dict_emd['max_errorplot_yaxis']
    figsize    = input_dict_emd['figsize'            ]
    labelsize  = input_dict_emd['labelsize'          ]
    linewidth  = input_dict_emd['linewidth'          ]

    # Create figure with 4 subplots: one per season
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.ravel()  # Flatten 2x2 into 1D array of axes
    for season_idx, season in enumerate(list_season):
        ax = axes[season_idx]
        pos=ymax-1.9
        med_val=[] # calculate the season average

        for j,var in enumerate(list_var):
            df = hht_season_var_dict[var][season].copy()

            # safety check for missing data
            if isinstance(df, list) or df is None or len(df) == 0:
                debug and print(f"Season: {season}, Variable: {var}, Data points: 0 (skipped)")
                continue  # skip this var/season because no data

            # print the count of points for debugging
            print(f"Season: {season}, Variable: {var}, Data points: {len(df)}")
            # Gather timescale and power values from the DataFrame
            times = df[col_timescale].values 
            powers = df[col_power].values
        
            qlow =np.nanquantile(times/3600,quant_low)
            qhigh=np.nanquantile(times/3600,quant_high)
            median = np.nanmedian(times)/3600
            med_val.append(median)
            times=times/3600
            
            # Filter timescales within [q_low, q_high]
            mask = (times >= qlow) & (times <= qhigh)
            times_filtered  = times [mask]
            powers_filtered = powers[mask]
            # 'non_explained'
            n=2
            if (len(times_filtered)>n and len(powers_filtered)>n):
                color = color_maps.get(var)
                if len(df) <= min_num:
                    color=mcolors.to_rgba(color,alpha=0.2)

                # Plot errorbar
                ax.errorbar(
                    x=median, 
                    y=pos, 
                    xerr=[[median       - qlow], [qhigh - median      ]],
                    fmt='o', color=color, ecolor=color, 
                    elinewidth=linewidth, capsize=5, capthick=2, label=label_list[var]
                )
            
                # prepare well the inputs for the KDE
                times_filtered = np.asarray(times_filtered , dtype=float)
                powers_filtered = np.asarray(powers_filtered, dtype=float)
                # Define refined x_grid in the same extent as time_scale_centers
                x_min, x_max = np.min(time_scale_centers) / 3600, np.max(time_scale_centers) / 3600
                x_grid = np.linspace(x_min, x_max, 200) 
                # KDE representation
            
                kde = gaussian_kde(times_filtered, weights=powers_filtered, bw_method='scott')
                kde_vals = kde(x_grid)
                # Normalize to [0,1] 
                kde_vals = kde_vals / np.nanmax(kde_vals)
                
                x_mask=(x_grid > qlow) & (x_grid < qhigh)
                ax.fill_between(
                    x_grid[x_mask], pos, pos + kde_vals[x_mask],
                    color=color, alpha=0.2,
                    label=(r'$\mathbf{S_{rel}}$ ' + label_list[var])
                )

                # Optionally also draw the KDE line for clarity
                ax.plot(x_grid[x_mask], pos + kde_vals[x_mask], color=color, lw=linewidth*0.8)
                # title=('IMFs min number: '+str(min_num)),
                legend=ax.legend(loc='center right',fontsize=labelsize)   
                legend.get_title().set_fontsize(labelsize)
                ax.set_title((r'$\Delta M(t)$ timescales in ' + season),fontsize=labelsize+2)
                pos-=2 # increase the position of one

        med_val=np.array(med_val)
        ax.vlines(x=med_val.mean(), ymin=-0.2,ymax=ymax, 
                  colors='k',linestyle='dotted',linewidth=linewidth*0.5,zorder=-1)
        vline_proxy = mlines.Line2D([], [], color='k', linestyle='dotted', linewidth=linewidth*0.5, label='avg. timescale')

        # Add it to the legend
        legend = ax.legend(handles=ax.get_legend_handles_labels()[0] + [vline_proxy],
                            loc='center right', fontsize=labelsize)
        legend.get_title().set_fontsize(labelsize)
    for ax in axes:
        # ax.set_xscale('log')
        ax.set_xlabel('Timescale (hours)',fontsize=labelsize)
        ax.grid(lw=0.15,which='both',axis='x')
        ax.set_ylim(-0.1,ymax)
        ax.set_xlim((2,39))
        ax.tick_params(axis='x', labelsize=labelsize-1)  
        ax.set_yticks([])  # Remove y-axis ticks
    plt.tight_layout(pad=2)
    plt.show()
    

def emd_event(col_q,col_ssc,
                sim_dict,
                event_class_dict,
              input_dict_FPCA,
              input_sim_station_difference,
              input_dict_emd,
              list_season,
              debug):
    #  -----------------------------------------------------------
    #                       PREPARE TARGET
    # ------------------------------------------------------------
    target_columns      =input_dict_emd              ['target_columns']
    target_col          =input_dict_emd              ['target_colors']
    window_length       =input_dict_FPCA              ['window_size']
    polyorder           =input_dict_FPCA              ['polyorder']
    time_step_downscale =input_sim_station_difference['time_step_downscale']
    method_downscale    =input_dict_emd               ['method_downscale']
    list_events         =list                         (sim_dict.keys())
    # 
    #  prepare the target smooth and interpolate
    target_dict=prepare_emd_target(col_q,col_ssc,sim_dict,list_events,
                                   window_length,polyorder,target_columns,
                                   time_step_downscale,method_downscale,
                                   debug)

    #  -----------------------------------------------------------
    #            EMD DECOMPOSITION & HILBERT-HUANG TRANSFORM
    # ------------------------------------------------------------
    imf_opts       =input_dict_emd['imf_opts'       ]
    max_imfs       =input_dict_emd['max_imfs'       ]
    sift_thresh    =input_dict_emd['sift_thresh'    ]
    return_residual=input_dict_emd['return_residual']
    # Hilbert analysis
    EMD_calculate=input_dict_emd['EMD_calculate'       ]
    N_bins       =input_dict_emd['nbins_HHT'        ]
    min_timestep =input_dict_emd['min_timestep_sec' ]
    timestep = min_timestep
    max_timestep        = input_dict_emd['max_timestep_sec'          ]
    scale_freq          = input_dict_emd['scale_grid'                ]
    min_resolution      = input_dict_emd['min_resolution_sec'        ]
    method_hht          = input_dict_emd['method_HHT'                ]
    amplitude_threshold = input_dict_emd['amplitude_share_threshold' ]
    eps                 = input_dict_emd['eps_boundaries'            ]
    result_columns      = input_dict_emd['result_columns']
    # 
    if EMD_calculate==True:
        # performe the HHT decomposition
        HHT_power_dict=emd_HHT_analysis(target_dict,target_columns,list_events,
                        max_timestep,min_timestep,min_resolution,N_bins,scale_freq,
                        eps,max_imfs,sift_thresh,return_residual,imf_opts,
                        amplitude_threshold,timestep,method_hht,result_columns,
                        
                        debug=False)
        # save and re-import
        save_df(HHT_power_dict,input_dict_emd,False)
        
    HHT_power_dict=import_df(input_dict_emd,debug) # if you want just to import it in a second moment

    # -----------------------------------------------------------------------------
    #               SIMPLE PLOT FOR EACH VARIABLE
    # -----------------------------------------------------------------------------
    
    time_scale_centers=input_dict_emd['time_scale_centers'  ]
    title_list        =input_dict_emd['title_conversion'    ]
    max_cs            =input_dict_emd['max_power_colorscale']
    max_pdf_freq      =input_dict_emd['max_pdf_freq'        ]

    plot_HHT_analysis(HHT_power_dict,list_events,
                      target_columns,target_col,result_columns,
                      time_scale_centers,title_list,amplitude_threshold,
                      max_cs,max_pdf_freq)
    
    # -------------------------------------------------------------------------------
    #              HIST of SIGNAL POWER SHARE - SEASON - LEADING VARIABLE
    # -------------------------------------------------------------------------------
    
    plot_HHT_var_season(event_class_dict,
                        list_season,list_events,
                        HHT_power_dict,
                        input_dict_emd,
                        result_columns,
                        debug)



def mass_classification(station_couple,
                        q_dict_data, ssc_dict_data_10m,
                        rel_mass_df,sim_dict,
                        event_variable_df,
                        input_FPCA_variables,
                        # 
                        input_sim_station_difference,
                        input_dict_FPCA,
                        input_dict_massVar_plot,
                        input_dict_emd,
                        debug):
    '''
    station_couple: list with the two stations: upstream-downstream
    q_dict_data: dictionary with the discharge data for each station
    ssc_dict_data_10m: dictionary with the concentration data for each station
    rel_mass_df: dataframe with the events, the relative mass difference and the error ( sim - obs)
    sim_dict: dictionary with all simulations and observed data at the downstream station

    event_variable_df: FPCA results with R2, init and end date, and weights for each variable ( SSC)
    input_FPCA_variables: event input and target, real values
    '''
    
    
    # analyze the classified events, what drives mass changes 
    debug and print('>> FPCA_variables_classification <<')
    
    event_class_dict=FPCA_variables_classification( event_variable_df,
                                                    input_dict_FPCA,
                                                    debug)

    # # 0. ------------- prepapare data ------------------
    col_values_q=input_dict_FPCA['col_values_q']
    col_values_ssc=input_dict_FPCA['col_values_ssc']

    # time step of the simulation
    time_step=input_sim_station_difference['time_step']
    dt=convert_to_seconds(time_step)
    
    # dataframe from observation - downscale discharge
    q_dict_data_10m={}
    for loc in station_couple:
        station=station_couple[loc]
        q_dict_data_10m[station]={}
        
        q_df=q_dict_data[station].copy()
        q_df_10min = q_df.resample(time_step).interpolate(
            method='linear')
        #  save
        q_dict_data_10m[station]=q_df_10min
    
    
    # 1. --------------------------- TOTAL MASS ----------------------------------
    debug and print('>> total_mass <<')
    mass_station_dict = total_mass(station_couple,
           col_values_q,col_values_ssc,
           q_dict_data_10m,ssc_dict_data_10m,
           sim_dict,dt,
           event_class_dict,
           debug)
    

    # 2. --------------- MASS VARIATION -----------------
    debug and print('>> mass_variation <<')
    delta_mass_df,delta_mass_df_obs =mass_variation(col_values_q,col_values_ssc,dt,
                                rel_mass_df,sim_dict,
                                input_dict_massVar_plot,
                                debug)
   


    # # mass variation during the event
    col_mass='delta_mass_[kg/sm]'
    col_mass_exch='ratio_totMass_exchange'

    debug and print('>> mass_time_variation <<')
    delta_mass_time_dict,df_mass_exchange=mass_time_variation(
                            col_values_q,col_values_ssc,
                            sim_dict,dt,col_mass,col_mass_exch) 


    debug and print('>> plot_mass_time_month<<')
    plot_mass_time_month(input_dict_massVar_plot,
                          delta_mass_time_dict,df_mass_exchange,
                          col_mass,col_mass_exch,debug)
    
    
    # # ------------ variables classification -----------------
    #  classify by season and by delta mass main driver
    debug and print('>> plot_mass_classification <<')
    plot_mass_classification(dt,event_class_dict,
                        input_dict_massVar_plot,
                        delta_mass_time_dict,col_mass,
                        df_mass_exchange,col_mass_exch,
                        debug)
    
    debug and plot_mass_classification_yearly(dt, event_class_dict,
                                    input_dict_massVar_plot,
                                    delta_mass_time_dict, col_mass,
                                    df_mass_exchange, col_mass_exch)
    
    # --------------------------------------------------------------------
    #                   study the event frequencies
    # ---------------------------------------------------------------------
    list_season=input_dict_massVar_plot['list_season']
    debug and print('>> emd_event <<')
    emd_event(col_values_q,col_values_ssc,
                sim_dict,
                event_class_dict,
              input_dict_FPCA,
              input_sim_station_difference,
              input_dict_emd,
              list_season,
              debug)
    
    
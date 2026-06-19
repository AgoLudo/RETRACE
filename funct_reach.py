
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from datetime import timedelta
from scipy.signal import savgol_filter,hilbert,correlate,correlation_lags,find_peaks
import matplotlib.colors as mcolors
from scipy.stats import linregress
from scipy.optimize import minimize_scalar

# 1. ---------------------- DATA IMPORT ---------------------------------
def import_df(input_dataframe,debug):
    file_path=input_dataframe['path_input']
    file_name=input_dataframe['file_name_in']

    full_file_path = os.path.join(file_path, file_name)
   
    # Load the pickle file
    with open(full_file_path, 'rb') as f:
        import_df = pickle.load(f)
    debug and print('import  dataframe: ',file_name,'\n')
    return import_df


def save_df(df2save,input_file_dict,debug):
    file_path=input_file_dict['path_output']
    file_name=input_file_dict['file_name_out']
    
    file_output = os.path.join(file_path, file_name)
    
    if not debug:
        print('save: ',input_file_dict['file_name_out'],'\n')
        with open(file_output, 'wb') as f:
            pickle.dump(df2save, f)





def import_data_stationCouple(input_dict,debug):
    path_dir=input_dict['path_input']
    filename_upStation=input_dict['file_name_upstream']
    filename_downStation=input_dict['file_name_downstream']
    
    upStation=input_dict['upstream_station']
    downStation=input_dict['downstream_station']
    col_time=input_dict['time_values']
    col_values=input_dict['col_values']
    sep=input_dict['sep']
    head=input_dict['head']
    
    window_size=input_dict['window_size']
    polyorder=input_dict['polyorder']
    
    filename_list_station=[filename_upStation,filename_downStation]
    list_station=[upStation,downStation]
    
    print('*** '+upStation+':'+downStation+' ***')
    debug and print('sampling in ', col_values,'\n')

    df_dict={}
    for name,filename in zip(list_station,filename_list_station):
        file_path=os.path.join(path_dir, filename)
        
        df=pd.read_csv(file_path,sep=sep,header=head)
        
        df[col_time]=pd.to_datetime(df[col_time])
        df=df.set_index(col_time)
        df=df.replace([np.inf, -np.inf], np.nan).dropna()
        df_smooth_vect=savgol_filter(df[col_values], window_size,polyorder)  # window_size_points must be odd
        df_data_smooth=df.copy()
        df_data_smooth[col_values]=df_smooth_vect
        
        df_dict[name]=df_data_smooth
    return df_dict

# ----------------------- REACH MORPHOMETRY ---------------------------------
def reach_morphometry(debug):
    
    # channel characteristics
    geom_dict={}
    # width of the reach refers to the downstream station of the reach
    reaches=['Arval_Amont:Arval_Amont','Arval_Amont:Arvan_Aval','Arvan_Aval:Pontamafrey',
             'Pontamafrey:Aiguebelle','Aiguebelle:Montmelian','Montmelian:Grenoble_campus']
    
    variables=['width','slope','distance']
    
    for reach in reaches: 
        geom_dict[reach]={}
        for var in variables:
            geom_dict[reach][var]={}
     
    rDec=3   
    geom_dict['Arval_Amont:Arval_Amont'    ]['width']= np.round(15124.908   /1071.805846 ,rDec)
    geom_dict['Arval_Amont:Arvan_Aval'     ]['width']= np.round(17092.849   /621.338481  ,rDec)
    geom_dict['Arvan_Aval:Pontamafrey'     ]['width']= np.round(35815.191   /845.334191  ,rDec) # use the weir formula as there is a power plant
    geom_dict['Pontamafrey:Aiguebelle'     ]['width']= np.round(89675.049   /2177.777558 ,rDec) # use the weir formula as there is a power plant
    geom_dict['Aiguebelle:Montmelian'      ]['width']= np.round(1015880.118 /8976.71734  ,rDec)
    geom_dict['Montmelian:Grenoble_campus' ]['width']= np.round(430703.825  /6201.639868 ,rDec)
    
    # Slope
    
    geom_dict['Arval_Amont:Arval_Amont'    ]['slope']=np.nan
    geom_dict['Arval_Amont:Arvan_Aval'     ]['slope']= 0.055
    geom_dict['Arvan_Aval:Pontamafrey'     ]['slope']= 0.014
    geom_dict['Pontamafrey:Aiguebelle'     ]['slope']= 0.006 # From Jaballah 2015 (majority of the reach)
    geom_dict['Aiguebelle:Montmelian'      ]['slope']=0.004
    geom_dict['Montmelian:Grenoble_campus' ]['slope']= 0.002 # From Carlin 2021
    #
    
    geom_dict['Arval_Amont:Arval_Amont'    ]['distance']=np.nan
    geom_dict['Arval_Amont:Arvan_Aval'     ]['distance'] = 15396.922580
    geom_dict['Arvan_Aval:Pontamafrey'     ]['distance'] = 2341.930130
    geom_dict['Pontamafrey:Aiguebelle'     ]['distance'] = 31158.549420
    geom_dict['Aiguebelle:Montmelian'      ]['distance'] =(14102.041320+9005.985078)
    geom_dict['Montmelian:Grenoble_campus' ]['distance'] = 49603.532360
    
    if debug:
        for reach in reaches:
            print(reach ,geom_dict[reach]['width'],'m')
            
    return geom_dict

# 2. ------------ DATA HANDLING  ----------------
def data_prep(z1, z2, colName,newfreq='10min',minLength=5):
    """
    Aligns two dataframes to have the same time index, interpolates missing values,
    and ensures their columns are ready for comparison or further processing.

    Parameters:
    z1 (pd.DataFrame): First dataframe with a time index.
    z2 (pd.DataFrame): Second dataframe with a time index.
    colName (str): The column name to process in both dataframes.
    newfreq
    
    Returns:
    list: A list containing the two aligned and interpolated columns [x1, x2].
    """
    # Align the dataframes by joining on their indices
    z12 = z1.join(
        z2, on=None, 
        how='inner', 
        lsuffix='_1', 
        rsuffix='_2', 
        sort=False, 
        validate='one_to_one'
    )
    if len(z12)<minLength:
        x1=np.zeros(len(z12))+np.nan
        x2=x1
        return [x1,x2,z12.index]
    else:
        new_time_index = pd.date_range(
            start=z12.index[0], 
            end=z12.index[-1], 
            freq=newfreq
        )
        
        z12 = z12.reindex(new_time_index)
        
        # Interpolate NaN values for both columns
        z12[f'{colName}_1'] = z12[f'{colName}_1'].interpolate(method='linear', limit_area='inside')
        z12[f'{colName}_2'] = z12[f'{colName}_2'].interpolate(method='linear', limit_area='inside')

        # Extract the aligned columns
        x1 = z12[f'{colName}_1'].values
        x2 = z12[f'{colName}_2'].values

        return [x1,x2,z12.index]

def cross_corr_global(df_dict,
                      geom_dict,
                      input_dict,
                      debug):
    '''
    calculates an average time lag between the two stations. Takes the whole timeseries
    '''

    # to run this function just with the 1h data
    col_values    =input_dict['col_values'        ]
    dt_sec        =input_dict['dt_sec'            ]
    upStation     =input_dict['upstream_station'  ]
    downStation   =input_dict['downstream_station']
    df_upstream   =df_dict   [upStation           ]
    df_downstream =df_dict   [downStation         ]
    
    # join in unique dataframe to have corresponding timesteps
    df12=df_upstream.join(
                df_downstream,
                on=None, 
                how='inner', 
                lsuffix='_up', 
                rsuffix='_down',
                sort=False,
                validate='one_to_one'
                )
    
    x_up=df12[(col_values+'_up')].values
    x_down=df12[(col_values+'_down')].values
    
    # study the cross correlation
    correlation = correlate((x_up-np.mean(x_up))/np.std(x_up),
                            (x_down-np.mean(x_down))//np.std(x_down),
                                mode="full",
                                method='fft')
    lags = correlation_lags(len(x_up), len(x_down), mode="full")
    lag = lags[np.argmax(abs(correlation))]
    
    L=geom_dict['distance']
    cel=L/(np.abs(lag)*dt_sec)
    
    # save 
    df_global=pd.DataFrame(index=[0],columns=['cel_[m/s]','lag_[h]'])
    df_global.loc[0,'cel_[m/s]']=cel
    df_global.loc[0,'lag_[h]']=lag
    
    debug and print(upStation+':'+downStation+
                    '\n avg time lag',lag,'h'+
                    '\n avg cel',cel,'m/s\n')
    
    return df_global

def prepare_df2store_cc(event_dict,debug):
    # events_variable: the variable from which the events were segmented
    columns=['lag_Q', 'lag_Q_hilb',
             'lag_SSC', 'lag_SSC_hilb',
             'init_event','end_event']
    lags_dict={}
    for event in list(event_dict.keys()):
        lags_df=pd.DataFrame(columns=columns,index=[0])
        lags_df.iloc[0,:]=np.nan
        lags_dict[event]=lags_df

    return lags_dict


def cross_corr(x1,x2,
               time_step,topo_prominence,
               height,min_lag,max_lag,
               debug):
    '''
    x1,x2 : arrays
    time_step: of the interpolated values
    
    topo_prominence: topographic prominence of the cross correlation function peak
    height: minimum cross correlation to consider
    min_lag,max_lag: number of lags (it is NOT a time) to bound the cross correlation research
    interp_factor=10 # one order of magnitude at least
    minLength: minimum length of the common in the time range
    DEBUG= boolean
    '''
    #  
    if np.all(np.isnan(x1)) or np.all(np.isnan(x2)):
        return [np.nan,np.nan]
    else:
        # --------------- cross correlation-----------------------------
        # 1. normalize the signal
        x1_std = np.nanstd(x1)
        x2_std = np.nanstd(x2)

        if x1_std == 0 or np.isnan(x1_std):
            x1_norm = x1 - np.nanmean(x1)  # no std, just center
        else:
            x1_norm = (x1 - np.nanmean(x1)) / x1_std

        if x2_std == 0 or np.isnan(x2_std):
            x2_norm = x2 - np.nanmean(x2)  # no std, just center
        else:
            x2_norm = (x2 - np.nanmean(x2)) / x2_std
        
        # 2. Analytical signal
        x1_analytic=hilbert(x1_norm)
        x2_analytic=hilbert(x2_norm)

        # 3.correlate
        x12_corr=correlate(x2_analytic, x1_analytic,mode='full',method='direct')
        # lags
        lags_signal = correlation_lags(len(x2_analytic),len(x1_analytic),mode='full')
        
        # take only positive and reasonable values (UPSTREAM x1 --> DOWNSTREAM x2)
        mask_positive=(lags_signal>=min_lag)*(lags_signal<=max_lag) # already have an idea on the possible lag
        
        x12_corr_pos=x12_corr[mask_positive]
        lags_signal_pos=lags_signal[mask_positive]

        # 4. Normalize
        if not np.all(np.isnan(x12_corr_pos)):
            norm_denom = np.nanmax(np.abs(np.real(x12_corr_pos)))
            if norm_denom != 0.0:
                x12_corr_pos = x12_corr_pos / norm_denom
        else:
            # debug and print('No peak found')                                
            return [np.nan,np.nan]

        # 5. find peaks, max of the real part
        x12_corr_real=np.real(x12_corr_pos)
        x12_corr_imag=np.imag(x12_corr_pos)
        
        # find the peaks on the interpolated values
        peaks_index, prop = find_peaks(x12_corr_real,height=height,prominence=topo_prominence)

        # process the peaks if there is at least one found    
        if peaks_index.size > 0:
            # Extract magnitudes of the peaks
            peak_magnitudes = prop['peak_heights']
            sorted_peak_indices = np.argsort(peak_magnitudes)[::-1]  # Get indices of peaks sorted by magnitude
            ordered_index = peaks_index[sorted_peak_indices]   # Order peaks indexes on total range
            # chose first one
            max_peak_index = ordered_index[0]
            lag = lags_signal_pos[max_peak_index]
            
            if lag==lags_signal_pos[0] or lag==lags_signal_pos[-1]: # meaning the peak is at the interval extrems
                max_peak_index=ordered_index[1] # take the second peak in order
                lag = lags_signal_pos[max_peak_index]
            
            # 6: Find zero crossings
            zero_cross_lagVect = []
            sign_array=np.sign(x12_corr_imag)
            for i in range(1, len(sign_array)):
                if (sign_array[i-1]*sign_array[i]<0): # change of sign
                        # Linear interpolation to find the exact zero crossing point
                        zero_cross_lag = (lags_signal_pos[i-1] + lags_signal_pos[i])/2 # assume the zero is in the middle of the interval
                        zero_cross_lagVect.append(zero_cross_lag)
                        
            zero_cross_lagVect=np.array(zero_cross_lagVect) # make in an array

                # imaginary part as uncertainty estimation, find the closest
            if zero_cross_lagVect.size>0: # If also the hilbert transform has at leas 1-zero crossing
                idx= np.sqrt((zero_cross_lagVect-lag)**2).argmin() # find index of the closest
                imag_lag=zero_cross_lagVect.flat[idx] # fint the lag associated
                
                # debug and print('1st lag & 1st imag ok')
                                
                return [lag*time_step,imag_lag*time_step]     
            
            else:
                # debug and print('1st lag ok')
                                
                return [lag*time_step,np.nan]
        else:
            # debug and print('No peak found')                                
            return [np.nan,np.nan]


def cross_corr_events(lags_dict,variable,event_dict, # target event 
                      cc_input_dict,df_cc_global, # properties of the target variable
                      df_data_dict,input_dict, # data of the target variable
                      #---------------------------------------------------
                      variable_aux,
                      cc_input_dict_aux,df_cc_global_aux, # auxiliary variable 
                      df_data_dict_aux,input_dict_aux,
                      debug):
    '''
    perform the cross correlation event per event
    '''
    # target event variable input data
    col_values    =input_dict  ['col_values'        ]
    upStation     =input_dict  ['upstream_station'  ]
    downStation   =input_dict  ['downstream_station']
    df_upstream   =df_data_dict[upStation           ]
    df_downstream =df_data_dict[downStation         ]
    
    # cross correlation parameters
    min_numb_data         =cc_input_dict['min_numb_data'         ]
    dt_error              =cc_input_dict['dt_error'              ]
    str_timeStep_downscale=cc_input_dict['str_timeStep_downscale']
    timeStep_downscale    =cc_input_dict['timeStep_downscale'    ]
    min_lag               =cc_input_dict['min_lag'               ]
    max_lag               =cc_input_dict['max_lag'               ]
    topo_prom             =cc_input_dict['topo_prom'             ]
    height_peak           =cc_input_dict['height_peak'           ]
    
    lag_global=int(abs(df_cc_global.loc[0,'lag_[h]']))
    dt_downstream=timedelta(hours=lag_global)
    #---------------------------------------------------------------
    # auxiliary variable input data
    col_values_aux   =input_dict_aux  ['col_values']
    df_upstream_aux  =df_data_dict_aux[upStation   ]
    df_downstream_aux=df_data_dict_aux[downStation ]
    
    # cross correlation parameters
    min_numb_data_aux          =cc_input_dict_aux['min_numb_data'          ]
    dt_error_aux               =cc_input_dict_aux['dt_error'               ]
    str_timeStep_downscale_aux =cc_input_dict_aux['str_timeStep_downscale' ]
    timeStep_downscale_aux     =cc_input_dict_aux['timeStep_downscale'     ]
    min_lag_aux                =cc_input_dict_aux['min_lag'                ]
    max_lag_aux                =cc_input_dict_aux['max_lag'                ]
    topo_prom_aux              =cc_input_dict_aux['topo_prom'              ]
    height_peak_aux            =cc_input_dict_aux['height_peak'            ]
    
    lag_global_aux=int(abs(df_cc_global_aux.loc[0,'lag_[h]']))
    dt_downstream_aux=timedelta(hours=lag_global_aux)
    
    
    
    # --- routine ---
    events=list(event_dict.keys()) # list of the events
    
    
    # iterate for each event
    for date in events:
        start=event_dict[date].index[ 0]
        end  =event_dict[date].index[-1]
            
        # get the data around the event
        df_up  =df_upstream  [start:end+dt_downstream+dt_error]
        df_down=df_downstream[start:end+dt_downstream+dt_error]
        
        df_up_aux   =df_upstream_aux   [start:end+dt_downstream_aux+dt_error_aux]
        df_down_aux =df_downstream_aux [start:end+dt_downstream_aux+dt_error_aux]
        
        # prepare the data 
        x_up,x_down,x_time=data_prep(df_up,
                        df_down,
                        col_values,
                        str_timeStep_downscale,
                        min_numb_data)
       
        y_up,y_down,y_time=data_prep(df_up_aux,
                        df_down_aux,
                        col_values_aux,
                        str_timeStep_downscale_aux,
                        min_numb_data_aux)
        
        
        # study the cross correlation
        lag  ,lag_hilb  =cross_corr(x_up  ,x_down  ,
                                        timeStep_downscale,
                                        topo_prom   ,
                                        height_peak   ,
                                        min_lag  ,max_lag  ,
                                        debug)
        
        lag_aux  ,lag_hilb_aux  =cross_corr(y_up  ,y_down  ,
                                        timeStep_downscale_aux,
                                        topo_prom_aux   ,
                                        height_peak_aux   ,
                                        min_lag_aux  ,max_lag_aux  ,
                                        debug)
        
        lags_dict[date][('lag_'+variable)]=lag
        lags_dict[date][('lag_'+variable+'_hilb')]=lag_hilb
        lags_dict[date][('lag_'+variable_aux)]=lag_aux
        lags_dict[date][('lag_'+variable_aux+'_hilb')]=lag_hilb_aux
        lags_dict[date]['init_event']=start
        lags_dict[date]['end_event']=end
    debug and print('lags analyzed: ',len(list(lags_dict.keys())))  
    return lags_dict

def merge_save(lags_dict_q, lags_dict_ssc,merge_save_input_dict,debug):
    tr_time=merge_save_input_dict['tr_time_overlap']
    # -------------- 1. Join and drop overlapping events----------------
    lags_dict_JOINT={}
    for event_ssc in list(lags_dict_ssc.keys()):
            df_ssc=lags_dict_ssc[event_ssc].copy()
            init_ssc=df_ssc.init_event[0]
            end_ssc =df_ssc.end_event[0]
            
            for event_q in list(lags_dict_q.keys()):
                df_q=lags_dict_q[event_q].copy()
                
                init_q=df_q.init_event[0]
                end_q=df_q.end_event[0]
                # Calculate the overlap
                start_max = max(init_ssc, init_q)  # Latest start time
                end_min = min(end_ssc, end_q)  # Earliest end time
                
                if start_max < end_min:  # Valid overlap
                    overlap_duration = end_min - start_max  # Result is a timedelta
                # If the time difference is below the threshold, drop the discharge event
                    if overlap_duration > tr_time:
                        lags_dict_q.pop(event_q)
                        break  # Exit the loop once the event is removed

        # Save the filtered events to the joint dictionary
    lags_dict_JOINT['discharge']= lags_dict_q
    lags_dict_JOINT['susp_sed' ] = lags_dict_ssc  
    
    
    # ------------ 2. Create a unique dictionary ------------------

    q_joint  =lags_dict_JOINT['discharge'].copy()
    ssc_joint=lags_dict_JOINT['susp_sed' ].copy()
    
   
    init=[]
    end=[]
    lag_hilb_Q=[]
    lag_Q=[]
    lag_hilb_SSC=[]
    lag_SSC=[]
    index=[]
    for eventQ in list(q_joint.keys()):
        index.append(eventQ)
        q_df=q_joint[eventQ].copy()
        init         .append(q_df['init_event'   ][0])
        end          .append(q_df['end_event'    ][0])
        lag_hilb_Q   .append(q_df['lag_Q_hilb'   ][0])
        lag_Q        .append(q_df['lag_Q'        ][0])
        lag_hilb_SSC .append(q_df['lag_SSC_hilb' ][0])
        lag_SSC      .append(q_df['lag_SSC'      ][0])
            
    for eventSSC in list(ssc_joint.keys()):
        ssc_df=ssc_joint[eventSSC].copy()
        index.append(eventSSC)
        init         .append(ssc_df['init_event'   ][0])
        end          .append(ssc_df['end_event'    ][0])
        lag_hilb_Q   .append(ssc_df['lag_Q_hilb'   ][0])
        lag_Q        .append(ssc_df['lag_Q'        ][0])
        lag_hilb_SSC .append(ssc_df['lag_SSC_hilb' ][0])
        lag_SSC      .append(ssc_df['lag_SSC'      ][0])
            
            # Create a DataFrame using the collected data
    concatenated_df = pd.DataFrame({
            'init_event'   : init        ,
            'end_event'    : end         ,
            'lag_Q_hilb'   : lag_hilb_Q  ,
            'lag_Q'        : lag_Q       ,
            'lag_SSC_hilb' : lag_hilb_SSC,
            'lag_SSC'      : lag_SSC
        }, index=index)

    # Store the DataFrame in the new dictionary
    lags_CONCAT_df = concatenated_df.copy()
    
    #---------- 3. save------------
    save_df(lags_CONCAT_df,merge_save_input_dict,debug)
        
    return lags_CONCAT_df



def to_seconds(time_value):
    if isinstance(time_value, timedelta):
        return time_value.total_seconds()
    return np.nan

def lags_cel_convert(lags_CONCAT_df,geom_dict,
                     lag2cel_input_dict,debug):
    
    lags_df=lags_CONCAT_df.copy() #--lags calculated
    #  events list
    events_index=lags_df.index
    
    col_celQ,col_celQ_hilb,col_celSSC,col_celSSC_hilb=(
        'cel_Q_ms','cel_Q_hilb_ms','cel_SSC_ms','cel_SSC_hilb_ms')
    
    df2save=pd.DataFrame(index=events_index, 
                         columns=['init_event','end_event',
                                    col_celQ,col_celQ_hilb,
                                    col_celSSC,col_celSSC_hilb])
    
    # distance between the two stations
    L=geom_dict['distance']
    
    # saving arrays 
    N_events=len(events_index)
    (celQ_vect, celQ_hilb_vect,
     celSSC_vect,celSSC_hilb_vect) = (
        np.zeros(N_events) for _ in range(4))
     
    
    for n,date in enumerate(events_index):
        
        # data
        timeQ        =lags_df.loc[date,'lag_Q'       ]
        timeQ_hilb   =lags_df.loc[date,'lag_Q_hilb'  ]
        timeSSC      =lags_df.loc[date,'lag_SSC'     ]
        timeSSC_hilb =lags_df.loc[date,'lag_SSC_hilb']
        
        # convert into seconds
        lagQ_sec        = to_seconds(timeQ       )
        lagQ_hilb_sec   = to_seconds(timeQ_hilb  )
        lagSSC_sec      = to_seconds(timeSSC     )
        lagSSC_hilb_sec = to_seconds(timeSSC_hilb)
        
        # calculate celerities
        celQ       =L/lagQ_sec
        celQ_hilb  =L/lagQ_hilb_sec
        celSSC     =L/lagSSC_sec
        celSSC_hilb=L/lagSSC_hilb_sec
        
        # save in array
        celQ_vect        [n]=celQ
        celQ_hilb_vect   [n]=celQ_hilb
        celSSC_vect      [n]=celSSC
        celSSC_hilb_vect [n]=celSSC_hilb
    
    df2save.loc[:,'init_event']=lags_df.loc[:,'init_event']
    df2save.loc[:,'end_event' ]=lags_df.loc[:,'end_event' ]
    
    df2save.loc[:,col_celQ       ]=celQ_vect
    df2save.loc[:,col_celQ_hilb  ]=celQ_hilb_vect
    df2save.loc[:,col_celSSC     ]=celSSC_vect
    df2save.loc[:,col_celSSC_hilb]=celSSC_hilb_vect
    
    df2save=df2save.dropna()
    
    # filter for the percentile
    debug and print('celerity measures: ',len(df2save))
    perc=lag2cel_input_dict['upper_quantile']
    tr_up_q=lag2cel_input_dict['tr_velocity_upper_q']
    tr_up_ssc=lag2cel_input_dict['tr_velocity_upper_ssc']
    
    Q_perc        = df2save.loc[:,col_celQ       ].quantile(perc)
    Q_hilb_perc   = df2save.loc[:,col_celQ_hilb  ].quantile(perc)
    SSC_perc      = df2save.loc[:,col_celSSC     ].quantile(perc)
    SSC_hilb_perc = df2save.loc[:,col_celSSC_hilb].quantile(perc)
    
    # clean for extremes
    df2save=df2save[(df2save[col_celQ].values <= Q_perc
                     )*(df2save[col_celQ_hilb].values <= Q_hilb_perc)]
    df2save=df2save[(df2save[col_celSSC].values <= SSC_perc
                     )*(df2save[col_celSSC_hilb].values <= SSC_hilb_perc)]
    
    # clean from unpysical data
    df2save=df2save[(df2save[col_celQ].values <=tr_up_q)*(
        df2save[col_celQ_hilb].values <=tr_up_q)]
    df2save=df2save[(df2save[col_celSSC].values <=tr_up_ssc)*(
        df2save[col_celSSC_hilb].values <=tr_up_ssc)]
    
    # merge unique data
    if not df2save.index.is_unique:
        # compensate for the long zero dataset timings (no ssc in the measurements)
        df2save = df2save.loc[df2save.groupby(df2save.index)[col_celSSC].idxmax()]
    
    debug and print('celerity measures clean: ',len(df2save))
    # save the dataframe to return
    celerity_df=df2save.copy()
    
    #-------------save------------
    save_df(celerity_df,lag2cel_input_dict,debug) 
            
    return celerity_df


def celerities_boxplot(celerity_df,plot_celerities_features):
    # columns from celerities calculation
    col_celQ,col_celQ_hilb,col_celSSC,col_celSSC_hilb=(
        'cel_Q_ms','cel_Q_hilb_ms','cel_SSC_ms','cel_SSC_hilb_ms')
    
    # custom quantiles for the error plot
    quant_up=plot_celerities_features['quant_up']
    quant_low=plot_celerities_features['quant_down']

    df=celerity_df.copy()
    # 1------------ prepare dataframe with plotting quantities
    cel_df=pd.DataFrame(index=[0],columns=[col_celQ,col_celQ_hilb,'std_Q',
                                           col_celSSC,col_celSSC_hilb,'std_SSC',
                                           'kin_wave','kin_wave_hilb','std_kin_wave'])
    
    qMedian=np.nanmedian(df.loc[:,col_celQ])
    sscMedian=np.nanmedian(df.loc[:,col_celSSC])
        
    qMedian_hilb=np.nanmedian(df.loc[:,col_celQ_hilb])
    sscMedian_hilb=np.nanmedian(df.loc[:,col_celSSC_hilb])
        
    kin_distr=(df.loc[:,col_celQ]
               /df.loc[:,col_celSSC])
    kin_distr_hilb=(df.loc[:,col_celQ_hilb]
                    /df.loc[:,col_celSSC_hilb])
        # Print or store the peak celerities
        
    cel_df.loc[0,col_celQ        ]= qMedian
    cel_df.loc[0,col_celSSC      ]= sscMedian
    cel_df.loc[0,col_celQ_hilb   ]= qMedian_hilb
    cel_df.loc[0,col_celSSC_hilb ]= sscMedian_hilb
    cel_df.loc[0,'kin_wave'      ]= np.nanmedian(kin_distr     )
    cel_df.loc[0,'kin_wave_hilb' ]= np.nanmedian(kin_distr_hilb)
    
    cel_df.loc[0,'std_Q'         ]=    0.5*(np.nanquantile(df.loc[: ,col_celQ], quant_up)
                                       -np.nanquantile(df.loc[: ,col_celQ], quant_low))
    cel_df.loc[0,'std_SSC'       ]=    0.5*( np.nanquantile(df.loc[: ,col_celSSC],quant_up)
                                       -np.nanquantile(df.loc[: ,col_celSSC],quant_low))
    cel_df.loc[0,'std_kin_wave'  ]=0.5*( np.nanquantile(kin_distr , quant_up)
                                         -np.nanquantile(kin_distr , quant_low))
        
#2.--------------- PLOT ---------------------
    
    cel_df=cel_df.dropna()

    ratio_color=plot_celerities_features['ratio_color']
    q_color=plot_celerities_features['q_color']
    ssc_color=plot_celerities_features['ssc_color']
    font=plot_celerities_features['font']
    figsize=plot_celerities_features['figsize']
    
    
    fig, ax1 = plt.subplots(figsize=figsize)
    # Define offsets for error barscel_
    offset_Q = 0.1  # Offset for Discharge Celerity
    offset_SSC = -0.1  # Offset for SSC Celerity

    # Plot `cel_Q` with error bars
 
    ax1.errorbar(offset_Q, cel_df.loc[0,col_celQ], yerr=cel_df.loc[0,'std_Q'], 
                label=r'Q celerity $\lambda_Q$ ', color=q_color, 
                linestyle='', marker='o', capsize=3, markerfacecolor='white', 
                markeredgecolor=q_color)
    
    ax1.plot(offset_Q, cel_df.loc[0,col_celQ_hilb], label=r'Q celerity from Hilbert',
            color=q_color, marker='o', linestyle='None',zorder=4)
    
    # # Plot `cel_SSC` with error bars
    ax1.errorbar(offset_SSC, cel_df.loc[0,col_celSSC], yerr=cel_df.loc[0,'std_SSC'], 
                label=r'$\varphi$ celerity $\lambda_{\varphi}$ ', color=ssc_color, 
                linestyle='', marker='o', capsize=3, markerfacecolor='white', 
                markeredgecolor=ssc_color)

    ax1.plot(offset_SSC, cel_df.loc[0,col_celSSC_hilb], label=r'$\varphi$ celerity from Hilbert',
            color=ssc_color, marker='o', linestyle='None',zorder=4)

    pos=0
    labelpad=plot_celerities_features['labelpad']
    ax1.set_ylabel('Celerity [m/s]',fontsize=font,labelpad=labelpad)
    ax1.tick_params(axis='y')

    # Create a secondary y-axis for `kin_wave`
    ax2 = ax1.twinx()
    ax2.plot(pos, cel_df.loc[0,'kin_wave_hilb']/(5/3) -1, label=r' $\dfrac{\lambda_{Q,H}/\lambda_{\varphi,H}}{5/3} -1 $',
            color=ratio_color, marker='*', markersize=13, markerfacecolor=ratio_color,linestyle='None',zorder=4)
    
    ax2.errorbar(pos, cel_df.loc[0,'kin_wave']/(5/3) -1 , yerr=cel_df.loc[0,'std_kin_wave']/(5/3), 
                label=r' $\dfrac{\lambda_Q/\lambda_{\varphi}}{5/3} -1 $', color=ratio_color, 
                linestyle='', marker='*',markersize=13, capsize=3, markerfacecolor='white', 
                markeredgecolor=ratio_color)
    
    ax2.set_ylabel(r'$\dfrac{\lambda_Q/\lambda_{\varphi}}{5/3} -1$  [-]',fontsize=font,labelpad=labelpad,color=ratio_color)
    ax2.tick_params(axis='y', labelcolor=ratio_color)
    ax2.hlines(y=0,xmin=-0.5,xmax=3.5,lw=0.5,color=ratio_color)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2,loc='upper left',bbox_to_anchor=(0.78,0.98))
    ax2.grid(lw=0.3,ls=':',axis='y')
    ax1.set_xlim(2*offset_SSC,2*offset_Q)
    # 
    ax2.set_xticks([])
    ax1.set_xticks([])
    # 
    ax1.text(offset_SSC, ax1.get_ylim()[0] - 0.05,r'$\varphi$',
             color=ssc_color, fontsize=font, ha='center', va='top')
    ax1.text(offset_Q, ax1.get_ylim()[0] - 0.05, 'Q',
             color=q_color, fontsize=font, ha='center', va='top')
    # Show the plot
    plt.title(r'Virtual celerities and kinematic wave assumption',fontsize=font+2,pad=labelpad+5)
    plt.show()
    




# ---------------------------------------------------------------------------------
#           REACH SCALE PROP
# ---------------------------------------------------------------------------------



def quantile_funct(df,col,quant):
    cdf=np.cumsum(df.loc[:,col])/np.sum(df.loc[:,col])
    time=df.index[np.searchsorted(cdf, quant, side='right')]
    return time

def q_u_reach(station_couple,
              input_dict_q_1h,input_dict_ssc_1h, # data format
                q_dict_data,ssc_dict_data, # hourly data from the station
                df_global_q,df_global_ssc, # global cross correlation between stations
                cel_df, # celerity df
                geom_dict, # morpho properties fo channel
                qU_reach_input_dict,
                debug): # variables for the qU reach analaysis
    
    up_station=station_couple['upstream']
    down_station=station_couple['downstream']
    
    col_values_q=input_dict_q_1h['col_values']
    col_values_ssc=input_dict_ssc_1h['col_values']
    
    timeDeltaQ  =qU_reach_input_dict['dt_error_q'  ]
    timeDeltaSSC=qU_reach_input_dict['dt_error_ssc']
    quantHigh   =qU_reach_input_dict['quant_up'    ]
    quantLow    =qU_reach_input_dict['quant_low'   ]
    quantMid    =qU_reach_input_dict['quant_mid'   ]
    lenMin      =qU_reach_input_dict['min_len'     ]
    max_delay   =timedelta(hours=qU_reach_input_dict['max_delay']).total_seconds() 
    
    lag_ssc_global = int(abs(df_global_ssc['lag_[h]'].iloc[0]))  # Global lag for SSC
    dt_downstream_ssc = timedelta(hours=lag_ssc_global)
    
    lagQ_global = int(abs(df_global_q['lag_[h]'].iloc[0]))  # Global lag for Q
    dt_downstream_q = timedelta(hours=lagQ_global)
    
    # channel properties
    L=geom_dict['distance']
    W=geom_dict['width']
    
    df_save=pd.DataFrame(index=cel_df.index,
                         columns=['q','U','D','dqdD',
                                  'delay','dQdt','dQdt_max',
                                  'init_event','end_event'])
    
    for date in list(df_save.index):
        # event extent
        init=cel_df.loc[date,'init_event']
        end =cel_df.loc[date,'end_event' ]
        
        # crop event data
        SSC_up   = ssc_dict_data[up_station   ].loc[init:(end+dt_downstream_ssc+timeDeltaSSC),:] # Upstream station data
        SSC_down = ssc_dict_data[down_station ].loc[init:(end+dt_downstream_ssc+timeDeltaSSC),:]
        Q_up     = q_dict_data  [up_station   ].loc[init:(end+dt_downstream_q  +timeDeltaQ  ),:]
        Q_down   = q_dict_data  [down_station ].loc[init:(end+dt_downstream_q  +timeDeltaQ  ),:]
        
        time_step=(Q_up.index[2]-Q_up.index[1]).total_seconds()
        
        # ---------------- calculate the discharge gradient in advance ---------------
        Q_up_vect  =Q_up.loc[:,col_values_q].values
        Q_down_vect=Q_down.loc[:,col_values_q].values
        
        dQdt_up   = np.gradient(Q_up_vect  , time_step) #
        dQdt_down = np.gradient(Q_down_vect, time_step)
        
        dQdt_up_df  =pd.DataFrame(index=Q_up.index,columns=['dQdt'])
        dQdt_down_df=pd.DataFrame(index=Q_down.index,columns=['dQdt'])
        
        dQdt_up_df  .loc[:,'dQdt']=dQdt_up
        dQdt_down_df.loc[:,'dQdt']=dQdt_down
        
        if  (   (len(    SSC_up)>lenMin)
            and (len(    SSC_down)>lenMin)
            and (len(    Q_up )>lenMin)
            and (len(    Q_down)>lenMin)):

            # Quantile positioning, center
            time_up_ssc   = quantile_funct(SSC_up  ,col_values_ssc,quantMid)
            time_down_ssc = quantile_funct(SSC_down,col_values_ssc,quantMid)
            
            time_up_q   = quantile_funct(Q_up,col_values_q,quantMid)
            time_down_q = quantile_funct(Q_down,col_values_q,quantMid)
            
            # get time delay in seconds
            diff_up  =(time_up_q  -time_up_ssc  ).total_seconds()
            diff_down=(time_down_q-time_down_ssc).total_seconds()
            
            # get time spread of the plume
            timeRange_up_ssc   =0.5*(quantile_funct(SSC_up  ,col_values_ssc,quantHigh)
                                     -quantile_funct(SSC_up,col_values_ssc,quantLow))
            timeRange_down_ssc =0.5*(quantile_funct(SSC_down,col_values_ssc,quantHigh)
                                     -quantile_funct(SSC_down,col_values_ssc,quantLow))
            
            # time ranges of the upstream station
            init_range_up=time_up_ssc-timeRange_up_ssc
            end_range_up=time_up_ssc+timeRange_up_ssc
            # downstream station
            init_range_down=time_down_ssc-timeRange_down_ssc
            end_range_down=time_down_ssc+timeRange_down_ssc
            
            Q_median_up   = np.nanmean(Q_up  .loc[init_range_up   :end_range_up   ,col_values_q])
            Q_median_down = np.nanmean(Q_down.loc[init_range_down :end_range_down ,col_values_q])
                
            dQdt_median_up   = np.nanmean(dQdt_up_df  .loc[init_range_up   :end_range_up   ,'dQdt'])
            dQdt_median_down = np.nanmean(dQdt_down_df.loc[init_range_down :end_range_down ,'dQdt'])
                
            dQdt_max_up   =np.nanmax(dQdt_up_df  .loc[init_range_up:end_range_up,'dQdt'])
            dQdt_max_down =np.nanmax(dQdt_down_df.loc[init_range_up:end_range_up,'dQdt'])
                
            q_up=Q_median_up/W
            q_down=Q_median_down/W
                
            U=cel_df.loc[date,'cel_SSC_ms']
            dqdD=cel_df.loc[date,'cel_Q_ms']
            df_save.loc[date,'init_event']=cel_df.loc[date,'init_event']
            df_save.loc[date,'end_event']=cel_df.loc[date,'end_event']
            df_save.loc[date,'U']=U
            df_save.loc[date,'dqdD']=dqdD
            df_save.loc[date,'q']=np.mean([q_up,q_down])
            df_save.loc[date,'D']=np.mean([q_up,q_down])/U
            df_save.loc[date,'delay']=np.mean([diff_up,diff_down])
            df_save.loc[date,'dQdt']=np.mean([dQdt_median_up/W,dQdt_median_down/W])
            df_save.loc[date,'dQdt_max']=np.mean([dQdt_max_up/W,dQdt_max_down/W])
            df_save=df_save[(df_save['delay']>-max_delay)]
            
    df_save=df_save.dropna()
    
    return df_save

def qDu_plot(qDu_df,qDu_plot_input_dict,
             station_couple,
             debug):
    

    # Midpoint Normalize class to center the colormap at 0
    class MidpointNormalize(mcolors.Normalize):
        def __init__(self, vmin=None, vmax=None, vcenter=0.0, clip=False):
            self.vcenter = vcenter
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # Rescale values so that the midpoint (vcenter) is 0
            value = np.ma.masked_array(value, np.isnan(value))
            value -= self.vcenter
            value = np.ma.masked_array(value, np.isnan(value))
            return super().__call__(value)

    fit_df=pd.DataFrame(index=[0],columns=['a','b'])
    
    up_station=station_couple['upstream']
    down_station =station_couple['downstream']
    x_var=qDu_plot_input_dict['x_axis_var']
    y_var=qDu_plot_input_dict['y_axis_var']
    z_cvar=qDu_plot_input_dict['colorbar_var']
    z_label=qDu_plot_input_dict['colorbar_label']
    
    # process the variables
    y= qDu_df[y_var]
    x= qDu_df[x_var] 
    z_cbar=qDu_df[z_cvar]
    
    # log-log lliinear uregression
    log_x = np.log10(x.astype('float64'))
    log_y = np.log10(y.astype('float64'))
    mask = np.isfinite(log_y) & np.isfinite(log_x)  # Mask to handle NaN/Inf values
    
    # linear fit
    slope, intercept, r_value, p_value, std_err = linregress(log_x[mask], log_y[mask])
    a = 10 ** intercept  # Convert intercept from log10 space to linear space
    b = slope

    # Save the fitting parameters
    fit_df.loc[0,'a']=a
    fit_df.loc[0,'b']=b
    
    # ----------------------------------------------------------
    save_df(fit_df,qDu_plot_input_dict,debug)
    if debug:
        #  plot properties
        vmin,vmax=qDu_plot_input_dict['color_range_min'],qDu_plot_input_dict['color_range_max']
        figsize=qDu_plot_input_dict['figsize']
        fontsize=qDu_plot_input_dict['fontsize']
        ylim=qDu_plot_input_dict['ylim']
        xlim=qDu_plot_input_dict['xlim']
        
        # plot
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
        y_fit = a * x_fit ** b
        
        # colorbar properties
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "blue_red", [(0, 'blue'), (0.5, 'white'), (1, 'red')])
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, vcenter=0)
        
        plt.figure(figsize=figsize)
        plt.plot(x_fit, y_fit,
                'k--', label=f'Power-law fit: {y_var} = {a:.2f} * {x_var}^{b:.2f}')
        scatter = plt.scatter(x, y,
                            c=z_cbar, 
                            cmap=cmap,norm=norm, 
                            edgecolor='k') 
        
        # Add colorbar to show the delay values with the transformed log scale
        cbar = plt.colorbar(scatter, label=f'{z_cvar}')  # Create colorbar
        cbar.set_label(f'{z_label}', 
                       rotation=0, labelpad=30)  # Rotate label to vertical
        cbar.set_ticks(np.linspace(vmin, vmax, 5))  # Set colorbar ticks
        
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        # Set plot labels
        x_unit=qDu_plot_input_dict['x_axis_unit']
        y_unit=qDu_plot_input_dict['y_axis_unit']
        plt.xlabel(f'{x_var}{x_unit}',fontsize=fontsize)  # X-axis label
        plt.ylabel(f'{y_var}{y_unit}',fontsize=fontsize)  # Y-axis label
        plt.title(f'Reach scale rating curve for {up_station}:{down_station}',
                  fontsize=fontsize+3) 
        # Title with the reach
        plt.grid(lw=0.15,which='both')
        plt.ylim(ylim)
        plt.xlim(xlim)
        # Show the plot
        plt.show()
        
   
    return fit_df
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
    # define the conversion functions: Recking 2011
g=9.81
def U_star_funct(q_star):
    arg=1+((q_star/10.31)**(0.6317))
    U_star=1.5471*(q_star**0.7062)*(arg**(-0.4930))
    return U_star

def C_funct_recking(D,d84):
    d=D/d84
    arg=1+((d/1.283)**1.618)
    return 4.416*(d**1.904)*(arg**-1.083)

def optim_d84(d84,U,q,S):
    q_star=q/np.sqrt(g*S*(d84**3))
    U_star=U_star_funct(q_star)
    U_star_data=U/np.sqrt(g*S*d84)
    return ((U_star-U_star_data)/U_star_data)**2

def beta_funct(W,D):
    return 0.5*W/D
    
#  optimisation procedure 
def optimize_d84(U, q, S, d84_bounds=(1e-6, 10)):  # Adjust bounds as needed
    result = minimize_scalar(
            optim_d84,
            bounds=d84_bounds,
            args=(U, q, S),
            method='bounded'  # Ensures the search is within bounds
        )
    return result.x, result.fun  # Returns optimal d84 and the minimized error

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
def recking_universal_roughness(qDu_df,
                               recking_input_dict,
                               geom_dict,
                               debug):
    
    S=geom_dict['slope']
    
    reach_hydro_df=qDu_df.copy()
    q_values= reach_hydro_df['q'].values
    U_values= reach_hydro_df['U'].values
    D_values= reach_hydro_df['D'].values
    
    # new columns to save on same dataframe
    new_cols=recking_input_dict['new_var']
    reach_hydro_df[new_cols] = np.nan 
    
    #  optimisation routine
    d84_list = []
    error_list=[]
    C_list= []
    for q, U, D in zip(q_values, U_values,D_values):
        #  calculate d84
        d84_opt, error = optimize_d84(U, q, S)
        d84_list.append(d84_opt)
        error_list.append(error)
        C=C_funct_recking(D,d84_opt)
        C_list.append(C)

    #  save in the dataframe
    d84_d50_conv=recking_input_dict['d84_d50']
    reach_hydro_df['C']=C_list
    reach_hydro_df['d84']=d84_list
    reach_hydro_df['d50']=reach_hydro_df['d84'].values/d84_d50_conv
    reach_hydro_df['err_d84']=error_list
    
    debug and print('reach scale measurements: ',len(reach_hydro_df))
    # drop non-physical valued
    tr_err_zero=recking_input_dict['tr_error_zero'  ]
    min_d84    =recking_input_dict['min_d84'        ]
    min_C      =recking_input_dict['min_C'          ]
    min_sub    =recking_input_dict['min_submergence']
    
    reach_hydro_df = reach_hydro_df[reach_hydro_df['err_d84'] <= tr_err_zero]
    reach_hydro_df = reach_hydro_df[reach_hydro_df['d84'] >= min_d84 ]
    reach_hydro_df = reach_hydro_df[reach_hydro_df['C']> min_C   ]
    reach_hydro_df = reach_hydro_df[reach_hydro_df['D']/reach_hydro_df['d84']>min_sub] # recking 2011
    
    debug and print('reach scale VALID: ',len(reach_hydro_df))
    
   
    
    return reach_hydro_df

# ------------------------------------------------------------------------
coeff_k_dict={}
coeff_k_dict['Elder']=np.array([5.93,0,0])
coeff_k_dict['Fisher']=np.array([0.011,2,2])
coeff_k_dict['Liu']=np.array([0.18,0.5,2])
coeff_k_dict['Iwasa_Aya']=np.array([2,0,1.5])
coeff_k_dict['Seo_Cheong']=np.array([5.915,1.428,0.62])
coeff_k_dict['Koussis']=np.array([0.6,0,2])

def Kphi_funct(coeff_dict,q,C,beta,type):
    coeff=coeff_dict[type]
    a=coeff[0]
    b=coeff[1]
    c=coeff[2]
    k=a*q*(C**(b-1))*((2*beta)**c)
    return k

def K_q_funct(q,S):
    return q/(2*S)

def diffusion_coeff(reach_hydro_df,
                    geom_dict,
                    diffusion_input_dict,
                    debug):
    
    
    method_list=list(coeff_k_dict.keys())
    
    reach_hydro_df_model=reach_hydro_df.copy()
    reach_hydro_df_model[method_list] = np.nan 
    reach_hydro_df_model['K_q']=np.nan
    
    # prepare variables arrays
    D=reach_hydro_df_model.loc[:,'D'].values
    q=reach_hydro_df_model.loc[:,'q'].values
    C=reach_hydro_df_model.loc[:,'C'].values
    W=geom_dict['width']
    S=geom_dict['slope']
    beta=0.5*W/D
    
    #  susp. sediment diffusion
    for method in method_list:
        kphi=Kphi_funct(coeff_k_dict,q,C,beta,method)
        reach_hydro_df_model[method]=kphi
    
    # discharge
    reach_hydro_df_model['K_q']=K_q_funct(q,S)
    
     # save module
    save_df(reach_hydro_df_model,diffusion_input_dict,debug)
    
    return reach_hydro_df_model

# ------------------------
def hydro_plot(qDu_plot_input_dict,reach_hydro_df_model):
    x_var=qDu_plot_input_dict['var_x']
    y_var=qDu_plot_input_dict['var_y']
    xscale=qDu_plot_input_dict['x_scale']
    yscale=qDu_plot_input_dict['y_scale']

    # extract DataFrame data
    x = reach_hydro_df_model[x_var]
    y = reach_hydro_df_model[y_var]
    
    # Crea la figura e gli assi
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotta i dati con marker e linea
    ax.plot(x, y, marker='o', linestyle='', color='k',
            linewidth=2, markersize=6, alpha=0.85)
    
    # Imposta etichette e titolo
    ax.set_xlabel(x_var, fontsize=12, labelpad=10)
    ax.set_ylabel(y_var, fontsize=12, labelpad=10)
    ax.set_title(f"Hydrological Plot:  {y_var} = funct({x_var} )", fontsize=14, pad=15)
    
    # Imposta le scale degli assi
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # Configura la griglia
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Ottimizza il layout e mostra il grafico
    fig.tight_layout()
    plt.show()
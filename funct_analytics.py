import numpy as np
import os
import math
import pandas as pd
from datetime import timedelta, datetime
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dask import delayed, compute
from dask.diagnostics import ProgressBar
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 450
plt.rcParams['mathtext.default'] = 'regular'


from funct_reach import (data_prep,save_df)


# --------- Litarature solutions (Socolovsky et al. 2005) ------------------
def adv_diff_analytic(x,t,l,k,C0):
    arg=-((x-l*t)**2)/(4*k*t)
    coeff=C0/np.sqrt(4*np.pi*k*t)
    return coeff*np.exp(arg)

def diff_source_analytic(x,t,k,C0):
    arg=(x)/np.sqrt(4*k*t)
    return 0.5*C0*(1-erf(arg))
# ---------------------------------------------------------
#  ---------------- analytic solution ludo ----------------
# ---------------------------------------------------------
def funct_adv(x, t, l, k, A0,eps):
    arg=(l/(2 * k))*(x-0.5*l*t )
    arg=np.max([arg,-1/eps])
    return A0 *np.exp(arg)

def funct_diff(x, t,dt, l, k, g_values,t_values,A0,eps):
    t_iter = np.arange(eps, t-eps, dt)
    
    val_sum = 0.0
    x0=0
    for n in range(len(t_iter) - 1):
        
        tn = t_iter[n] 
        tn1 = t_iter[n + 1]
        tn12 = (tn + tn1) / 2
        
        delta_t_n = t - tn
        delta_t_n1 =t - tn1
        
        arg_n = x / np.sqrt(4 * k * delta_t_n)
        arg_n1 = x / np.sqrt(4 * k * delta_t_n1)
        
       
        g_val = np.interp(tn12, t_values, g_values)
        denom=funct_adv(x0, tn12, l, k, A0,eps)
        
        # if denom < eps:
        #     continue  # Skip this interval if the denominator is too small
          
        val12 = (g_val/denom)*(erf(arg_n1) - erf(arg_n))
        
        val_sum += val12
    
    return val_sum

def funct_diff_integral(x, tend,tau,
                        l,k, g_values,t_values,A0,eps):
    diff_t=max(tend-tau,eps)
    
    denom= (diff_t)**(3/2)
    arg=-(x**2)/(4*k*(diff_t))

    denom_adv=funct_adv(0, tau, l, k, A0,eps)
    g_val = np.interp(tau, t_values, g_values)
    
    return (g_val/denom_adv)*(np.exp(arg)/denom)

def integrate_diff(x,tend,l,k,A0,g_values, t_values,eps):
    integral,_=quad_vec(lambda tau: funct_diff_integral(
        x, tend,tau, l,k, g_values,t_values,A0,eps), 0, tend-eps)
    
    coeff=x/np.sqrt(4*np.pi*k)
    return coeff*integral

def funct_C_analytic(x, t,dt, l, k, g_values,t_values, x0,A0,eps):
    A = funct_adv(x, t, l, k, A0,eps)
    B = funct_diff(x, t,dt, l, k, g_values,t_values,A0,eps)
    return A*B + x0

def funct_C_num(x, t, l, k, g_values, t_values, x0,A0,eps):
    A = funct_adv(x, t, l, k, A0,eps)
    B = integrate_diff(x,t,l,k,A0,g_values,t_values,eps)
    
    return A*B + x0

# ----------------------------------------------------------------
# -----------------------------------------------------------------

def comparison_advdiff(input_sim_ssc,debug):
    debug and print('*** Test with advection-diffusion solution ***')
    t=input_sim_ssc['sim_time']
    l=input_sim_ssc['advection_coeff']
    k=input_sim_ssc['diffusion_coeff']
    C0=input_sim_ssc['C0_pulse']
    C_base=input_sim_ssc['C0_base']
    x0=input_sim_ssc['station_0']
    x1=input_sim_ssc['station_1']
    x2=input_sim_ssc['station_2']
    A0=input_sim_ssc['scale_A0']
    eps=input_sim_ssc['eps']
    
    # generate input data
    data = adv_diff_analytic(x0, t, l, k, C0)
    # input data for the model
    g_funct = interp1d(t, data, kind='linear', fill_value="extrapolate")
    g_values=g_funct(t)
    dt=t[-1]/len(t)
    
    # station where edata is recorded
    data1_test = adv_diff_analytic(x1, t, l, k, C0) 
    data2_test = adv_diff_analytic(x2, t, l, k, C0)
    
    # ---------- simulation ------------------
    t_values=t-eps
    t_values[0]=2*eps 
    
    L01=x1-x0
    L02=x2-x0
    #  first station
    C_values_analytic1 = [funct_C_analytic(L01, ti,dt, 
                                 l, k, 
                                 g_values,t_values, C_base,A0,eps
                                 ) for ti in t_values]
    
    C_values_num1 = [funct_C_num(L01, ti,
                            l, k, 
                            g_values,t_values, C_base,A0,eps
                            ) for ti in t_values]
    
    # second station
    C_values_analytic2 = [funct_C_analytic(L02, ti,dt, 
                                 l, k, 
                                 g_values,t_values, C_base,A0,eps
                                 ) for ti in t_values]
    
    C_values_num2 = [funct_C_num(L02, ti,
                            l, k, 
                            g_values,t_values, C_base,A0,eps
                            ) for ti in t_values]
    
    # ------ plot -------------
    colors      =input_sim_ssc['colors'      ]
    linewidth   =input_sim_ssc['linewidth'   ]
    linestyles  =input_sim_ssc['linestyles'  ]
    markers     =input_sim_ssc['markers'     ]
    alpha_values=input_sim_ssc['alpha_values']
    font        =input_sim_ssc['font'        ]
        
        # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each dataset
    ax.plot(
        t, data,
        label="Input",
        color=colors["input"],
        linestyle=linestyles["input"],
        linewidth=0.9,
        marker=markers["input"],
        alpha=alpha_values["input"]
    )


    ax.plot(
        t_values, np.transpose(C_values_analytic1),
        label="Analytic integration",
        color=colors["analytic"],
        linestyle=linestyles["analytic"],
        linewidth=linewidth["analytic"],
        marker=markers["analytic"],
        alpha=alpha_values["analytic"]
    )



    ax.plot(
        t_values, np.transpose(C_values_analytic2),
        color=colors["analytic"],
        linestyle=linestyles["analytic"],
        linewidth=linewidth["analytic"],
        marker=markers["analytic"],
        alpha=alpha_values["analytic"]
    )

    ms=6
    ax.plot(
        t_values, np.transpose(C_values_num2),
        color=colors["numeric"],
        linestyle=linestyles["numeric"],
        marker=markers["numeric"],
        markerfacecolor="white",  # White inside
        markeredgecolor=colors["numeric"],  # Edge color matches the line
        markersize=ms,
        alpha=alpha_values["numeric"],zorder=4
    )

    ax.plot(
        t_values, np.transpose(C_values_num1),
        label="Numerical integration",
        color=colors["numeric"],
        linestyle=linestyles["numeric"],
        marker=markers["numeric"],
        markerfacecolor="white",  # White inside
        markeredgecolor=colors["numeric"],  # Edge color matches the line
        markersize=ms,
        alpha=alpha_values["numeric"],zorder=4
    )

    ax.plot(
        t, data1_test,
        label="Point source analytic solution",
        color=colors["noisy"],
        linestyle=linestyles["noisy"],
        linewidth=linewidth["noisy"],
        marker=markers["noisy"],
        alpha=alpha_values["noisy"]
    )

    ax.plot(
        t, data2_test,
        color=colors["noisy"],
        linestyle=linestyles["noisy"],
        linewidth=linewidth["noisy"],
        marker=markers["noisy"],
        alpha=alpha_values["noisy"]
    )


    # Add annotations for analytical solutions with distances
    annotations = [(str(x0), data), (str(x1), data1_test), (str(x2), data2_test)]
    # Example distances from origin
    for label, values in annotations:
        # Find the peak value and its corresponding time
        peak_index = np.argmax(values)  # Index of the maximum value
        peak_time = t[peak_index]
        peak_value = values[peak_index]

        # Add annotation at the peak
        ax.annotate(
            f"x = {label}",
            xy=(peak_time, peak_value),
            xytext=(peak_time + 0.3, peak_value ),  # Offset for clarity
            fontsize=font["legend_size"],
            arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
            color="black"
        )


    # Customize labels, title, and legend
    ax.set_xlabel("Time", fontsize=font["label_size"],labelpad=10)
    ax.set_ylabel("Concentration", fontsize=font["label_size"],labelpad=10)
    # ax.set_title(f"Advection-Diffusion solution", fontsize=font["title_size"],pad=15)
    ax.legend(loc="best", fontsize=font["legend_size"], frameon=True)

    # Add grid with customization
    ax.grid(visible=True, linestyle='--', alpha=0.7)

    # Customize axis ticks
    ax.tick_params(axis='both', which='major', labelsize=font["size"])

    # Adjust layout for clarity
    plt.tight_layout()

    # Show the plot
    plt.show()   

def comparison_diff(input_sim_ssc,debug):
    debug and print('*** Test with diffusion solution, constant source ***')
    
    t=input_sim_ssc['sim_time']
    l=0.0 # no advection
    k     =input_sim_ssc['diffusion_coeff']
    C0    =input_sim_ssc['C0_pulse'       ]
    C_base=input_sim_ssc['C0_base'        ]
    x0    =input_sim_ssc['station_0'      ]
    x1    =input_sim_ssc['station_1'      ]
    x2    =input_sim_ssc['station_2'      ]
    A0    =input_sim_ssc['scale_A0'       ]
    eps   =input_sim_ssc['eps'            ]
    
    # generate input data
    
    data = diff_source_analytic(x0,t,k, C0)
    # input data for the model
    g_funct = interp1d(t, data, kind='linear',
                        fill_value="extrapolate")
    g_values=g_funct(t)
    dt=t[-1]/len(t)
    
    # station where edata is recorded
    data1_test = diff_source_analytic( x1,t,k, C0) 
    data2_test = diff_source_analytic( x2,t,k, C0)
    
    # ---------- simulation ------------------
    t_values=t-eps
    t_values[0]=2*eps #start with non zero time
    
    L01=x1-x0
    L02=x2-x0
    #  first station
    C_values_analytic1 = [funct_C_analytic(L01, ti,dt, 
                                 l, k, 
                                 g_values,t_values, C_base,A0,eps
                                 ) for ti in t_values]
    
    C_values_num1 = [funct_C_num(L01, ti,
                            l, k, 
                            g_values,t_values, C_base,A0,eps
                            ) for ti in t_values]
    
    # second station
    C_values_analytic2 = [funct_C_analytic(L02, ti,dt, 
                                 l, k, 
                                 g_values,t_values, C_base,A0,eps
                                 ) for ti in t_values]
    
    C_values_num2 = [funct_C_num(L02, ti,
                            l, k, 
                            g_values,t_values, C_base,A0,eps
                            ) for ti in t_values]
    
    # ------ plot -------------
    colors      =input_sim_ssc['colors'      ]
    linewidth   =input_sim_ssc['linewidth'   ]
    linestyles  =input_sim_ssc['linestyles'  ]
    markers     =input_sim_ssc['markers'     ]
    alpha_values=input_sim_ssc['alpha_values']
    font        =input_sim_ssc['font'        ]
        
        # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each dataset
    ax.plot(
        t, data,
        label="Input",
        color=colors["input"],
        linestyle=linestyles["input"],
        linewidth=0.9,
        marker=markers["input"],
        alpha=alpha_values["input"]
    )


    ax.plot(
        t_values, np.transpose(C_values_analytic1),
        label="Analytic integration",
        color=colors["analytic"],
        linestyle=linestyles["analytic"],
        linewidth=linewidth["analytic"],
        marker=markers["analytic"],
        alpha=alpha_values["analytic"]
    )



    ax.plot(
        t_values, np.transpose(C_values_analytic2),
        color=colors["analytic"],
        linestyle=linestyles["analytic"],
        linewidth=linewidth["analytic"],
        marker=markers["analytic"],
        alpha=alpha_values["analytic"]
    )

    ms=6
    ax.plot(
        t_values, np.transpose(C_values_num2),
        color=colors["numeric"],
        linestyle=linestyles["numeric"],
        marker=markers["numeric"],
        markerfacecolor="white",  # White inside
        markeredgecolor=colors["numeric"],  # Edge color matches the line
        markersize=ms,
        alpha=alpha_values["numeric"],zorder=4
    )

    ax.plot(
        t_values, np.transpose(C_values_num1),
        label="Numerical integration",
        color=colors["numeric"],
        linestyle=linestyles["numeric"],
        marker=markers["numeric"],
        markerfacecolor="white",  # White inside
        markeredgecolor=colors["numeric"],  # Edge color matches the line
        markersize=ms,
        alpha=alpha_values["numeric"],zorder=4
    )

    ax.plot(
        t, data1_test,
        label="Point source analytic solution",
        color=colors["noisy"],
        linestyle=linestyles["noisy"],
        linewidth=linewidth["noisy"],
        marker=markers["noisy"],
        alpha=alpha_values["noisy"]
    )

    ax.plot(
        t, data2_test,
        color=colors["noisy"],
        linestyle=linestyles["noisy"],
        linewidth=linewidth["noisy"],
        marker=markers["noisy"],
        alpha=alpha_values["noisy"]
    )


    # Add annotations for analytical solutions with distances
    annotations = [(str(x0), data), (str(x1), data1_test), (str(x2), data2_test)]
    # Example distances from origin
    for label, values in annotations:
        # Find the peak value and its corresponding time
        peak_index = np.argmax(values)  # Index of the maximum value
        peak_time = t[peak_index]
        peak_value = values[peak_index]

        # Add annotation at the peak
        ax.annotate(
            f"x = {label}",
            xy=(peak_time, peak_value),
            xytext=(peak_time + 0.3, peak_value ),  # Offset for clarity
            fontsize=font["legend_size"],
            arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
            color="black"
        )


    # Customize labels, title, and legend
    ax.set_xlabel("Time", fontsize=font["label_size"],labelpad=10)
    ax.set_ylabel("Concentration", fontsize=font["label_size"],labelpad=10)
    # ax.set_title(f"Advection-Diffusion solution", fontsize=font["title_size"],pad=15)
    ax.legend(loc="best", fontsize=font["legend_size"], frameon=True)

    # Add grid with customization
    ax.grid(visible=True, linestyle='--', alpha=0.7)

    # Customize axis ticks
    ax.tick_params(axis='both', which='major', labelsize=font["size"])

    # Adjust layout for clarity
    plt.tight_layout()

    # Show the plot
    plt.show()   

def kinematic_wave(x_up_df,x_down_df,
                   celerity,distance,
                   time_step,  time_step_downscale,
                   debug):
    
    # signal propagation time
    time_delay=distance/celerity
    
    x_up_df_new=x_up_df.copy()
    x_up_df_new.index = x_up_df_new.index + pd.Timedelta(seconds=time_delay)
    
    # resample to downscale time frequency    
    x_up_new_downscale=x_up_df_new.resample(time_step_downscale).interpolate(method='linear')
    x_down_new_downscale=x_down_df.resample(time_step_downscale).interpolate(method='linear')
    
    result_df = x_up_new_downscale.join(x_down_new_downscale,
            how='inner', lsuffix='_up', rsuffix='_down')
    
    # clean the dataframe
    result_df = result_df.dropna()
    
    # resample to original time frequency
    result_df_return=result_df.resample(time_step).mean()
    
    return result_df_return

def adv_diff_wave(x_down_df,event_index,col_values,col_time,
                #   simulation parameters
                  g_values,t_values,distance,
                  celerity, diffusion,
                  C_base,A0,eps,
                #   debut part
                  debug):
    
    result_df=pd.DataFrame(index=event_index,columns=[(col_values+'_up')])
    result_df = result_df.rename_axis(col_time)
     
    adv_diff_sim = [funct_C_num(distance, ti,
                            celerity, diffusion, 
                            g_values,t_values, C_base,A0,eps
                            ) for ti in t_values]
    
    
    # save the results in the dataframe
    result_df.loc[:,(col_values+'_up')]=adv_diff_sim
    result_df[(col_values+'_down')]=x_down_df[col_values].values
    
    return result_df  

# ----------------------------------------------------------------
def plot_sim_execution_test(x_up_df,result_df,col_values,variable):
    start_time = result_df.index[0]
    end_time = result_df.index[-1]
    # Plotting the concatenated results
    plt.figure(figsize=(15, 6))
    plt.plot( x_up_df.index, x_up_df[col_values].values, 
             color='red',linestyle=':', label='Upstream Station')
    
    plt.plot( result_df.index,result_df[(col_values+'_up')].values,
             color='red',linestyle='-', label='sim Upstream station')
    plt.plot( result_df.index,result_df[(col_values+'_down')].values,
             'blue',linestyle='-', label='Downstream station')
    
    plt.title(f"Event Routing {start_time}:{end_time}")
    plt.xlabel("Time (h)")
    plt.ylabel(f"{variable}")
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=5))  # Set ticks every hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))  # Format as 'Hour'
    plt.legend()
    plt.grid(lw=0.15)
    plt.show()
    
# -----------------------------------------------------------------
# comparison between station 
# ----------------------------------------------------------------
  
def sim_execution_test(input_data_dict,data_dict,station_couple,
                    geom_dict,df_global_corr,
                    crosscorr_input_dict,
                    hydro_reach_df,
                    input_sim_station_difference,variable,
                    debug):
    # coluimns name
    col_values=input_data_dict['col_values' ]
    col_time  =input_data_dict['time_values'] 
    # distance in meters
    x_meters=geom_dict['distance']
    
    # chose only one random event
    random_event = hydro_reach_df.sample(n=1)
    # hydro_reach_df.loc[pd.to_datetime('2016-04-18 14:00:00'),:]
    debug and print('simulating ',random_event.index[0],' event')


    dt_average=np.abs(df_global_corr['lag_[h]'])
    decimal_part, integer_part = math.modf(float(dt_average.iloc[0]))
    hours=int(integer_part)
    minutes=np.round(60*decimal_part,0)
    # correction to average time lag in the reach
    dt=timedelta(hours=hours,minutes=minutes) 
    dt_error=crosscorr_input_dict['dt_error']
    
    # event
    init=random_event['init_event'].iloc[0]
    end =random_event['end_event' ].iloc[0]
   
    total_seconds_event=(end - init).total_seconds()
    # data
    x_up_all=data_dict[station_couple['upstream'  ]]
    x_down_all=data_dict[station_couple['downstream']]

    #  data cropping for the event
    x_up,x_down=(x_up_all.loc[init-dt_error:end+dt+dt_error],
                 x_down_all.loc[init-dt_error:end+dt+dt_error])
    
    
    Pe_tr=input_sim_station_difference['Pe_threshold']
    
    # parameters
    if variable=='ssc':
        method=input_sim_station_difference['diffusion_method_ssc']
        l=random_event['U'].values[0]
        k=random_event[method].values[0]
        # 
    elif variable=='q':
        l=random_event['dqdD'].values[0]
        k=random_event['K_q'].values[0]
    
    # data preparation, , ssc remains to the same resolution, q is downscalesd to 10 minutes
    x_up_sim,x_down_sim,event_time=data_prep(x_up,x_down,col_values, newfreq='10min')

    # prepare dataframe for easier handling
    x_up_df,x_down_df=(pd.DataFrame(index=event_time,columns=[col_values]),
                       pd.DataFrame(index=event_time,columns=[col_values]))
    x_up_df[col_values],x_down_df[col_values]= (x_up_sim,x_down_sim)
    x_up_df,x_down_df=(x_up_df.rename_axis(col_time),
                       x_down_df.rename_axis(col_time))
    # 
    # timing of the simulaiton
    eventTime_delta = event_time.diff().total_seconds()
    eventTime_step = np.nancumsum(eventTime_delta)
    # Avoid singularities
    eps=input_sim_station_difference['eps']
    eventTime_step[0] = eventTime_step[0] + eps
    eventTime_step[-1] = eventTime_step[-1] - eps
    # 
    time_step=input_sim_station_difference['time_step']
    time_step_downscale=input_sim_station_difference['time_step_downscale']
    
    # ---------- prepare the input data ----------------
    g_funct = interp1d(eventTime_step, x_up_sim,
                       kind='linear', fill_value="extrapolate")
    val_min_input=np.min(x_up_sim[x_up_sim >0])
    # array of inputs
    A0=1
    g_values = g_funct(eventTime_step) - val_min_input  +eps
    g_df=pd.DataFrame(index=event_time,columns=['g'])
    g_df['g']=g_values
      
    # calculate peclet number
    Pe=total_seconds_event*(l**2)/k
    if Pe>Pe_tr:
        debug and print('*** kineamtic wave ***')
        debug and print('Peclet',Pe,'\n',
              'advection m/s',l,'\n',
              'diffusion m2/s', k)
        # kinematic wave routing
        result_df=kinematic_wave(x_up_df,x_down_df,
                   l,x_meters,
                   time_step,  time_step_downscale,
                   debug)
        # plot
        if not result_df.empty:
            debug and print('results:',result_df.head())
            plot_sim_execution_test(x_up_df, result_df, col_values, variable)
        else:
            print(result_df.head(),'\n',
                  x_up_df.index[0],x_up_df.index[-1])
            debug and print("Warning: result_df is empty. Skipping plot.")
    else:
        debug and print('*** advection diffusion wave ***')
        debug and print('Peclet',Pe,'\n',
              'advection m/s',l,'\n',
              'diffusion m2/s', k)
        result_df=adv_diff_wave(x_down_df,event_time,col_values,col_time,
                                g_values,eventTime_step,x_meters,
                                l,k,val_min_input,A0,eps,
                                debug)
        # crop initial flat term 
        date2crop=init+timedelta(seconds=x_meters/l)
        result_df=result_df[result_df.index>=date2crop]
        
        # plot
        if not result_df.empty:
            debug and print(result_df.head())
            # crop for the plot also the input
            x_up_df=x_up_df[x_up_df.index>=date2crop]
            plot_sim_execution_test(x_up_df, result_df, col_values, variable)
        else:
            debug and print("Warning: result_df is empty. Skipping plot.")



def process_single_event(event_date, event_df,
                         col_values_q, col_time_q, q_up_all, q_down_all, dt_error_q, dt_q,
                         col_values_ssc, col_time_ssc, ssc_up_all, ssc_down_all, dt_error_ssc, dt_ssc,
                         x_meters, input_sim_station_difference, Pe_tr, debug):
    
    init = event_df['init_event']
    end = event_df['end_event']
    total_seconds_event = (end - init).total_seconds()
    
    # simulate for discharge
    result_df_q = simulation_event(col_values_q, col_time_q,
                                   q_up_all, q_down_all, x_meters,
                                   init, end, dt_error_q, dt_q, total_seconds_event,
                                   'q', event_df, input_sim_station_difference, Pe_tr, debug)
                                   
    # simulate for ssc
    result_df_ssc = simulation_event(col_values_ssc, col_time_ssc,
                                     ssc_up_all, ssc_down_all, x_meters,
                                     init, end, dt_error_ssc, dt_ssc, total_seconds_event,
                                     'ssc', event_df, input_sim_station_difference, Pe_tr, debug)

    # merge
    result_df = merge_event_results(result_df_q, result_df_ssc, debug)
    
    if not result_df.empty:
        return event_date, result_df.dropna()
    else:
        return event_date, None



def sim_events(station_couple,geom_dict,
                input_dict_q_1h,q_dict_data,
                input_dict_ssc_10min,ssc_dict_data_10min,
                df_global_q,df_global_ssc,
                crosscorr_input_dict_q,crosscorr_input_dict_ssc,
                hydro_reach_df,
                input_sim_station_difference,
                debug):
    
    # import the columns name
    col_values_q=input_dict_q_1h['col_values' ]
    col_time_q  =input_dict_q_1h['time_values'] 
    col_values_ssc=input_dict_ssc_10min['col_values' ]
    col_time_ssc  =input_dict_ssc_10min['time_values']

    # distance in meters
    x_meters=geom_dict['distance']
    
    # raw data
    q_up_all  =q_dict_data[station_couple['upstream'   ]]
    q_down_all=q_dict_data[station_couple['downstream' ]]
    ssc_up_all  =ssc_dict_data_10min[station_couple['upstream'   ]]
    ssc_down_all=ssc_dict_data_10min[station_couple['downstream' ]]
    
    # average time lag between stations
    dt_average_q=np.abs(df_global_q['lag_[h]'])
    dt_average_ssc=np.abs(df_global_ssc['lag_[h]'])

    decimal_part_q, integer_part_q = math.modf(float(dt_average_q.iloc[0]))
    hours_q=int(integer_part_q)
    minutes_q=np.round(60*decimal_part_q,0)
    # correction to average time lag in the reach
    dt_q=timedelta(hours=hours_q,minutes=minutes_q) 
    dt_error_q=crosscorr_input_dict_q['dt_error']

    decimal_part_ssc, integer_part_ssc = math.modf(float(dt_average_ssc.iloc[0]))
    hours_ssc=int(integer_part_ssc)
    minutes_ssc=np.round(60*decimal_part_ssc,0)
    # correction to average time lag in the reach
    dt_ssc=timedelta(hours=hours_ssc,minutes=minutes_ssc) 
    dt_error_ssc=crosscorr_input_dict_ssc['dt_error']
    
    
   # list of events
    events_list=hydro_reach_df.index

    Pe_tr=input_sim_station_difference['Pe_threshold']
    # saving dictionary
    # 1. Queue up delayed tasks
    delayed_tasks = []
    for i, event_date in enumerate(events_list):
        event_df = hydro_reach_df.loc[event_date, :]
        
        # Wrap the function call in compute-lazy delayed()
        task = delayed(process_single_event)(
            event_date, event_df,
            col_values_q, col_time_q, q_up_all, q_down_all, dt_error_q, dt_q,
            col_values_ssc, col_time_ssc, ssc_up_all, ssc_down_all, dt_error_ssc, dt_ssc,
            x_meters, input_sim_station_difference, Pe_tr, debug
        )
        delayed_tasks.append(task)
    
    # 2. Execute all tasks in parallel using all available CPU cores
    n_workers=max(2,os.cpu_count() - 1)  # Leave one core free
    debug and print(f"Simulating events in parallel using {n_workers} CPU cores...")
    with ProgressBar():
        computed_results = compute(*delayed_tasks,scheduler='processes',
                                   num_workers=n_workers)  
    
    # 3. Reassemble the dictionary
    result_dict = {}
    for event_date, res_df in computed_results:
        if res_df is not None:
            result_dict[event_date] = res_df
        else:
            debug and print(f'\n*** failed sim ***\n {event_date} \n')

    return result_dict

def simulation_event(col_values,col_time,
                     x_up_all,x_down_all,
                     x_meters,
                     init,end,dt_error,dt,total_seconds_event,
                     variable,event_df,
                     
                     input_sim_station_difference,Pe_tr,
                     debug):
        
#  data cropping for the event
    x_up,x_down=(x_up_all.loc[init-dt_error:end+dt+dt_error],
                x_down_all.loc[init-dt_error:end+dt+dt_error])
    
    # parameters
    if variable=='ssc':
        method=input_sim_station_difference['diffusion_method_ssc']
        l= event_df['U']
        k= event_df[method]
        # 
    elif variable=='q':
        l= event_df['dqdD']
        k= event_df['K_q']
    
    # data preparation, , ssc remains to the same resolution, q is downscalesd to 10 minutes
    x_up_sim,x_down_sim,event_time=data_prep(x_up,x_down,col_values, newfreq='10min')
    # prepare dataframe for easier handling
    x_up_df,x_down_df=(pd.DataFrame(index=event_time,columns=[col_values]),
                    pd.DataFrame(index=event_time,columns=[col_values]))
    x_up_df[col_values],x_down_df[col_values]= (x_up_sim,x_down_sim)
    x_up_df,x_down_df=(x_up_df.rename_axis(col_time),
                    x_down_df.rename_axis(col_time))
    # 
    
    # timing of the simulaiton
    eventTime_delta = event_time.diff().total_seconds()
    eventTime_step = np.nancumsum(eventTime_delta)
    # Avoid singularities
    eps=input_sim_station_difference['eps']
    eventTime_step[0] = eventTime_step[0] + eps
    eventTime_step[-1] = eventTime_step[-1] - eps
    # 
    time_step=input_sim_station_difference['time_step']
    time_step_downscale=input_sim_station_difference['time_step_downscale']
        
    # ---------- prepare the input data ----------------
    g_funct = interp1d(eventTime_step, x_up_sim,
                    kind='linear', fill_value="extrapolate")
    val_min_input=np.min(x_up_sim[x_up_sim >0])
    # array of inputs
    A0=1
    g_values = g_funct(eventTime_step) - val_min_input  +eps
    g_df=pd.DataFrame(index=event_time,columns=['g'])
    g_df['g']=g_values
        
    # calculate peclet number
    Pe=total_seconds_event*(l**2)/k
    if Pe>Pe_tr:
        debug and print('*** kineamtic wave ***')
        debug and print('Peclet',Pe,'\n',
            'advection m/s',l,'\n',
            'diffusion m2/s', k)
        
        # kinematic wave routing
        result_df=kinematic_wave(x_up_df,x_down_df,
                l,x_meters,
                time_step,  time_step_downscale,
                debug)
    else:
        debug and print('*** advection diffusion wave ***')
        debug and print('Peclet',Pe,'\n',
            'advection m/s',l,'\n',
            'diffusion m2/s', k)
        result_df=adv_diff_wave(x_down_df,event_time,col_values,col_time,
                                g_values,eventTime_step,x_meters,
                                l,k,val_min_input,A0,eps,
                                debug)
        
        
    # only data span, do not include initial baseflow
    date2crop=init+timedelta(seconds=x_meters/l)
    result_df=result_df[result_df.index>=date2crop]
    
    return result_df


# def merge_event_results(df_q,df_ssc,debug):
#     # check that the simulations went well
#     if df_q is None or df_ssc is None:
#         debug and print("One of the inputs is None.")
#         return pd.DataFrame()
    
#     merged_df = df_q.join(df_ssc,
#                           how='inner', 
#                           lsuffix='_q', rsuffix='_ssc')
#     return merged_df

def merge_event_results(df_q, df_ssc, debug):
    if df_q is None or df_ssc is None:
        debug and print("One of the inputs is None.")
        return pd.DataFrame()

    merged_df = df_q.join(df_ssc, how='inner', lsuffix='_q', rsuffix='_ssc')
    return merged_df.apply(pd.to_numeric, errors='coerce').astype('float64')


def save_sim(result_dict,
               input_sim_diff_merge_save,
               debug):
    debug and print('events analysed: ',len(list(result_dict.keys())),'\n')
    # saving routine
    save_df(result_dict,input_sim_diff_merge_save,debug)


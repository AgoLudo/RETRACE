import numpy as np
from datetime import timedelta
import matplotlib as mpl
from funct_event_segmentation_main import event_segmentation
from funct_reach_main import time_lag_analysis,reach_scale_prop
from fuct_analytics_main import test_function,station_diff

from funct_var_analysis_main import (
    simulation_uncertainty,FPCA_regression_main,
    event_mass_analysis)

station_couple={'upstream'  :'Pontamafrey',
                'downstream':'Aiguebelle'}

# reaches=['Arval_Amont:Arval_Amont','Arval_Amont:Arvan_Aval','Arvan_Aval:Pontamafrey',
#              'Pontamafrey:Aiguebelle','Aiguebelle:Montmelian','Montmelian:Grenoble_campus']

# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   MODULE 1: Extraction of the mass-transporting events
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DISCHARGE input dictionary
input_dict={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_1h',
    'file_name_in': 'Q_'+station_couple['upstream']+'_1h.txt',
    'station_name': station_couple['upstream'],
    'col_values': 'Valeur_[m3/s]', # column name : VALUES
    'time_values':'DateHeure', # clumn name : TIME
    'head':0, # header
    'sep':' '} # separator sign

# TRACER input dictionary
input_dict_aux={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_1h',
    'file_name_in': 'SSC_'+station_couple['upstream']+'_1h.txt',
    'station_name': station_couple['upstream'],
    'col_values': 'Valeur_[mg/L]',
    'time_values':'DateHeure',
    'head':0,
    'sep':' '  
}

# EMD EVENT SEGMENTATION - timescale selection
timescale_dict={
    'time_resolution_sec':3600, # seconds
    'WN_timescale':0.9, # short time floctuations - events [days]
    'Signal_timescale':7, # event - weather[days]
    'Climate_timescale':90 # weather - season days
}

# EMD EVENT SEGMENTATION - event segmentation
event_param_dict={
    'dist_timeSteps': 7, # max number of time steps to consider for event ending or init
    'quantile': 0.75 , # statistical quantile for flood definition
    'minimum_quantile': 0.75, # second filtering: event higher than 'time_span'
    'time_span': 'ME', # second filtering: 'time_span' for quantile
    'time_span_days': 30,
}

# where to save the events
path_output=r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData'
file_name_output    =(input_dict['station_name']+r'_q_data_events.pkl'  )
file_name_output_aux=(input_dict['station_name']+r'_ssc_data_events.pkl')


input_events_ssc_1h={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData',
    'file_name_in':(input_dict['station_name']+r'_ssc_data_events.pkl')}

input_hour2_10min_input={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_10min',
    'file_name_in':('SSC_'+ input_dict['station_name']+'_10m.txt'),
    'col_values': 'Valeur_[mg/L]',
    'time_values':'DateHeure',
    'head':0,
    'sep':' ',
    'input_dict_ssc1h':input_events_ssc_1h,
    'timedelta_err':timedelta(hours=1),
    # saving
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\SSC_10min',
    'file_name_out':(input_dict['station_name']+r'_ssc_data_events_10m.pkl')
    }


# amplitude frequency plot for the whole timeseries
input_ampfreq_plot={
          'cols':['rel_energy','timescale_days'],
          'var_imf':'Q [m3/s]',
          'var_imf_aux':'SSC [g/L]',
        #   plot
          'band_names':['Sub-event', 'Event', 'Weather','Season', 'Climate'],
          'band_colors':['royalblue', 'forestgreen', 'yellow', 'orange','crimson']  ,
          'band_alpha':0.15,
          'title': 'Relative energy contribution - Timescale',
          'xlabel':r'timescale $\tau$ [days]',
          'ylabel':r'$S_{rel}$',
          'fontsize':15

}



# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       MODULE 2: Estimation of the virtual velocity
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


input_dict_q_1h={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_1h',
    'upstream_station': station_couple['upstream'],
    'downstream_station': station_couple['downstream'],
    'file_name_upstream': ('Q_'+station_couple['upstream']+'_1h.txt'),
    'file_name_downstream': ('Q_'+station_couple['downstream']+'_1h.txt'),
    'col_values': 'Valeur_[m3/s]',
    'time_values':'DateHeure',
    'head':0,
    'sep':' ',
    'dt_sec':3600,
    'window_size':2, # minimla smoothing
    'polyorder':1}

input_dict_ssc_1h={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_1h',
    'upstream_station': station_couple['upstream'],
    'downstream_station': station_couple['downstream'],
    'file_name_upstream': ('SSC_'+station_couple['upstream']+'_1h.txt'),
    'file_name_downstream': ('SSC_'+station_couple['downstream']+'_1h.txt'),
    'col_values': 'Valeur_[mg/L]',
    'time_values':'DateHeure',
    'head':0,
    'sep':' ',
    'dt_sec':3600,
    'window_size':2, # minimal smoothing
    'polyorder':1}

input_dict_ssc_10m={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\Code_sedAnalysis\data_10min',
    'upstream_station': station_couple['upstream'],
    'downstream_station': station_couple['downstream'],
    'file_name_upstream': ('SSC_'+station_couple['upstream']+'_10m.txt'),
    'file_name_downstream': ('SSC_'+station_couple['downstream']+'_10m.txt'),
    'col_values': 'Valeur_[mg/L]',
    'time_values':'DateHeure',
    'head':0,
    'sep':' ',
    'dt_sec':600,
    'window_size':2,
    'polyorder':1}

input_events_q={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData',
    'station': station_couple['upstream'],
    'file_name_in': (station_couple['upstream']+r'_q_data_events.pkl')}

input_events_ssc={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\SSC_10min',
    'station': station_couple['upstream'],
    'file_name_in': (station_couple['upstream']+'_ssc_data_events_10m.pkl')}

timeStep_downscale_q=timedelta(minutes=10) # dowscale all Q timeseries to 10 minutes
crosscorr_input_dict_q={
    'min_numb_data':5, # minimum number of datapoints
    'dt_error':timedelta(hours=2), # uncertainty over the average propagation time A--> B
    'height_peak':0.3, # the peak in the cross correlation must be > 0.4 (normalized)
    'topo_prom':0.01, # small topographical prominence
    'timeStep_downscale':timeStep_downscale_q,
    'str_timeStep_downscale':'10min',
    'min_lag':timedelta(minutes=10)/timeStep_downscale_q,
    'max_lag':timedelta(hours=24)/timeStep_downscale_q} # user input from average knowledge on the data

timeStep_downscale_ssc=timedelta(minutes=1) # dowscale all Q timeseries to 1 minute
crosscorr_input_dict_ssc={
    'min_numb_data':5,
    'dt_error':timedelta(hours=1), # uncertainty over the average propagation time A--> B
    'height_peak':0.4,
    'topo_prom':0.01,
    'timeStep_downscale':timeStep_downscale_ssc,
    'str_timeStep_downscale':'1min',
    'min_lag':timedelta(minutes=9)/timeStep_downscale_ssc,
    'max_lag': timedelta(hours=24)/timeStep_downscale_ssc}

# merge the events into unique list
merge_save_input_dict={
    'tr_time_overlap':timedelta(hours=12), # max N of hours overlap between events to be crossed out
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\time_lags',
    'file_name_out':('lags_df_'+station_couple['upstream']+'_'+station_couple['downstream']+r'.pkl')}

lag2cel_input_dict={
    'upper_quantile':0.99, # eliminate extreme measures
    'tr_velocity_upper_q':7, # m/s  upper thresholds of celerities
    'tr_velocity_upper_ssc':6, # m/s
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\time_lags',
    'file_name_out':('cels_df_'+station_couple['upstream']+'_'+station_couple['downstream']+r'.pkl')}

# properties to plot the celerity Q ans SSC
plot_celerities_features={
    'quant_up':0.75,
    'quant_down':0.25,
    'q_color':'b',
    'ssc_color':'r',
    'ratio_color':'blueviolet',
    'font':13,
    'figsize':(9,7),
    'labelpad':10
}


# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      MODULE 3: Fitting of a reach-averaged rating curve 
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

input_cel_df={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\time_lags',
    'file_name_in':('cels_df_'+station_couple['upstream']+'_'+station_couple['downstream']+r'.pkl'),
}

input_lag_df={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\time_lags',
    'file_name_in':('lags_df_'+station_couple['upstream']+'_'+station_couple['downstream']+'.pkl'),
}

qU_reach_input_dict={
    'dt_error_q':crosscorr_input_dict_q['dt_error'],
    'dt_error_ssc':crosscorr_input_dict_ssc['dt_error'],
    'quant_up':0.75,
    'quant_low':0.25,
    'quant_mid':0.5,
    'min_len': 4, # minimum number of measurements per event
    'max_delay':12 #hours
    }

#plot properties 
qDu_plot_input_dict={
    'figsize':(10,7),
    'x_axis_var':'q',
    'x_axis_unit':'[m2/s]',
    'y_axis_var':'D',
    'y_axis_unit':'[m]',
    'colorbar_var':'U',
    'colorbar_label':'U [m/s]',
    'color_range_min': 0.5,
    'color_range_max': 3,
    'fontsize':13,
    'ylim':[0.09,1.1],
    'xlim':[0.09,1.1],
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\reach_hydro',
    'file_name_out':('qU_fit_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df_.pkl'),
    # ---- plot 2 -------------
    # var list [q,U,D,dqdD,delay,dQdt,dQdt_max,'init_event',
    #    'end_event', 'C', 'd84', 'd50', 'err_d84', 'Elder', 'Fisher', 'Liu',
    #    'Iwasa_Aya', 'Seo_Cheong', 'Koussis', 'K_q']
    'var_x': 'U',
    'var_y':'D',
    'x_scale':'linear',
    'y_scale':'linear'
}

reckin_input_dict={
    'new_var':['C','d84','d50','err_d84'],
    'tr_error_zero':1e-6, # error in the d84 search routine
    'min_d84':2e-3, # minimum vale of d84
    'min_C':2,  # recking 2011: minimum C value for empirical relation
    'min_submergence':0.5, # recking 2011: minimum sub. value for empirical relation
    'd84_d50':2.2, # recking 2011: estimation d84/d50 value for empirical relation
                   }


diffusion_input_dict={
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\reach_hydro',
    'file_name_out':('reach_hydro_'+station_couple['upstream']+'_'+station_couple['downstream']+r'_df.pkl')
}


# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            MODULE 4: Signal routing
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1. define plotting properties
colors_dict = {
        "input": "black",
        "analytic": "mediumblue",
        "numeric": "mediumblue",
        "noisy": "crimson",
        }
linewidth_dict = {
    "input": 2,
    "analytic": 2,
    "numeric": 2,
    "noisy": 2.5,
}

linestyles_dict = {
        "input": "-",
        "analytic": "-",
        "numeric": "",
        "noisy": ":",
            }
markers_dict = {
    "input": "",
    "analytic": "",
    "numeric": "o",
    "noisy": "",
}
alpha_values_dict = {
    "input": 0.5,
    "analytic": 1.0,
    "numeric": 1.0,
    "noisy": 1.0,
}
font_dict = {
    "size": 17,
    "title_size": 22,
    "label_size": 19,
    "legend_size": 16,
}

# Analytical solution testing
configuration_test='advdiff' #(advdiff/diff) 
input_sim_ssc_test={
    'diffusion_coeff': 10, #m²/s
    'advection_coeff': 1, #m/s
    'C0_pulse': 1, #g/L
    'C0_base':0.0,
    'scale_A0':1,
    'station_0':1, #distance at which save data
    'station_1':3,
    'station_2':5,
    'sim_time':np.linspace(0.0001,10,150),
    'eps':1e-25,
    # ------- Plot features --------------
    'colors': colors_dict,
    'linewidth': linewidth_dict,
    'linestyles': linestyles_dict,
    'alpha_values':alpha_values_dict,
    'markers':markers_dict,
    'font':font_dict    
}

input_sim_station_difference={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\reach_hydro',
    'file_name_in':('reach_hydro_'+station_couple['upstream']+'_'+station_couple['downstream']+r'_df.pkl'),
    'diffusion_method_ssc':'Iwasa_Aya',# Iwasa_Aya/Fisher/Elder/Liu/Seo_Cheong/Koussis
    'Pe_threshold':2200,
    'eps':1e-25,
    'time_step':'10min',
    'time_step_downscale':'1min'
}

input_sim_diff_merge_save={
    'col_time':input_dict_q_1h['time_values'],
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\simulation',
    'file_name_out':('simulation_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df.pkl')
}




# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           MODULE 5: Time differencing interpretation
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
columns_FPCA=['delta_q',
             'q_shear',
             'D_der_sec',
             'ssc_input',
             'target']

input_dict_FPCA={
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\simulation',
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\FPCA_analysis',
    'file_name_in':('simulation_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df.pkl'),
    'file_name_out':('FPCA_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df.pkl'),
    'col_values_q':input_dict_q_1h['col_values'],
    'col_values_ssc':input_dict_ssc_1h['col_values'],
    'rel_mass_threshold':0.6, # max error on the mass routing # <<<<
    'window_size':3, # number of time steps (6 == 1 hour)
    'polyorder':1,
    'exp_variance_threshold':0.99, #explained variance by the FPCAs 
    'orthogonality_threshold':0.1, # threshold over the normalized dot product
    'r2_threshold':0.3, # fitting threshold (adjusted r2) # <<<<
    'columns_FPCA':columns_FPCA
}

input_dict_plot_mass = {
    'figure_size': (10, 6),
    'marker_style': 'o',
    'marker_color': 'navy',
    'avg_line_color': 'red',
    'error_color': 'grey',
    'marker_edge_color': 'navy',
    'marker_size': 4,
    'error_line_width': 0.7,
    'alpha':0.2,
    'x_label': 'Time',
    'y_label': r'$\frac{M_{sim}-M_{obs}}{M_{obs}}$',
    'labelfont':14,
    'title': 'Analytical model mass conservation',
    'grid': True,
    'grid_line_width': 0.1,
    'annotations': [
        {
            'position': (0.05, 0.95),
            'fontsize': 12,
            'horizontal_alignment': 'left',
            'vertical_alignment': 'center'
        },
        {
            'position': (0.05, 0.90),
            'fontsize': 12,
            'horizontal_alignment': 'left',
            'vertical_alignment': 'center'
            }
            ]
    }


list_season= ['spring','summer','autumn','winter']
list_month =['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
color_vars={
    columns_FPCA[0]:'darkorange',
    columns_FPCA[1]:'forestgreen',
    columns_FPCA[2]:'crimson',
    columns_FPCA[3]:'royalblue',
    'non_explained':'dimgrey'
}
color_season={
    list_season[0]:'darkorange',
    list_season[1]:'forestgreen',
    list_season[2]:'crimson',
    list_season[3]:'royalblue'
}
label_list={
    'delta_q':r'$\Delta\,Q\,$',
    'q_shear': r'$\tau$',
    'D_der_sec': r'${d\,D}/{d\,t}$',
    'ssc_input':r'$\varphi$',
    'non_explained':r'comb.'
}


input_dict_massVar_plot={
    'erosion_color':'red',
    'deposition_color':'blue',
    'erosion_color_obs':'firebrick',
    'deposition_color_obs':'navy',
    'figsize':(15,7),
    'title':'Reach scale - event based mass variation',
    'xlabel':'Time [year]',
    'ylabel':r'$\Delta\,Mass\;[ton]$',
    'ylabel2':r'Cumulative $Mass\;[ton]$',
    'label_sim':r'$\Delta\,Mass\;[ton]$ simulations',
    'label_obs':r'$\Delta\,Mass\;[ton]$ observations',
    'linestyle_obs':' ',
    'markerstyle_obs':'.',
    'markersize_obs':8,
    'label2':r'$\sum\,\Delta\,Mass\;[ton]$',
    'labelsize':19,
    'labelpad':17,
    'cum_linestyle':'--',
    'linthreshold_cum_mass':100, # tons under which there is uncertainty, the log plot will be linear
    # ------ ratio plot -----
    'figsize_ratio':(10, 6),
    'title_ratio': r'Event Mass Exchange Ratio',
    'label_ratio': r'$EMER = (M_{out} - M_{in})\,/\, \int^{t}{|\Delta\,Mass(t)|}$',
    'color_marker_ratio':'k',
    'color_mean_ratio':'red',
    'linewidth_ratio':1.5,
    'linestyle_mean':'-',
    'linestyle_median':':',
    'label_mean':'mean',
    'label_median':'median', 
    'label_xaxis': 'EMER', 
    'label_yaxis': r'$pdf$',
    'bins_mass_exchange':25,  
    'window_ratio':3,
    'polyorder_ratio':1,
    # ----------- monthly plot----------
    'list_season':list_season,
    'list_month':list_month,
    'fontsize_month':18,
    'color_month_fluxes':'k',
    'color_month_emer':'darkviolet',
    'ls_month_emer':'-',
    'figsize_month':(15,10),
    'ylabel_month':r'$|Q_s| \; [kg \cdot sec^{-1}]$',
    'ylabel_emer_month':r'emer index',
    'title_month':r'Sediment fluxes during the year',
    'title_month_emer':r'EMER index during the year',
     # -------- var classification plot-----
    'color_variables':color_vars,
    'label_list':label_list,
    'figsize_season_var':(14,10),
    'ylim_season_var':[0.0,0.8],
    'legend_var':r'$Q_{s,rel}$'
    }


# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                           MODULE 6: Energy-spectrum analysis
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


imf_opts = {'stop_method':'sd',
            'sd_thresh': 0.001}

min_timestep=(timedelta(minutes=1).total_seconds()) # max resolution of the data downscaled
min_resolution=(timedelta(minutes=10).total_seconds())
max_timestep=(timedelta(hours=48).total_seconds()) # max length of an event
n=3 # minimum resolution
bins=25
centers_timescale=np.logspace(np.log10(min_resolution*n), 
                           np.log10(max_timestep),
                           num=bins )

# bins for the variables plot analysis
centers_timescale_var=np.logspace(np.log10(min_resolution*n), 
                           np.log10(max_timestep),
                           num=bins )
title_list={
    'delta_ssc':r'$\Delta\,\varphi(t)\;[g/L]$',
    'delta_q':r'$\Delta\,Q(t)\;[m^3/s]$',
    'delta_mass':r'$\Delta\,Mass(t)\;[kg/s]$'
}
#s 
target_color_dict={
          'delta_ssc':mpl.cm.Reds,
          'delta_q':mpl.cm.Blues,
          'delta_mass':mpl.cm.plasma
}
input_dict_emd={
    'EMD_calculate':False, # if true calculate, if false, then it loads it from the folder
    'target_columns':['delta_ssc','delta_q','delta_mass'],
    'target_colors':target_color_dict,
    # emd decomposition
    'imf_opts':imf_opts,
    'max_imfs':10,
    'sift_thresh':1e-10,
    'return_residual':True,
    'method_downscale':'linear', #downscale the signal to have more points
    'eps_boundaries': 1e-5,# extra boudaries, how should they be approximated to 0
    # frequency analysis
    'method_HHT':'nht', #method to perform Hilber Huag Transform : normalized Hilbert transform
    'nbins_HHT':1000,
    'min_timestep_sec': min_timestep,  
    'max_timestep_sec': max_timestep, 
    'scale_grid':'log',
    'min_resolution_sec':min_resolution,
    'amplitude_share_threshold':0.1, # 10% of the total power (excluding trend)
    'path_output':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\HHT_analysis',
    'path_input':r'C:\Users\Agostini\OneDrive - ETH Zurich\ssc_first\Data_Analysis\Suspended_Sediments\Isere_Topkapi_calibration\code\HydroData\HHT_analysis',
    'file_name_in':('hht_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df.pkl'),
    'file_name_out':('hht_'+station_couple['upstream']+'_'+station_couple['downstream']+'_df.pkl'),
    # histogram analysis and signal power share
    'time_scale_centers':centers_timescale,
    'result_columns':['norm_power','time_scale_sec'],
    'title_conversion': title_list,
    'max_power_colorscale':0.4,
    'max_pdf_freq':0.135,
    # season var plot
    'figsize':(13,10),
    'time_scale_centers_var':centers_timescale_var,
    'var2plot':'delta_mass', #'delta_q','delta_ssc' # can decompose also on the other results
    'min_imf_num':30, # min number of imf to be in the plot ( 3/4 valid imfs per event >> 10 events)
    'label':label_list,
    'colors':color_vars,
    'quantile_low':0.20,
    'quantile_high':0.75,
    'labelsize':12,
    'linewidth':2,
    'max_errorplot_yaxis':10.9, # adapt to include more vars in the plot
    'title':r'$Q_s(t)$ timescales in '
}



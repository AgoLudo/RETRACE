
# import functions
import warnings
warnings.filterwarnings("ignore")

from funct_event_segmentation_main import event_segmentation,hour2_10min_crop,ampfreq_contribution_plot
from funct_reach_main import time_lag_analysis,reach_scale_prop
from fuct_analytics_main import test_function,station_diff

from funct_var_analysis_main import (
    simulation_uncertainty,FPCA_regression_main,
    event_mass_analysis)

# import input dictionaries
from main_input import (input_dict,input_dict_aux,timescale_dict,event_param_dict,
                        path_output,file_name_output,file_name_output_aux,input_hour2_10min_input,input_ampfreq_plot)

from main_input import(station_couple,input_dict_q_1h,input_dict_ssc_1h,input_dict_ssc_10m,
                       input_events_q,input_events_ssc,
                       crosscorr_input_dict_q,crosscorr_input_dict_ssc,lag2cel_input_dict,
                       merge_save_input_dict,plot_celerities_features)

from main_input import(input_cel_df,qU_reach_input_dict,qDu_plot_input_dict,
                 reckin_input_dict,diffusion_input_dict)

from main_input import(configuration_test,input_sim_ssc_test)

from main_input import(input_sim_station_difference,input_sim_diff_merge_save,input_dict_plot_mass)

from main_input import (input_dict_FPCA,input_dict_massVar_plot,input_dict_emd)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   MODULE 1: Extraction of the mass-transporting events
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# debug = True
# # perform segmentation on discharge Q
# event_segmentation(input_dict,
#                    input_dict_aux,
#                    timescale_dict,
#                    event_param_dict,
#                    path_output,
#                    file_name_output,
#                     debug=debug)
# debug = False
# # perform segmentation on SSC hour
# event_segmentation(input_dict_aux      , # <<< the aux becomes the main variable
#                    input_dict          , # <<< the auxiliary variable is the discharge
#                    timescale_dict      ,
#                    event_param_dict    ,
#                    path_output         ,
#                    file_name_output_aux, # <<<
#                    debug=debug)
# debug = False
# # # crop the 10min events to the save 1h events segmentation of SSC
# hour2_10min_crop(input_dict,
                #  input_hour2_10min_input,debug)

# debug = False
# ampfreq_contribution_plot(input_dict,input_dict_aux,
#                               input_ampfreq_plot,timescale_dict,
#                               debug)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       MODULE 2: Estimation of the virtual velocity
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# debug=True
# # ---------------- run the cross correlation ---------------
# time_lag_analysis(station_couple,
#                       input_dict_q_1h,
#                       input_dict_ssc_1h,
#                       input_dict_ssc_10m,
                      
#                       input_events_q, # 1h
#                       input_events_ssc, # 10min
                      
#                       crosscorr_input_dict_q,
#                       crosscorr_input_dict_ssc,
#                       lag2cel_input_dict,
                      
#                       merge_save_input_dict,
#                       plot_celerities_features,
#                       debug)

# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      MODULE 3: Fitting of a reach-averaged rating curve 
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# debug=False
# reach_scale_prop(station_couple,
#                  input_dict_q_1h,input_dict_ssc_1h,input_dict_ssc_10m,
#                  input_cel_df,
#                  qU_reach_input_dict,qDu_plot_input_dict,
#                  reckin_input_dict,
#                  diffusion_input_dict,
#                  debug)


# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            MODULE 4: Signal routing
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function to test the accuracy of the simulation to the analytical solutions
# test_function(configuration_test,input_sim_ssc_test, debug=True)

'''
MODE1: try the code with a random event, chose the variable (debug=True & single_station=True)
MODE2: if single_station=False, then run for all the events
# '''
# debug=False
# single_station=False # simulate just one event, to try and see results
# variable_test='q' # q,ssc
# station_diff(station_couple,
#              input_dict_q_1h,input_dict_ssc_1h,
#              input_dict_ssc_10m,
#              crosscorr_input_dict_q,crosscorr_input_dict_ssc,
#              input_sim_station_difference,input_sim_diff_merge_save,
#              debug,single_station,variable_test)

# ---------------- simulaiton mass uncertainty estimation ---------------------
# debug=True
# _=simulation_uncertainty(station_couple,
#                        input_dict_q_1h,input_dict_ssc_10m,
#                        input_sim_station_difference,
#                        input_dict_plot_mass,
#                        input_dict_FPCA,debug)

# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         MODULE 5: Time differencing interpretation
#                         MODULE 6: Energy-spectrum analysis
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
debug=True
event_mass_analysis(station_couple,
                    input_dict_q_1h,input_dict_ssc_10m, 
                    input_sim_station_difference,
                    input_dict_FPCA,
                    input_dict_massVar_plot,
                    input_dict_emd,
                        debug)

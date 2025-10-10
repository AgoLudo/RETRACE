from funct_analytics import (comparison_advdiff,comparison_diff,
                sim_execution_test,sim_events,save_sim)
from funct_reach import (import_data_stationCouple,cross_corr_global,
                         reach_morphometry,import_df)

def test_function(config,input_sim, debug):
    if config=='diff':
        comparison_diff(input_sim,debug)
    elif config=='advdiff':
        comparison_advdiff(input_sim, debug)
    else:
        print('Error: config not recognized')        
    return None

def station_diff(station_couple,
                 input_dict_q_1h,input_dict_ssc_1h,input_dict_ssc_10min,
                 crosscorr_input_dict_q,crosscorr_input_dict_ssc,
                 input_sim_station_difference,input_sim_diff_merge_save,
                 debug,single_station,variable_test):
    
    #  import data 
    q_dict_data         =import_data_stationCouple(input_dict_q_1h     ,debug )
    ssc_dict_data       =import_data_stationCouple(input_dict_ssc_1h   ,debug )
    ssc_dict_data_10min =import_data_stationCouple(input_dict_ssc_10min,debug )
    
    # channel morphology
    geom_dict_system=reach_morphometry(debug)
    geom_dict=geom_dict_system[(station_couple['upstream']+':'+station_couple['downstream'])]
    
    
    # ----------------GLOBAL CROSS CORRELATION ----------------------------------
    debug and print('>> cross_corr_global *main* <<')
    df_global_q=cross_corr_global(q_dict_data,geom_dict,input_dict_q_1h,debug)
    df_global_ssc=cross_corr_global(ssc_dict_data,geom_dict,input_dict_ssc_1h,debug)

    # channel hydraulics
    hydro_reach_df=import_df(input_sim_station_difference,
                             debug)
    
    if debug or single_station:

        if variable_test=='ssc':
            debug and print('>> sim_execution_test *main* <<')
            sim_execution_test(input_dict_ssc_10min,ssc_dict_data_10min,station_couple,
                        geom_dict,df_global_ssc,
                        crosscorr_input_dict_ssc,
                        hydro_reach_df,
                      input_sim_station_difference,variable_test,
                      debug)
        elif variable_test=='q':
                debug and print('>> sim_execution_test *main* <<')
                sim_execution_test(input_dict_q_1h,q_dict_data,station_couple,
                        geom_dict,df_global_q,
                        crosscorr_input_dict_q,
                        hydro_reach_df,
                      input_sim_station_difference,variable_test,
                      debug)
    
    
    if not single_station:

        # simulation all 
        debug and print('>> sim_events *main* <<')
        result_dict=sim_events(station_couple,geom_dict,
                                 input_dict_q_1h,q_dict_data,
                                 input_dict_ssc_10min,ssc_dict_data_10min,
                        df_global_q,df_global_ssc,
                        crosscorr_input_dict_q,crosscorr_input_dict_ssc,
                        hydro_reach_df,
                      input_sim_station_difference,
                      debug)
        
        print(len(list(result_dict.keys())),'simulations')

        # merge and save
        save_sim(result_dict,
                   input_sim_diff_merge_save,debug)
        
  
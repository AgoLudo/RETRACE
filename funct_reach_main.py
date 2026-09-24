import matplotlib.pyplot as plt
import pandas as pd
from funct_reach import (import_data_stationCouple,
                         reach_morphometry,
                         cross_corr_global,prepare_df2store_cc,cross_corr_events,
                         lags_cel_convert,merge_save,celerities_boxplot)

from funct_reach import(import_df,q_u_reach,qDu_plot,hydro_plot,
                        recking_universal_roughness,diffusion_coeff)


def time_lag_analysis(station_couple,
                      input_dict_q_1h,
                      input_dict_ssc_1h,
                      input_dict_ssc_10min,
                      
                      input_events_q,
                      input_events_ssc,
                      
                      crosscorr_input_dict_q,
                      crosscorr_input_dict_ssc,
                      lag2cel_input_dict,
                      
                      merge_save_input_dict,
                      plot_celerities_features,
                      debug):
    
    # -------------------------------------------------------------------------------
    #  import data 
    debug and print('>> import_data_stationCouple *main* <<')
    q_dict_data         =import_data_stationCouple(input_dict_q_1h     ,debug )
    ssc_dict_data       =import_data_stationCouple(input_dict_ssc_1h   ,debug )
    ssc_dict_data_10min =import_data_stationCouple(input_dict_ssc_10min,debug )
    
   
    # import events 
    debug and print('>> import_df *main* <<')
    event_dict_q  =import_df(input_events_q,debug ) # hourly
    event_dict_ssc=import_df(input_events_ssc,debug) # 10 minutes
    
    # channel morphology
    geom_dict_system=reach_morphometry(debug)
    geom_dict=geom_dict_system[(station_couple['upstream']+':'+station_couple['downstream'])]
    
    # ----------------GLOBAL CROSS CORRELATION ----------------------------------
    debug and print('>> cross_corr_global *main* <<')
    df_global_q  =cross_corr_global(q_dict_data  ,geom_dict,input_dict_q_1h  ,debug)
    df_global_ssc=cross_corr_global(ssc_dict_data,geom_dict,input_dict_ssc_1h,debug)
    
    

    # -------------- EVENT CROSS CORRELATION ---------------------------
    lags_dict_q=prepare_df2store_cc(event_dict_q,debug)
    lags_dict_ssc=prepare_df2store_cc(event_dict_ssc,debug)
    
    # analyse flood event data
    variable_q='Q'
    variable_q_aux='SSC'
    debug and print('>> Q: cross_corr_events *main* <<')
    lags_dict_q=cross_corr_events(lags_dict_q,variable_q,event_dict_q,
                                    crosscorr_input_dict_q,df_global_q,
                                    q_dict_data,input_dict_q_1h,
                                    #--------------------------
                                    variable_q_aux,
                                    crosscorr_input_dict_ssc,df_global_ssc,
                                    ssc_dict_data_10min,input_dict_ssc_10min,
                                    debug)
    
    # analyse pulse event data (SSC)
    variable_ssc='SSC'
    variable_ssc_aux='Q'
    debug and print('>> SSC: cross_corr_events *main* <<')
    lags_dict_ssc=cross_corr_events(lags_dict_ssc,variable_ssc,event_dict_ssc,
                                    crosscorr_input_dict_ssc,df_global_ssc,
                                    ssc_dict_data_10min,input_dict_ssc_10min,
                                    # ---------------------------------------------
                                    variable_ssc_aux,
                                    crosscorr_input_dict_q,df_global_q,
                                    q_dict_data,input_dict_q_1h,
                                    debug)
  
    
    # ------------------- merge and save -----------------------------
    debug and print('>> merge_save *main* <<')
    lags_concat_df=merge_save(lags_dict_q,lags_dict_ssc,merge_save_input_dict,debug)
    debug and print('>> lags_cel_convert *main* <<')
    cel_concat_df=lags_cel_convert(lags_concat_df,geom_dict,lag2cel_input_dict,debug)

    debug and print('>> celerities_boxplot *main* <<')
    debug and celerities_boxplot(cel_concat_df,plot_celerities_features)
    

    return None
 #--------------------------------------------------------------------------------
#           REACH SCALE PROPERTIES
#---------------------------------------------------------------------------------
def reach_scale_prop(station_couple,
                     input_dict_q_1h,input_dict_ssc_1h,input_dict_ssc_10min,
                     input_cel_df,
                     qU_reach_input_dict,qDu_plot_input_dict,
                     recking_input_dict,
                     diffusion_input_dict,
                     debug):
#  import data 
    q_dict_data         =import_data_stationCouple(input_dict_q_1h     ,debug )
    ssc_dict_data       =import_data_stationCouple(input_dict_ssc_1h   ,debug )
    # ssc_dict_data_10min =import_data_stationCouple(input_dict_ssc_10min,debug )

#  properties
    geom_dict_system=reach_morphometry(debug)
    key = (station_couple['upstream'] + ':' + station_couple['downstream'])
    geom_dict=geom_dict_system[key]

# calculate global cross correlations
# ----------------GLOBAL CROSS CORRELATION ----------------------------------
    debug and print('>> cross_corr_global *main* <<')
    df_global_q  =cross_corr_global(q_dict_data  ,geom_dict,input_dict_q_1h  ,debug)
    df_global_ssc=cross_corr_global(ssc_dict_data,geom_dict,input_dict_ssc_1h,debug)
    cel_df=import_df(input_cel_df,debug)
    
    debug and print('>> q_u_reach *main* <<')
    qDu_df=q_u_reach(station_couple,
                    input_dict_q_1h,input_dict_ssc_1h,
                    q_dict_data,ssc_dict_data,
                    df_global_q,df_global_ssc,  
                    cel_df,
                    geom_dict,
                    qU_reach_input_dict,
                    debug)
        
    debug and print('>> recking_universal_roughness *main* <<')
    reach_hydro_df=recking_universal_roughness(qDu_df,
                               recking_input_dict,
                               geom_dict,
                              debug)
    
    debug and print('>> qDu_plot *main* <<')
    fit_df=qDu_plot(reach_hydro_df,qDu_plot_input_dict,
                   station_couple,
                   debug)
    
    debug and print('>> diffusion_coeff *main* <<')
    reach_hydro_df_model=diffusion_coeff(reach_hydro_df,
                                         geom_dict,
                                         diffusion_input_dict,
                                         debug)
    

    debug and print('>> hydro_plot *main* <<')
    debug and hydro_plot(qDu_plot_input_dict,reach_hydro_df_model)
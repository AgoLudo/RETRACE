



import matplotlib.pyplot as plt
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from funct_event_segmentation import (calculate_index_of_orthogonality,
index_orthogonality_plot,weighted_median,peak_pit_funct,peaks_clean_funct,save_events,
emd_funct,hilbert_deconvolution,import_data,flushing_events,events2exclude,plot_events,plot_events_nat,
ssc_crop_h2min,plot_ampfreq)

from funct_reach import import_df

def event_segmentation(input_dict,input_dict_aux,
                       timescale_dict,event_param_dict,
                       path_output,file_name_output,
                       debug):
    
    '''
    input_dict: dictionary with the information to handle the input file
    input_dict_aux: dictionary with the information to handle the input file to then crop with the first input file events
    smooth_dict: dictionary with the smoothing variables
    timescale_dict: dictionary with timescale selection parameters
    event_param_dict: dictionary with the event parameters
    clean_event_dict: dictionary with the dates to remove from the event list    
    '''
    col_values=input_dict['col_values']
    col_values_aux=input_dict_aux['col_values']
    col_time=input_dict['time_values']
    station_name=input_dict['station_name']
    
    # #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    debug and print('*** ',station_name,' ***')
    
    # ------------------- IMPORT DATA ----------------------- 
    df_data    ,df    =import_data(input_dict    )
    df_aux_data,df_aux=import_data(input_dict_aux)
    
    # --------------------- EMD analysis------------------------
    debug and print('>>emd_funct *main*<<')
    imf=emd_funct(df,col_values,debug)
    # -------------------- FREQUENCY ANALYSIS -------------------
    debug and print('>>hilbert_deconvolution *main*<<')
    signal_events = hilbert_deconvolution(imf,timescale_dict,debug)
    
    # ------------------- FIND MAXIMA & MINIMA  ---------------------
    debug and print('>>peak_pit_funct & peaks_clean_funct *main*<<')
    peaks_df,pits_df = peak_pit_funct(df,signal_events,
                   event_param_dict,
                   input_dict,
                   debug)
    
    # ---------------Clean the data and identify the events--------------
    peaks_df_clean=peaks_clean_funct(df,peaks_df,pits_df,
                      event_param_dict,input_dict,debug)
    
    # #--------------prepare the dataframe with the events to exclude (1 event each year)-----------------------
    debug and print('>>flushing_events & events2exclude *main*<<')
    list_station=['Pontamafrey','Aiguebelle','Montmelian','Grenoble_campus']
    if station_name in list_station: #stations that havee flushing events
        exclude_df=flushing_events(col_time,station_name,debug)
        # # ----------- clean the event dataframe from the flushing events, save flushing events df --------------------
        peaks_df_clean,flushing_event=events2exclude(peaks_df_clean,
                                                 col_values,col_time,exclude_df,debug)
    # # ---------------- plot results for debugging --------------------------
        variable=col_values
        debug and plot_events(df,peaks_df_clean,flushing_event,
                            col_values,col_time,station_name,variable)
    else:
        variable=col_values
        debug and plot_events_nat(df,peaks_df_clean,
                            col_values,col_time,station_name,variable)



    # # ------------------------ save -------------------------------------
    if not debug: 
        save_events(peaks_df_clean,df_data,df_aux_data,
                    col_time,col_values_aux,
                path_output,file_name_output)
    

def hour2_10min_crop(input_dict,
                     input_hour2_10min_input,debug):
    
    # import 1h resolution Q data
    
    col_values_q=input_dict['col_values']
    df_data_q, _=import_data(input_dict)
    debug and print('import timeseries ',input_dict['file_name_in'],'\n')

    # import the 10 min resolution data
    df_data_ssc, _=import_data(input_hour2_10min_input)
    debug and print('import timeseries ',input_hour2_10min_input['file_name_in'],'\n')

    # import 1h events
    input_event_dict=input_hour2_10min_input['input_dict_ssc1h']
    event_dict_ssc_1h =import_df(input_event_dict,debug )

    # perform segmentation
    dt_range=input_hour2_10min_input['timedelta_err']
    event_dict_10min=ssc_crop_h2min(df_data_q,col_values_q,
                                    df_data_ssc,
                                    event_dict_ssc_1h,dt_range)
    
    # ----------------- SAVING -------------------------------------
    path_output=input_hour2_10min_input['path_output']
    file_name_output=input_hour2_10min_input['file_name_out']
    # save .pkl file
    full_file_path = os.path.join(path_output, file_name_output)
    with open(full_file_path, 'wb') as f:
        pickle.dump(event_dict_10min, f)

def ampfreq_contribution_plot(input_dict,input_dict_aux,
                              input_ampfreq_plot,timescale_dict,
                              debug):
    debug and print('>> ampfreq_contribution_plot *main* <<')

    '''plot the amplitude contribution and the timescale of each IMF'''
    col_values    =input_dict    ['col_values']
    col_values_aux=input_dict_aux['col_values']
    # col_time=input_dict['time_values']
    # file_name=input_dict['file_name']
    # station_name=input_dict['station_name']


    # ------------------- IMPORT DATA ----------------------- 
    df_data,df=import_data(input_dict)
    df_aux_data,df_aux=import_data(input_dict_aux)
    
    # --------------------- EMD analysis------------------------
    imf=emd_funct(df,col_values,debug)
    imf_aux=emd_funct(df_aux,col_values_aux,debug)
    # plot
    plot_ampfreq(imf,imf_aux,input_ampfreq_plot,timescale_dict,debug)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,find_peaks
import emd
import plotly.graph_objects as go
import pickle

# --------------------- IMPORT DATA ---------------------
def import_data(input_dict):
    # initialize the variables
    path_input=input_dict['path_input']
    file_name=input_dict['file_name']
    col_time=input_dict['time_values']
    head,sep=input_dict['head'],input_dict['sep']
    
    #-----------------------------------
    file_dir=(path_input+file_name)
    df_data=pd.read_csv(file_dir,sep=sep,header=head)
    df_data[col_time]=pd.to_datetime(df_data[col_time])
    df_data=df_data.set_index(col_time)
    # clean up
    df_clean=df_data.replace([np.inf, -np.inf], np.nan).dropna()
    return (df_data,df_clean)

#----------------- ORTHOGONALITY MATRIX, CHECK --------------------
def calculate_index_of_orthogonality(imfs):
    num_imfs = imfs.shape[1]
    io_values = []

    for k in range(num_imfs - 1):
        imf_k = imfs[:, k]
        imf_k1 = imfs[:,k + 1]
        
        numerator = np.sum(imf_k * imf_k1)
        denominator = np.sum(imf_k**2 + imf_k1**2)

        io = numerator / denominator
        io_values.append(io)

    return io_values
#
def index_orthogonality_plot(imf):
    #------------- calculate ----------------------
    num_imfs = imf.shape[1]
    io_matrix = np.zeros((num_imfs, num_imfs))
    
    for k in range(num_imfs):
        for j in range(num_imfs):
            if k != j:  # Exclude comparison with itself
                imf_k = imf[:, k]
                imf_j = imf[:, j]
                numerator = np.sum(imf_k * imf_j)
                denominator = np.sum(imf_k**2 + imf_j**2)
                io_matrix[k, j] = numerator / denominator
            else:
                io_matrix[k, j] = np.nan  # Set diagonal to NaN
    
    #-------------- plot---------------
    io_matrix_upper = np.triu(io_matrix)
    io_matrix_upper = np.where(io_matrix_upper == 0.0, np.nan, io_matrix_upper)

    plt.figure(figsize=(10, 6))
    plt.imshow(io_matrix_upper, cmap='Reds', interpolation='none', vmin=0, vmax=0.1)
    plt.colorbar(label='Orthogonality Index')
    plt.title('Orthogonality Matrix')

    # Corrected tick labels
    plt.xticks(np.arange(num_imfs), np.arange(1, num_imfs + 1))
    plt.yticks(np.arange(num_imfs), np.arange(1, num_imfs + 1))
    plt.xlabel('IMF Index')
    plt.ylabel('IMF Index')

    # Set limits
    plt.xlim(-0.5, num_imfs - 0.5)
    plt.ylim(num_imfs - 0.5, -0.5)

    # Add the grid lines halfway between the ticks
    halfway_positions = np.arange(0.5, num_imfs - 0.5)
    for pos in halfway_positions:
        plt.axhline(pos, color='gray', linestyle=':', linewidth=0.5)
        plt.axvline(pos, color='gray', linestyle=':', linewidth=0.5)
    
    plt.show(block=True)
    
# --------------------------------------------------------------------------------
def weighted_median(values, weights):
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
    return sorted_values[median_idx]

# -------------------------------------------------------------------
def emd_funct(df,col_values,debug):
    # ------------------ EMD ---------------------------------------
    imf_opts = {'stop_method':'sd',
            'sd_thresh': 0.001}
    data2emd=df[col_values].values
    imf = emd.sift.sift(data2emd,
                        max_imfs=15,sift_thresh=1e-10,
                        return_residual=True,
                        imf_opts=imf_opts)
    
    if debug:
        fig, axs = plt.subplots(1, 1, figsize=(15, 30))
        # Create colormap object
        cmap = plt.cm.turbo
        emd.plotting.plot_imfs(imf,df.index,X=df[col_values].values, 
                               cmap=cmap,sharey=True, xlabel='Time (years)',step=5,ax=axs)
        axs.grid(lw=0.15)
        # Step 1: Compute the sum of all the IMFs
        imf_sum = np.sum(imf, axis=1)
        # ---------
        rmse=np.sqrt(np.mean((data2emd - imf_sum) ** 2))
        print('RMSE: ',rmse)
        io_values=calculate_index_of_orthogonality(imf)
        # plot
        index_orthogonality_plot(imf)
        
    return imf

def hilbert_deconvolution(imf,timescale_dict,debug):
    
    time_resolution_sec=timescale_dict['time_resolution_sec']
    WN,Signal,Climate=timescale_dict['WN_timescale'],timescale_dict['Signal_timescale'],timescale_dict['Climate_timescale']
    #  
    
    sample_rate=1/time_resolution_sec
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, method='nht')
    freq_edges, freq_centres = emd.spectra.define_hist_bins(data_min=1e-8, 
                                                            data_max=1e-3,
                                                            nbins=100,
                                                            scale='log')
    
    f1, spec_unweighted = emd.spectra.hilberthuang(IF, np.ones_like(IA),
                                                   freq_edges,
                                                   mode='amplitude',
                                                   scaling='density',
                                                   sum_imfs=False)
    # save the dominant frequency
    hilb_unweighted=[]
    for i in range(len(spec_unweighted[0])): 
        maxFreq_unweighted=np.log10(weighted_median(freq_centres, spec_unweighted[:, i]))
        hilb_unweighted.append(1/(10**maxFreq_unweighted)*(1/(time_resolution_sec*24)))
    hilb_unweighted=np.array(hilb_unweighted)
    
    # ------------ separate signal timescales -----------------------------------------
    saveWN      = np.where(hilb_unweighted <= WN)[0]
    saveSignal  = np.where((hilb_unweighted > WN) & (hilb_unweighted <= Signal))[0]
    saveClimate = np.where((hilb_unweighted > Signal) & (hilb_unweighted <= Climate))[0]
    saveSeasons = np.where(hilb_unweighted > Climate)[0]
    
    debug and print('imfs to use: ',saveSignal-1)
    # use just the event time scale signal
    event_signal = np.sum(imf[:, saveSignal], axis=1)
    
    return event_signal

def peak_pit_funct(df,signal_events,
                   event_param_dict,
                   input_dict,
                   debug):
    
    dist_timeSteps=event_param_dict['dist_timeSteps']
    quantile=event_param_dict['quantile']
    
    col_values=input_dict['col_values']
    col_time=input_dict['time_values']
    
    # ------------------- PEAKS ----------------
    upper_threshold=np.quantile(signal_events,quantile)
    upper_prominence=upper_threshold/10
    peaks_idx, dict_peaks=find_peaks(signal_events,
                                     distance=dist_timeSteps*2,
                                    height=upper_threshold,
                                    prominence=upper_prominence)
    
    # Extract peak times from Q_data index using FloodPeaks indices
    peaks_time=df.index[peaks_idx]
    peaks_val=dict_peaks['peak_heights']
    
    # Create DataFrame with peak heights and corresponding times
    peaks_df = pd.DataFrame({col_time: peaks_time, col_values: peaks_val})
    peaks_df[col_time]=pd.to_datetime(peaks_df[col_time])
    peaks_df=peaks_df.set_index(col_time)

    # sort peaks by intensity
    peaks_df = peaks_df.sort_values(by=col_values, ascending=False)
    debug and print('# flood events',len(peaks_df))

    #----------------- PITS--------------
    sigOppose=-(signal_events) # flip
    pits_idx,dict_pits=find_peaks(sigOppose,
                                  distance=dist_timeSteps//2,
                                  height=-upper_threshold,
                                  prominence=upper_prominence)
    
    pit_time = df.index[pits_idx]

    pits_val=-dict_pits['peak_heights']
    # Extract peak times from Q_data index using FloodPeaks indices
    pits_df =  pd.DataFrame({col_time: pit_time, col_values: pits_val})
    pits_df=pits_df.set_index(col_time)
    
    return [peaks_df,pits_df]

# -------------------------------------------------------------------
def peaks_clean_funct(df,peaks_df,pits_df,
                      event_param_dict,input_dict,debug):
    # prepare the variables from the dictionaries
    dist_timeSteps=event_param_dict['dist_timeSteps'  ]
    min_quantile  =event_param_dict['minimum_quantile']
    time_span     =event_param_dict['time_span'       ]
    time_span_days=event_param_dict['time_span_days'  ]
    # 
    col_values=input_dict['col_values' ]
    col_time  =input_dict['time_values']
    
    # set the time step
    dt=df.index[3]-df.index[2]
    
    # what is the time forward to look for the events
    pit_maxSearch_forward =dt*(dist_timeSteps)
    pit_maxSearch_backward=dt*(dist_timeSteps//2) # 
    # 
    peaks_df_clean=peaks_df.copy() # create cleaned dataframe
    peaks_df_clean[(col_time+'_init')]=pd.NaT
    peaks_df_clean[(col_time+'_end')]=pd.NaT
    debug and print('N events init algorithm',len(peaks_df_clean))
    # iterate through the peaks found
    for date,val_peak in peaks_df.iterrows():
        if date in peaks_df_clean.index:
            
            # 1. Calculate End of each event
            df_end=pits_df[pits_df.index>date].copy()
            if df_end.empty:
                # if dataframe is empty just chose the custom range of uncertainty
                pit_end=date+pit_maxSearch_forward
            else:
                pit_end=df_end.index[0] # take ithe first pit in line
            # -----------------------------------------------------------
            # 2. Calculate Init of each event
            df_init=pits_df[pits_df.index<date].copy()
            if (df_init.empty):
                pit_init=date-pit_maxSearch_forward
            else:    
                pit_init= df_init.index[-1]
            
            # --- crop the data to sarch the end and the init of each event
            data_end=df[(pit_end-pit_maxSearch_backward
                         ):(pit_end+pit_maxSearch_forward)].copy()
           
            data_init=df[(pit_init-pit_maxSearch_forward
                          ):(pit_init+pit_maxSearch_backward)].copy()
            
            # CONTROL over the dataseries, if within the expected ange there is not enough data, then skip
            if data_end[col_values].isna().all() or data_init[col_values].isna().all():
                debug and print(date,'event with no end or init')
                continue
            
            # 3. find the minimum real value in the vicinity of the identified pit
            realPit_end=np.nanmin(data_end[col_values])
            event_End=data_end[data_end[col_values]==realPit_end].index[0]
                
            # -----------------------------------------------------------
            realPit_init=np.nanmin(data_init[col_values])
            event_Init=data_init[data_init[col_values]==realPit_init].index[0]
            #
            #------------------------------------------------------------------------
            # 4. calculate the peak of the event
            data_peak=df[event_Init:event_End].copy() # crop the data to the specifics
            realPeak=np.nanmax(data_peak[col_values])
            peakTime=data_peak[data_peak[col_values]==realPeak].index[0]
            # 
            # 4. Update the dataframe, take out the events that fall within the event just calculated
            peaks_df_clean.loc[date,(col_time+'_init')]=event_Init
            peaks_df_clean.loc[date,(col_time+'_end')]=event_End
            peaks_df_clean.loc[date,col_values]=realPeak
           
            # Create a new row with peakTime index and the same values as the old 'date' row
            if (peakTime!=date):
                # substitute  peakTime to the original date
                peaks_df_clean.loc[peakTime] = peaks_df_clean.loc[date]
                # drop the date
                peaks_df_clean = peaks_df_clean.drop(date)
            
            # Find rows where the index values are between eventInit and eventEnd
            rows_to_delete = peaks_df_clean.loc[(peaks_df_clean.index >= event_Init) &
                                                (peaks_df_clean.index <= event_End)].index
            
            if (len(rows_to_delete)>1):
                # Iterate over the rows to delete
                for row_index in rows_to_delete:
                    if (row_index!=peakTime):# if it's not is own
                        if peaks_df_clean.loc[row_index,col_values]>peaks_df_clean.loc[peakTime,col_values]:
                            # if the flood is more important
                            peaks_df_clean.drop(index=row_index, inplace=True)
    
    # drop the non filled time slots
    peaks_df_clean=peaks_df_clean.dropna()
    debug and print('N events end algorithm ',len(peaks_df_clean))
    # ------------------------------------------------------------------------
    
    # df=df.replace([np.inf, -np.inf], np.nan).dropna()
    # clean excess of minima, use a time reference to delete the minimums
    month_quant = df.resample(time_span)[col_values].quantile(min_quantile)
    month_quant_df = month_quant.to_frame(name='quant')
    # 
    to_drop = []
    for month_end, row in month_quant_df.iterrows():
        month_tr=row.quant
        
        month_start=month_end - pd.DateOffset(days=time_span_days)
        mask=(peaks_df_clean.index>= month_start
              ) & (peaks_df_clean.index<= month_end)
        
        data_peaks = peaks_df_clean[mask]
        if data_peaks.empty:
            None
        else:
            to_drop.extend(data_peaks[data_peaks[col_values] < month_tr].index)       
    # drop the events
    debug and print('N events to drop',len(to_drop))
    df_out = peaks_df_clean.drop(to_drop) 
    return df_out


def flushing_events(col_time,name,debug):    
    # Cite Antoin G. et al. (2020) and report Julien Némery  and Jan De Keyser (2019)
    years= ['2007','2009','2010','2011','2012',
        '2014','2015','2016','2017','2018']
    start_dates = [
        ['2007-06-04 00:00:00', '2009-06-09 00:00:00', '2010-06-08 02:00:00', '2011-06-07 02:00:00'
        ,'2012-06-05 00:00:00','2014-06-17 00:00:00','2015-06-09 00:00:00','2016-06-16 00:00:00',
        '2017-06-14 00:00:00','2018-06-19 00:00:00'], # Pontamafrey
    
        ['2007-06-04 00:00:00', '2009-06-09 00:00:00', '2010-06-08 00:00:00', '2011-06-07 06:00:00'
        ,'2012-06-05 03:00:00','2014-06-17 03:00:00','2015-06-09 03:00:00','2016-06-16 03:00:00',
        '2017-06-14 03:00:00','2018-06-19 03:00:00'], # Aiguebelle

        ['2007-06-04 00:00:00', '2009-06-09 00:00:00', '2010-06-08 08:00:00', '2011-06-07 09:00:00'
        ,'2012-06-05 06:00:00','2014-06-17 06:00:00','2015-06-09 06:00:00','2016-06-16 06:00:00',
        '2017-06-14 06:00:00','2018-06-19 06:00:00'], # Montmelian

        ['2007-06-04 00:00:00', '2009-06-09 14:00:00', '2010-06-08 02:00:00', '2011-06-07 15:00:00'
        ,'2012-06-05 12:00:00','2014-06-17 12:00:00','2015-06-09 12:00:00','2016-06-16 12:00:00',
        '2017-06-14 12:00:00','2018-06-19 12:00:00']  # Grenoble_campus
    ]

    end_dates = [
        ['2007-06-06 00:00:00', '2009-06-10 04:00:00', '2010-06-08 18:00:00', '2011-06-08 00:00:00',
        '2012-06-06 00:00:00','2014-06-18 00:00:00','2015-06-10 00:00:00',
        '2016-06-17 00:00:00','2017-06-15 00:00:00','2018-06-20 00:00:00'], # Pontamafrey
        
        ['2007-06-06 00:00:00', '2009-06-10 00:00:00', '2010-06-09 00:00:00', '2011-06-08 03:00:00',
        '2012-06-06 03:00:00','2014-06-18 03:00:00','2015-06-10 03:00:00',
        '2016-06-17 03:00:00','2017-06-15 03:00:00','2018-06-20 03:00:00'], # Aiguebelle
        
        ['2007-06-06 00:00:00', '2009-06-10 00:00:00', '2010-06-09 00:00:00', '2011-06-08 06:00:00',
        '2012-06-06 06:00:00','2014-06-18 06:00:00','2015-06-10 06:00:00',
        '2016-06-17 06:00:00','2017-06-15 06:00:00','2018-06-20 06:00:00'], # Montmelian
        
        ['2007-06-06 00:00:00', '2009-06-10 14:00:00', '2010-06-09 18:00:00', '2011-06-08 12:00:00',
        '2012-06-06 12:00:00','2014-06-18 12:00:00','2015-06-10 12:00:00',
        '2016-06-17 12:00:00','2017-06-15 12:00:00','2018-06-20 12:00:00']  # Grenoble_campus
    ]

    list=['Pontamafrey','Aiguebelle','Montmelian','Grenoble_campus']

    # organize in a dictionary
    flush_dict = {}
    for station, start_dates_station, end_dates_station in zip(list, start_dates, end_dates):
        data = {'year': years,
                (col_time+'_init'): pd.to_datetime(start_dates_station),
                (col_time+'_end'): pd.to_datetime(end_dates_station)}
        flush_dict[station] = pd.DataFrame(data)
        flush_dict[station]=flush_dict[station].set_index('year')
    # print results
    if (name in list) and debug:
        print(flush_dict[name].head())

    return flush_dict[name]

def events2exclude(df,col_values,col_time,exclude_data,debug):
    # debug
    debug and print('numeber of events',len(df))
    init=[]
    end=[]
    val=[]
    idx_list=[]
    #  remove flushing / events not natural
    for idx,row in exclude_data.iterrows():
        start_date = row[(col_time+'_init')]
        end_date = row[(col_time+'_end')]
        mask_drop=((df.index > start_date) & (df.index < end_date))
        #
        # Select the events to be removed and add them to flushing_df
        removed_events = df[mask_drop]
        if not removed_events.empty:
            idx_list.append(removed_events.index[0])
            init.append(removed_events[(col_time+'_init')].iloc[0])
            end.append(removed_events[(col_time+'_end')].iloc[0])
            val.append(removed_events[col_values].iloc[0])
        df=df[~mask_drop]# clean the events list
    # save
    flushing_df=pd.DataFrame(columns=[col_values,
                                      (col_time+'_init'),
                                      (col_time+'_end')],
                             index=idx_list)
    
    flushing_df.loc[:,(col_time+'_init')]=init
    flushing_df.loc[:,(col_time+'_end')]=end
    flushing_df.loc[:,col_values]=val
    
    # debug 
    debug and print('number of natural events',len(df))
    return (df,flushing_df)

def plot_events(df,peaks_df_clean,flushing_df,col_values,col_time,name,variable):
    print('plot the event segmentation')
    width = 1000
    height = 500
    size=6
    
    layout = go.Layout(
    width=width,
    height=height,
    plot_bgcolor='rgb(250,250,250)',
    title=name,
    yaxis=dict(title=variable), # Secondary y-axis title
    xaxis=dict(rangeslider=dict(visible=False))
    )
    
    
    trace1 = go.Scatter(
    x=df.index, 
    y=df[col_values].values, 
    name=variable,
    line=dict(color='blue')  # Set the color for trace1 to blue
    )
    
    trace2 = go.Scatter(
    x=peaks_df_clean.index, 
    y=peaks_df_clean[col_values].values, 
    name='Events', 
    mode='markers',
    marker=dict(color='green', size=size)  # Customize marker appearance
    )
    
    trace3 = go.Scatter(
    x=peaks_df_clean[(col_time+'_end')].dt.strftime('%Y-%m-%d %H:%M:%S'), 
    y=df.loc[peaks_df_clean[(col_time+'_end')],col_values].values, 
    name=' End', 
    mode='markers',
    marker=dict(color='purple', size=size)  # Customize marker appearance
    )

    trace4 = go.Scatter(
        x=peaks_df_clean[(col_time+'_init')].dt.strftime('%Y-%m-%d %H:%M:%S'), 
        y=df.loc[peaks_df_clean[(col_time+'_init')],col_values].values,  
        name='Init', 
        mode='markers',
        marker=dict(color='orange', size=size)  # Customize marker appearance 
    )

    trace5 = go.Scatter(
        x=flushing_df.index, 
        y=flushing_df[col_values].values,  
        name='Flushing', 
        mode='markers',
        marker=dict(color='red', symbol='star',size=10)  # Customize marker appearance
    )

    # PREPARE THE DATA TO PLOT
    data2plot = [trace1,trace2,trace3,trace4,trace5]
    fw = go.Figure(data=data2plot, layout=layout)
    fw.show(block=False)


def plot_events_nat(df,peaks_df_clean,col_values,col_time,name,variable):
    print('plot the event segmentation')
    width = 1000
    height = 500
    size=6
    
    layout = go.Layout(
    width=width,
    height=height,
    plot_bgcolor='rgb(250,250,250)',
    title=name,
    yaxis=dict(title=variable), # Secondary y-axis title
    xaxis=dict(rangeslider=dict(visible=False))
    )
    
    
    trace1 = go.Scatter(
    x=df.index, 
    y=df[col_values].values, 
    name=variable,
    line=dict(color='blue')  # Set the color for trace1 to blue
    )
    
    trace2 = go.Scatter(
    x=peaks_df_clean.index, 
    y=peaks_df_clean[col_values].values, 
    name='Events', 
    mode='markers',
    marker=dict(color='green', size=size)  # Customize marker appearance
    )
    
    trace3 = go.Scatter(
    x=peaks_df_clean[(col_time+'_end')].dt.strftime('%Y-%m-%d %H:%M:%S'), 
    y=df.loc[peaks_df_clean[(col_time+'_end')],col_values].values, 
    name=' End', 
    mode='markers',
    marker=dict(color='purple', size=size)  # Customize marker appearance
    )

    trace4 = go.Scatter(
        x=peaks_df_clean[(col_time+'_init')].dt.strftime('%Y-%m-%d %H:%M:%S'), 
        y=df.loc[peaks_df_clean[(col_time+'_init')],col_values].values,  
        name='Init', 
        mode='markers',
        marker=dict(color='orange', size=size)  # Customize marker appearance 
    )

    # PREPARE THE DATA TO PLOT
    data2plot = [trace1,trace2,trace3,trace4]
    fw = go.Figure(data=data2plot, layout=layout)
    fw.show(block=False)
    
def save_events(df_events,df,df_aux,col_time,col_values_aux,
                path_output,file_name_output):
    print('saving',file_name_output)
    # prepare the dictionary to save
    data_event_dict = {}
    # Iterate through each row in peaks_df_clean
    for idx, row in df_events.iterrows():
        # Extract initial and end times
        init_time = row[(col_time + '_init')]
        end_time  = row[(col_time + '_end' )]
        
        # Crop the main variable DataFrame with the initial and end time
        mask=(df.index >= init_time) & (df.index <= end_time)
        cropped_data = df[mask]
        # crop auxiliary variables on the other timeseries
        mask_aux=(df_aux.index >= init_time) & (df_aux.index <= end_time)
        crop_aux=df_aux[mask_aux]
        # 
        save_data=cropped_data.copy()
        save_data = cropped_data.loc[crop_aux.index]
        if not save_data.empty:
            save_data[col_values_aux] = crop_aux[col_values_aux]
            # Append the discharge vector to the list
            data_event_dict[idx]=save_data
    
    # saving with pickle
    full_file_path = os.path.join(path_output, file_name_output)

    with open(full_file_path, 'wb') as f:
        pickle.dump(data_event_dict, f)
    
# ----------------CROP 1 HOUR DATA TO 10 MIN--------------------
def ssc_crop_h2min(df_data_q_1h,col_values_q,df_data_ssc_10m,
                   event_dict_ssc_1h,dt_range):
    event_dict_10min={}
    events_list=list(event_dict_ssc_1h.keys())

   
    for event_idx in events_list:
            df_event=event_dict_ssc_1h[event_idx]
            init=df_event.index[0]-dt_range
            end=df_event.index[-1]+dt_range

            # crop ssc data
            mask_ssc=(df_data_ssc_10m.index>= init)*(df_data_ssc_10m.index<= end)
            cropped_ssc_data=df_data_ssc_10m[mask_ssc].copy()
            
            # crop and interpolate q data
            mask_q=(df_data_q_1h.index>= init)*(df_data_q_1h.index<= end)
            cropQ=df_data_q_1h[mask_q].copy()
            interp_q_data=cropQ.reindex(cropped_ssc_data.index).interpolate(method='linear')

            cropped_ssc_data[col_values_q]=interp_q_data
            save_data=cropped_ssc_data.copy()

            if not save_data.empty:
                event_dict_10min[event_idx]=save_data

    return event_dict_10min


def ampfreq_analysis(imf,time_resolution_sec,cols):
    # smaple rate
    sample_rate=1/time_resolution_sec
    # df to store data
    list_imf=np.arange(1, imf.shape[1] + 1)
    df=pd.DataFrame(columns=cols,index=list_imf)

    # ---------- Relative energy ----------------
    E_imf=np.sum(imf**2,axis=0)
    E_total_imf=np.sum(E_imf)
    E_imf_rel=E_imf/E_total_imf

    # ---------- Timescale of each imf ----------------
    IP, IF, IA = emd.spectra.frequency_transform(imf,
                                                  sample_rate,
                                                    method='nht')
    freq_edges, freq_centres = emd.spectra.define_hist_bins(data_min=1e-8, 
                                                            data_max=1e-3,
                                                            nbins=100,
                                                            scale='log')
    
    f1, spec_unweighted = emd.spectra.hilberthuang(IF, np.ones_like(IA),
                                                   freq_edges,
                                                   mode='amplitude',
                                                   scaling='density',
                                                   sum_imfs=False)
    # save the dominant frequency
    timescale_days=[]
    for i in range(len(spec_unweighted[0])): 
        maxFreq_unweighted=np.log10(weighted_median(freq_centres, spec_unweighted[:, i]))
        timescale_days.append(1/(10**maxFreq_unweighted)*(1/(time_resolution_sec*24)))
    timescale_days=np.array(timescale_days)
    
    # save and return
    df.loc[:,cols[0]]=E_imf_rel
    df.loc[:,cols[1]]=timescale_days
    # 
    # drop trend
    trend_energy = df.iloc[-1, 0]  # energy of the last IMF (trend)
    df = df.iloc[:-2]  # drop the last row from df

    return df,trend_energy


def plot_ampfreq(imf,
                 imf_aux,
                 input_ampfreq_plot,
                 timescale_dict,debug):
    
    time_resolution_sec=timescale_dict['time_resolution_sec']
    cols=input_ampfreq_plot['cols']
    var_imf=input_ampfreq_plot['var_imf']
    var_imf_aux=input_ampfreq_plot['var_imf_aux']
    sample_rate=1/time_resolution_sec
    # 
    # timescales
    min_day= 0.1
    sub_day     =timescale_dict['WN_timescale']
    event_day   =timescale_dict['Signal_timescale']
    climate_day =timescale_dict['Climate_timescale']
    inter_annual=365
    max_day=365*4


    # itertate for main variable 
    debug and print('energy and freq analysis for ',var_imf)
    df,trend_df_energy=ampfreq_analysis(imf,time_resolution_sec,cols)

    # itertate for aux variable
    debug and print('energy and freq analysis for ',var_imf_aux)
    df_aux,trend_df_energy_aux=ampfreq_analysis(imf_aux,time_resolution_sec,cols)

    # ----------- plot -------------------------------
    fig, ax = plt.subplots(figsize=(13, 9))

    # Plot main variable (blue) and auxiliary variable (red)
    ax.plot(df[cols[1]], df[cols[0]], marker='o', linestyle='-', 
            color='blue', label=var_imf)
    ax.plot(df_aux[cols[1]], df_aux[cols[0]], marker='o', linestyle='-', 
            color='red', label=var_imf_aux)

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set labels and title
    fs=input_ampfreq_plot['fontsize']
    ax.set_xlabel(input_ampfreq_plot['xlabel'],fontsize=fs)
    ax.set_ylabel(input_ampfreq_plot['ylabel'],fontsize=fs)
    ax.set_title(input_ampfreq_plot['title'],fontsize=fs+2)
    
    # Create the legend and then add an annotation for trend energy below it.
   
    annotation_text = (f"Trend Energy: {trend_df_energy:.3f} (main), "
                       f"{trend_df_energy_aux:.3f} (aux)")

    # Add grid for easier reading of the log-scaled plot
    ax.grid(True, which='both', linestyle='--', lw=0.5)
    
    # -------------------------------
    # Draw vertical lines at the specified timescales
    timescales = [min_day,sub_day, event_day, climate_day, inter_annual,max_day]


    band_names = input_ampfreq_plot['band_names']
    band_colors = input_ampfreq_plot['band_colors']
    alpha_val = input_ampfreq_plot['band_alpha']

    # Draw vertical lines and shaded regions
    for i in range(1,len(timescales)-1):
        ax.axvline(x=timescales[i], color='grey', linestyle='--', linewidth=1)

    handles_for_legend = []

    for i in range(len(timescales) - 1):
        span = ax.axvspan(
            timescales[i], timescales[i + 1],
            color=band_colors[i],
            alpha=alpha_val,
            label=band_names[i]  # assign label for legend
        )

    handles_for_legend.append(span)
    line_legend = ax.legend(loc='best', fontsize=fs - 1)
    ax.add_artist(line_legend)

    
    # Add annotation box for trend energy + band info
    band_desc = "\n".join([
        f"{band_names[i]}: {timescales[i]}–{timescales[i+1]} days"
        for i in range(len(timescales)-2)
    ] + [
        f"{band_names[-1]}: > {timescales[-2]} days"
    ])

    annotation_text = (
    r"$\bf{Trend\ Energy:}$" + "\n"
    f"{var_imf}: {trend_df_energy:.3f}\n"
    f"{var_imf_aux}: {trend_df_energy_aux:.3f}\n\n"
    r"$\bf{Timescale:}$" + "\n" + band_desc
)

    ax.text(0.25, 0.02, annotation_text,
            transform=ax.transAxes,
            fontsize=fs-2,
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    ax.set_xlim(min_day,max_day)
    ax.set_ylim(0.0001,1.1)
    plt.tight_layout()
    plt.show()
    


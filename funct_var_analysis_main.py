from funct_reach import import_df,import_data_stationCouple,reach_morphometry
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from funct_var_analysis import (mass_conservation_routine,
                                plot_mass_error,
                                FPCA_regression,FPCA_Non_explained_plot,
                                FPCA_variables_classification,
                                # ssc_classification,
                                # ------------------------------
                                # FPCA_mass_regression,
                                mass_classification)

#  estimate the simulation relative uncertainty based on the mass conservation error
def simulation_uncertainty(station_couple,
                 input_dict_q_1h,input_dict_ssc_10m,
                 input_sim_station_difference,
                 input_dict_plot_mass,
                 input_dict_FPCA,
                 debug):
    
    #  import data 
    q_dict_data  =import_data_stationCouple(input_dict_q_1h  ,debug )
    ssc_dict_data_10m=import_data_stationCouple(input_dict_ssc_10m,debug )
    
    # channel morphology
    geom_dict_system=reach_morphometry(debug)
    geom_dict=geom_dict_system[(station_couple['upstream']+':'+station_couple['downstream'])]
    
    # import simulation data
    sim_dict =import_df(input_dict_FPCA      ,debug )
    reach_hydro_df= import_df(input_sim_station_difference,debug)
    
 
    rel_mass_df,delta_vars_dict=mass_conservation_routine(station_couple,
                                           q_dict_data,ssc_dict_data_10m,
                                           reach_hydro_df,geom_dict,
                                           input_sim_station_difference,
                                           input_dict_FPCA,sim_dict,
                                           debug)
    if debug:
        plot_mass_error(rel_mass_df,input_dict_plot_mass,debug)
    
    
    return delta_vars_dict
# ---------------------------------------------------------
    
def FPCA_regression_main(input_dict_PCA,
                             debug):
            # import simulation data
        sim_dict =import_df(input_dict_PCA      ,debug )
        # 
        event_variable_df,_=FPCA_regression(sim_dict,
                                   input_dict_PCA,
                                    debug)
        
        FPCA_Non_explained_plot(event_variable_df)
        FPCA_variables_classification(event_variable_df,
                                             input_dict_PCA,
                                             debug)

def event_mass_analysis(station_couple,
                        input_dict_q_1h,input_dict_ssc_10m,
                        input_sim_station_difference,
                        input_dict_FPCA,
                        # ---------- plotting------------
                        input_dict_massVar_plot,
                        input_dict_emd,
                        debug):

        
        # import the simulation results
        sim_dict =import_df(input_dict_FPCA      ,debug )
        
        # # # # --------- Mass analysis ------------------
        # channel morphology
        geom_dict_system=reach_morphometry(debug)
        geom_dict=geom_dict_system[(
            station_couple['upstream']+':'+station_couple['downstream'])]
        
        # perform the FPCA regression 
        debug and print('>> FPCA_regression *main*<<')
        event_variable_df,input_FPCA_variables=FPCA_regression(
                                                        sim_dict,
                                                        input_dict_FPCA,
                                                        geom_dict,
                                                        debug)
        
        
        #  import data 
        q_dict_data  =import_data_stationCouple(input_dict_q_1h  ,debug )
        ssc_dict_data_10m=import_data_stationCouple(input_dict_ssc_10m,debug )

        # reach hydro info
        reach_hydro_df= import_df(input_sim_station_difference,debug)

        
        
        debug and print('>> mass_conservation_routine *main*<<')
        rel_mass_df,delta_vars_dict=mass_conservation_routine(station_couple,
                                           q_dict_data,ssc_dict_data_10m,
                                           reach_hydro_df,geom_dict,
                                           input_sim_station_difference,
                                           input_dict_FPCA,sim_dict,
                                           debug)
        
        #  analysis of the mass variations
        debug and print('>> mass_classification *main*<<')
        mass_classification(station_couple,
                        q_dict_data, ssc_dict_data_10m,
                        rel_mass_df,sim_dict,
                        event_variable_df,
                        input_FPCA_variables,
                        # 
                        input_sim_station_difference,
                        input_dict_FPCA,
                        input_dict_massVar_plot,
                        input_dict_emd,
                        debug)
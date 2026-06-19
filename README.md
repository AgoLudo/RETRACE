[![DOI](https://zenodo.org/badge/1073848148.svg)](https://doi.org/10.5281/zenodo.20760539)
# RETRACE: Routing & Event Time-differencing for Response Analysis of mass Changing Events

This repository contains a Python-based framework for analyzing hydrological and sediment transport dynamics between two monitoring stations. The codebase, named RETRACE, processes time-series data of discharge (Q) and suspended sediment concentration (SSC) to characterize event-based transport phenomena between upstream and downstream monitoring stations.

## Core Modules

The analysis is structured into several distinct modules, which can be executed sequentially from the main [`run_main.py`](run_main.py ) script.

### Module 1: Extraction of the mass-transporting events
-   **File:** [`funct_event_segmentation_main.py`](funct_event_segmentation_main.py )
-   **Description:** Identifies significant mass-transporting events from raw time-series data. It uses Empirical Mode Decomposition (EMD) via the [`emd_funct`](funct_event_segmentation.py ) to separate signals into different timescales, allowing for robust event detection based on statistical thresholds defined in [`main_input.py`](main_input.py ).

### Module 2: Estimation of the virtual velocity
-   **File:** [`funct_reach_main.py`](funct_reach_main.py )
-   **Description:** This module, driven by the [`time_lag_analysis`](funct_reach_main.py ) function, calculates the time lag of Q and SSC signals between two stations for each identified event. It employs a cross-correlation method ([`cross_corr_events`](funct_reach.py )) enhanced with the Hilbert transform to accurately determine the lag, which is then used to compute the virtual celerities ($\lambda_Q$ and $\lambda_{C}$).

### Module 3: Fitting of a reach-averaged rating curve
-   **File:** [`funct_reach_main.py`](funct_reach_main.py )
-   **Description:** The [`reach_scale_prop`](funct_reach_main.py ) function establishes reach-averaged hydraulic properties. It derives relationships between discharge, flow velocity, and flow depth, and uses these to estimate reach-scale roughness coefficients, partly based on the methodologies from Rickenmann D. and Recking, A. (2011) implemented in [`recking_universal_roughness`](funct_reach.py ).

### Module 4: Signal routing
-   **File:** [`fuct_analytics_main.py`](fuct_analytics_main.py )
-   **Description:** This module simulates the propagation of a flood wave or sediment pulse down the reach. The [`station_diff`](fuct_analytics_main.py ) function applies an advection-diffusion model ([`adv_diff_wave`](funct_analytics.py )) to route the upstream signal to the downstream location. It also includes functions to test the numerical solution against known analytical solutions ([`test_function`](fuct_analytics_main.py )).

### Module 5: Time differencing interpretation
-   **File:** [`funct_var_analysis_main.py`](funct_var_analysis_main.py )
-   **Description:** Investigates the sources of variability and error in sediment mass conservation. This module uses Functional Principal Component Analysis (FPCA) in [`FPCA_regression`](funct_var_analysis.py ) to classify events and identify controlling variables.
### Module 6: Energy-spectrum analysis
-   **File:** [`funct_var_analysis_main.py`](funct_var_analysis_main.py )
-   **Description:** Furthermore, from the previous Module, [`emd_event`](funct_var_analysis.py ) performs an energy-spectrum analysis using the Hilbert-Huang Transform (HHT) to understand how signal power is distributed across different timescales.

## Workspace Structure

```
.
├── run_main.py                     # Main executable script to run the modules
├── main_input.py                   # Central configuration file for all parameters
│
├── funct_event_segmentation_main.py # Module 1 main script
├── funct_event_segmentation.py     # Module 1 core functions
│
├── funct_reach_main.py             # Modules 2 & 3 main script
├── funct_reach.py                  # Modules 2 & 3 core functions
│
├── fuct_analytics_main.py          # Module 4 main script
├── funct_analytics.py              # Module 4 core functions
│
├── funct_var_analysis_main.py      # Module 5 & 6 main script
├── funct_var_analysis.py           # Module 5 & 6 core functions
.
```

## How to Run

1.  **Configure Parameters:** Open [`main_input.py`](main_input.py ) to define the [`station_couple`](fuct_analytics_main.py ), input data paths, and parameters for each analysis step.
2.  **Select Modules:** Open [`run_main.py`](run_main.py ) and uncomment the sections corresponding to the analysis modules you wish to execute.
3.  **Execute:** Run the main script from your terminal.

    ```sh
    python run_main.py
    ```

## Dependencies

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scipy`
-   `emd`

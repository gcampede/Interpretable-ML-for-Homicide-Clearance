# Explainable ML for Homicide Clearance
This repository contains data and Python code to replicate the analysis presented in the paper "Explainable Machine Learning for Predicting and Explaining Cleared Homicides in the United States" (tentative title).

Data for this project and paper have to be downloaded directly at http://www.murderdata.org/p/data-docs.html. Data credits go to the *Murder Accountability Project*, an initiative seeking to improve transparency and accountability of law enforcement in relation to murder investigations in the United States.

The repository contains three folders:

  1. National_Level
  2. State_Level
  3. Robustness
  4. Additional_Graphics


## National_Level

The folder contains two scripts: "US_grid_search_models.py" and "US_SHAP_explanation.py". The first one should be used to perform national-level grid search for the nine selected algorithmic approaches. The second instead performs SHAP explanation modeling using the best performing algorithm (and configuration) found after grid search. Pre-processed pickle files are available for replication. Original data have to be gathered directly from the MAP website.

## State_Level

Similarly, this folder also contains two scripts: "state_wise_prediction.py" and "state_wise_SHAP.py". The first one performs grid search for XGBoost hyperparameters on each state dataset. The second, perform SHAP explanation modeling using the best configuration of XGBoost per each state, as found in the previous script. Pre-processed pickle files are available for replication. Original data have to be gathered directly from the MAP website

## Robustness

This folder contains the script to compare the explainability outcomes cross checking data from the MAP dataset with data collected by the Washington Post on homicides from the "Unsolved Homicides" database at https://github.com/washingtonpost/data-homicides. 

## Additional_Graphics

This folder contains two scripts: "descriptive_visualization.py" and "visualization_US_gridsearched_models". These contain code for replicating some of the figures presented in the paper, particularly those referring to description of the dataset and evaluation of national level grid-searched models. Code for the other plots is embedded in the other scripts contained in the other two folders. 

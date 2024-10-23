# DSC180A Capstone Q1

** Group 1: ** Kevin Wong, Kurumi Kaneko, Hillary Chang, Jevan Chahil

## Overview

This project implements a data pipeline for Quarter 1 of our DSC Capstone. For this project specifically, we are implementing a categorization model to categorize memos given by vendors. For instance, given a memo of "Amazon.com", we would want to create a model that can categorize this into "General Merchandise" with a high accuracy, while also being fast with low latency.

## Running the Project

### Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt

## File Structure

```

├── data
│   ├──ucsd-inflows.pqt   <- Inflow data (money getting put into bank)
│   └──ucsd-outflows.pqt  <- Outflow data (money getting out of bank)
│   
│                     
├── model             <- best model saved in pickle file   
│   └──xgb_reg.pkl 
│
│
├── plots                          <- plots created using bokeh, plotly, matplotlib, and seaborn
│   ├──demographics          
│   ├──descriptive_statistics     
│   ├──food
│   ├──heritability
│   ├──mode_prediction
│   ├──physical_activity
│   ├──sleep
│   └──Visualization.pdf
│
├── res  
│   └──predicted_result.csv     <- predicted model results
│
│
├── src                             <- src files to create the plots and train models
│   ├──all_visualization.ipynb      <- Contains visualization of all the analysis in a single .ipynb file
│   ├──demographics_sleep.ipynb     
│   ├──descriptive_stats.ipynb
│   ├──exercise_food.ipynb
│   ├──heritability.ipynb
│   ├──model.ipynb
│   └──model.py                 <- Contains model processing, transformation, normalization, training, and testing  in a single .py file 
│   
│
├── ECE_143_Final.pdf           <- Final Presentation
│
├── readme.md                   <- README
│
└── requirement.yaml            <- all required packages 
```
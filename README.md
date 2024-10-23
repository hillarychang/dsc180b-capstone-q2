# DSC180A Capstone Q1: Categorizing Memo

**Group 1:** Kevin Wong, Kurumi Kaneko, Hillary Chang, Jevan Chahal

## Overview

This project implements a data pipeline for Quarter 1 of our DSC Capstone. For this project specifically, we are implementing a categorization model to categorize memos given by vendors. For instance, given a memo of "Amazon.com", we would want to create a model that can categorize this into "General Merchandise" with a high accuracy, while also being fast with low latency.

## Running the Project

### Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### File Structure

```
├── data                            <- .gitignore (hidden from repo)
│   ├──ucsd-inflows.pqt             <- Inflow data (money getting put into bank)
│   └──ucsd-outflows.pqt            <- Outflow data (money getting out of bank)
│   
│                     
├── config                          <- parameters for model (currently all dummy files)
│   ├──data_params.json     
│   ├──feature_params.json
│   └──model_params.json
│
│
├── notebooks                       <- jupyter notebooks used for data analysis
│   ├──hillary_data_exploration.ipynb <- Hillary Chang's Data Analysis Notebook         
│   ├──kevin_data_exploration.ipynb <- Kevin Wong's Data Analysis Notebook         
│
├── res  
│   └──predicted_result.csv     <- predicted model results
│
│
├── src         <- src files for the creation of features, model training, and more (currently all dummy files)
│   ├──etl.py         
│   ├──features.py
│   └──model_training.py                  
│
├── README.md                   <- README
│
├── run.py                      <- used to run the entire project
│
└── requirements.txt            <- all required packages 
```

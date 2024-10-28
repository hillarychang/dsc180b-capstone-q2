# DSC180A Capstone Q1: Categorizing Memo

**Group 1:** Kevin Wong, Kurumi Kaneko, Hillary Chang, Jevan Chahal

## Overview

This project implements a data pipeline for Quarter 1 of our DSC Capstone. For this project specifically, we are implementing a categorization model to categorize memos given by vendors. For instance, given a memo of "Amazon.com", we would want to create a model that can categorize this into "General Merchandise" with a high accuracy, while also being fast with low latency.

## Running the Project

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/categorizing-memos.git
    cd categorizing-memos
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the data**:  
    Place the inflow and outflow data (`ucsd-inflows.pqt`, `ucsd-outflows.pqt`) in the `data/` directory. This data contains the transactions we will use for memo categorization.

4. **Run the data pipeline**:  
    To run the entire pipeline, use the following command:
    ```bash
    python run.py
    ```

### File Structure

```
├── data                            <- .gitignore (data not tracked in the repository)
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
├── notebooks                         <- jupyter notebooks used for data analysis
│   ├──hillary_data_exploration.ipynb <- Hillary Chang's Data Analysis Notebook         
│   ├──kevin_data_exploration.ipynb   <- Kevin Wong's Data Analysis Notebook
│   ├──kurumi_data_exploration.ipynb   <- Kurumi Kaneko's Data Analysis Notebook
│   ├──jevan_week2.ipynb              <- Jevan Chahal's notebook for train/test split
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

## Conclusion
The goal of this project for Quarter 1 is to categorize vendor memos. Next, we will be implementing a data pipeline to process inflow and outflow data, create features, and train a machine learning model to achieve this.

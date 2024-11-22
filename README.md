# DSC180A Capstone Q1: Categorizing Memo

**Group 1:** Kevin Wong, Kurumi Kaneko, Hillary Chang, Jevan Chahal

## Overview

This project implements a data pipeline for Quarter 1 of our DSC Capstone. For this project specifically, we are implementing a categorization model to categorize memos given by vendors. For instance, given a memo of "Amazon.com", we would want to create a model that can categorize this into "General Merchandise" with a high accuracy, while also being fast with low latency.

## Running the Project

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hillarychang/dsc180a-capstone-q1.git
    cd dsc180a-capstone-q1
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
├── config                               <- parameters for model (currently all dummy files)
│   ├──data_params.json     
│   ├──feature_params.json
│   └──model_params.json
│
│
├── data                                 <- .gitignore (data not tracked in the repository)
│   ├──ucsd-inflows.pqt                  <- Inflow data (money getting put into bank)
│   └──ucsd-outflows.pqt                 <- Outflow data (money getting out of bank)   
│
│
├── models                               <- Model Creation
│   ├──transformers.py                   <- Used to initialize the Transformer Model
│
│
├── src                               <- src files for the creation of features, model training, and more
│   ├──base                           <- Baseline Classes
│   │   ├──config.py                  <- Creates a configuration for all models
│   │   └──model.py                   <- Creates a Baseline Model
│   ├──configs                        <- Used to create baseline configurations for different models
│   │   └──transformer.yml            <- Creates baseline parameters for the transformer model
│   ├──models                         <- Contains all model code
│   │   └──transformer.py             <- Transformer Model code
│   ├──etl.py         
│   ├──features.py                    <- used to create features
│   ├──text_cleaner.py                <- used to clean the data 
│   ├──model_training.py              <- used to train the model
│   ├──notebooks                      <- Jupyter notebooks used for data analysis
│   │   ├──baseline_models.ipynb      <- Perform logistic regression and random forest with tf-idf
│   │   ├──hillary_data_exploration.ipynb    <- Hillary Chang's Data Analysis Notebook         
│   │   ├──kevin_data_exploration.ipynb      <- Kevin Wong's Data Analysis Notebook
│   │   ├──kurumi_data_exploration.ipynb     <- Kurumi Kaneko's Data Analysis Notebook
│   │   ├──kurumi_feature_engineering.ipynb  <- Kurumi Kaneko's Notebook for Feature Creation
│   │   └──jevan_week2.ipynb                 <- Jevan Chahal's notebook for train/test/split
│
│
├── report                            <- used to keep the report LaTeX pdf
│   ├──figure                         <- figures for the report
│   │   ├──amt_category.png       
│   │   ├──category_time.png
│   │   ├──clean_df.jpeg
│   │   ├──inflow.png
│   │   ├──nonclean_df.png  
│   │   └──outflow.png        
│   └──report.tex   
│
│
├── README.md                         <- README
│
├── requirements.txt                  <- all required packages
│
└── run.py                            <- used to run the model
```

## Conclusion
The goal of this project for Quarter 1 is to categorize vendor memos. Next, we will be implementing a data pipeline to process inflow and outflow data, create features, and train a machine learning model to achieve this.

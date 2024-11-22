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
├├── config                               <- parameters for model (currently all dummy files)
│   ├── data_params.json     
│   ├── feature_params.json
│   └── model_params.json
│
├── data                                 <- .gitignore (data not tracked in the repository)
│   ├── ucsd-inflows.pqt                 <- Inflow data (money getting put into bank)
│   └── ucsd-outflows.pqt                <- Outflow data (money getting out of bank)   
│
├── report                               <- used to keep the report LaTeX pdf
│   ├── figure                           <- figures for the report
│   │   ├── amt_category.png       
│   │   ├── category_time.png
│   │   ├── clean_df.jpeg
│   │   ├── inflow.png
│   │   ├── nonclean_df.png  
│   │   └── outflow.png        
│   ├── report.tex                       <- Quarter 1 report
│   └── reportq2.tex                     <- Quarter 2 report
│
├── src                                  <- src files for the creation of features, model training, and more
│   ├── base                             <- Baseline Classes
│   │   ├── config.py                    <- Creates a configuration for all models
│   │   └── model.py                     <- Creates a Baseline Model
│   ├── configs                          <- Used to create baseline configurations for different models
│   │   └── transformer.yml              <- Creates baseline parameters for the transformer model
│   ├── models                           <- Contains all model code
│   │   ├── distilbert_classifier.py     <- DistilBERT Model for classification
│   │   ├── fasttext_function.py         <- FastText Model for classification
│   │   ├── logistic_regression_function.py <- Logistic Regression Model for classification
│   │   ├── random_trees_function.py     <- Random Forest Model for classification
│   │   └── transformer.py               <- Transformer Model code
│   ├── notebooks                        <- Jupyter notebooks for data analysis and feature creation
│   │   ├── baseline_models.ipynb              <- Perform logistic regression and random forest with tf-idf
│   │   ├── hillary_data_exploration.ipynb     <- Hillary Chang's Data Analysis Notebook         
│   │   ├── kevin_data_exploration.ipynb       <- Kevin Wong's Data Analysis Notebook
│   │   ├── kevin_inflows_exploration.ipynb    <- Kevin Wong's Data Analysis Notebook
│   │   ├── kurumi_data_exploration.ipynb      <- Kurumi Kaneko's Data Analysis Notebook
│   │   ├── kurumi_feature_engineering.ipynb   <- Kurumi Kaneko's Notebook
│   │   ├── kurumi_inflows_data_exploration.ipynb <- Kurumi Kaneko's Notebook
│   │   └── jevan_quarter1_project.ipynb       <- Jevan Chahal's Notebook
│   ├── etl.py       
│   ├── features.py                      <- Used to create features
│   ├── text_cleaner.py                  <- Used to clean the data 
│   └── model_training.py                <- Used to train the model
│
├── README.md                            <- README
│
├── requirements.txt                     <- All required packages
│
└── run.py                               <- Used to run the model
```

## Conclusion
The goal of this project for Quarter 1 is to categorize vendor memos. Next, we will be implementing a data pipeline to process inflow and outflow data, create features, and train a machine learning model to achieve this.

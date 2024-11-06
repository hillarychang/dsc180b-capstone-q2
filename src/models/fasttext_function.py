#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fasttext
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Redirecting stdout to suppress detailed output during training
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def fastext_function(data):
    # Load your data
    # Assuming `w_cleaned_outflows` is your DataFrame with columns 'memo' and 'category'
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['memo'], 
        data['category'], 
        test_size=0.2, 
        random_state=42
    )
    print('Data split complete')

    # Save the training data in the format FastText expects (each line: __label__<label> <text>)
    train_file = 'train_data.txt'
    test_file = 'test_data.txt'

    # Save train data
    with open(train_file, 'w') as f:
        for text, label in zip(X_train_text, y_train):
            f.write(f'__label__{label} {text}\n')

    # Save test data
    with open(test_file, 'w') as f:
        for text, label in zip(X_test_text, y_test):
            f.write(f'__label__{label} {text}\n')

    # Suppress output during training
    with SuppressOutput():
        # Train a FastText model for text classification
        model = fasttext.train_supervised(input=train_file, epoch=25, lr=1.0, wordNgrams=2, bucket=200000, dim=50)
    print('FastText model training complete')

    # Predict categories for the test set
    y_pred = [model.predict(text)[0][0].replace('__label__', '') for text in X_test_text]
    print('FastText categorization complete')

    # Evaluating the predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Generate a classification report
    class_report = classification_report(y_test, y_pred, target_names=list(set(y_test)))
    print("\nClassification Report:\n", class_report)

    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=list(set(y_test)))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list(set(y_test)), columns=list(set(y_test)))
    print("\nConfusion Matrix:\n", conf_matrix_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





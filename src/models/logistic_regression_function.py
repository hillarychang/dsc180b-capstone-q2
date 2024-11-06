#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd


def logistic_regression_function(data):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['memo'], 
        data['category'], 
        test_size=0.2, 
        random_state=42
    )
    print('Data split complete')

    # Vectorizing the text data - fit only on training data, transform on test data
    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.95, min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print('Vectorization complete')

    # Encoding the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    print('Label encoding complete')

    # Fitting the Logistic Regression model
    log_reg = LogisticRegression(solver='saga', max_iter=200, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    print('Logistic Regression model trained')

    # Making predictions and calculating accuracy
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Generating classification report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nClassification Report:\n", class_report)

    # Generating and displaying confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nConfusion Matrix:\n", conf_matrix_df)

    # Getting predicted probabilities
    y_prob = log_reg.predict_proba(X_test)

    # Calculating ROC-AUC score for each category individually
    roc_auc_scores = {}
    for i, category in enumerate(label_encoder.classes_):
        y_test_binary = np.where(y_test == i, 1, 0)  # Binary label for the current category

        # Only calculate ROC-AUC if there are both positive and negative samples
        if len(np.unique(y_test_binary)) == 2:
            roc_auc_scores[category] = roc_auc_score(y_test_binary, y_prob[:, i])
        else:
            roc_auc_scores[category] = "Undefined (only one class in y_test)"

    # Displaying the ROC-AUC score for each category
    print("ROC-AUC Scores per Category:")
    for category, score in roc_auc_scores.items():
        print(f"{category}: {score}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





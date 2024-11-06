#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

def random_trees_function(data):
    # Splitting the data into train and test sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['memo'], 
        data['category'], 
        test_size=0.2, 
        random_state=42
    )
    print('Data split complete')

    # Vectorizing the text data - fit only on training data, transform on test data
    vectorizer = TfidfVectorizer(max_features=2000, max_df=0.95, min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print('Vectorization complete')

    # Encoding the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    print('Label encoding complete')

    # Fitting the Random Forest model
    ran_for = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=60)
    ran_for.fit(X_train, y_train)
    print('Random Forest model trained')
    print('-----------------------------------------------------------------')
    print()

    # Making predictions and calculating accuracy
    y_pred = ran_for.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print('-----------------------------------------------------------------')
    print()

    # Generating classification report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nClassification Report:\n", class_report)

    # Generating and displaying confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nConfusion Matrix:\n", conf_matrix_df)

    # Getting predicted probabilities
    y_prob = ran_for.predict_proba(X_test)

    # Binarizing labels for multi-class ROC AUC
    y_test_binarized = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

    # Calculating ROC-AUC score for each category individually
    roc_auc_per_class = roc_auc_score(y_test_binarized, y_prob, average=None)
    for idx, auc in enumerate(roc_auc_per_class):
        print(f"ROC AUC for category {label_encoder.classes_[idx]}: {auc:.4f}")
    print('-----------------------------------------------------------------')
    print()

    # Calculating accuracy for each category (one-vs-all approach)
    for category in range(len(label_encoder.classes_)):
        y_test_binary = (y_test == category).astype(int)
        y_pred_binary = (y_pred == category).astype(int)
        accuracy_per_category = accuracy_score(y_test_binary, y_pred_binary)
        print(f"Accuracy for category {label_encoder.classes_[category]}: {accuracy_per_category:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





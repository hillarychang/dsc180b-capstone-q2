import pandas as pd
import numpy as np
import warnings
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from joblib import dump
from joblib import load
import shap 

# Scikit-Learn Components
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.exceptions import UndefinedMetricWarning

# Imbalanced Learn
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

# XGBoost
from xgboost import XGBClassifier

# CatBoost
from catboost import CatBoostClassifier

# LightGBM
import lightgbm as lgb

# Hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# preprocess features and train test split
def preprocess(dataset):
    # Replace infinite values
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    # Drop or fill NaN values
    dataset = dataset.fillna(
        dataset.median()
    )  # Or use mean, or a specific strategy

    return dataset


def preprocess_features(feature_column, target_column, dataset, test_size = 0.2, random_state = 42):
    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
    dataset = dataset.dropna()

    # Define features and targett
    X = dataset[feature_column]
    X = preprocess(X)
    y = dataset[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    
    resampler = SMOTETomek(sampling_strategy = 1.0, random_state=random_state)
    X_train, y_train = resampler.fit_resample(X_train, y_train)
    train_id = X_train.index
    test_id = X_test.index
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, filename="scaler.joblib")
    return X_train, X_test, y_train, y_test, train_id, test_id

def individual_test(feature_column, name, dataset, random_state=42):
    consumer_id = dataset.index
    trained_model = load(f"{name}.joblib")
    scaler = load('scaler.joblib')
    X = dataset[feature_column]
    X = preprocess(X)
    X = scaler.transform(X)
    probabilities = trained_model.predict_proba(X)[:, 1]
    probabilities = np.round(1 + probabilities * (999 - 1)).astype(int)
    top_3_features, top_3_scores = shap_values(trained_model, X, feature_column)
    scores_df = pd.DataFrame({
        'probability': probabilities,
        'top_1_feature': [features[2] for features in top_3_features],
        'top_1_score': [scores[2] for scores in top_3_scores],
        'top_2_feature': [features[1] for features in top_3_features],
        'top_2_score': [scores[1] for scores in top_3_scores],
        'top_3_feature': [features[0] for features in top_3_features],
        'top_3_score': [scores[0] for scores in top_3_scores]
    }, index=consumer_id)

    value_counts_plot = (scores_df.top_1_feature.value_counts()
    .add(scores_df.top_2_feature.value_counts(), fill_value=0)
    .add(scores_df.top_3_feature.value_counts(), fill_value=0))

    value_counts_plot = value_counts_plot.sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(value_counts_plot.index[:7], value_counts_plot[:7])
    plt.xlabel('Feature')
    plt.ylabel('Top Three Features Count')
    plt.title(f'Top {7} Features by Top Three Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    return scores_df

def shap_values(model, X_train, feature_column):
        explainer = shap.TreeExplainer(model)
            
        shap_values = explainer.shap_values(X_train)
            
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap.summary_plot(shap_values, X_train, feature_names=feature_column, max_display=10)

        import numpy as np


        top_3_indices = np.argsort(shap_values, axis=1)[:, -3:] 
        top_3_values = np.take_along_axis(shap_values, top_3_indices, axis=1)  
        max_shap_feature_per_user = [[feature_column[i] for i in user_indices] for user_indices in top_3_indices]
        
        shap_value_means = np.mean(np.abs(shap_values), axis=0)  # axis=0 to get feature-wise means

        # Get indices of top 3 features
        top_x_indices_mean = np.argsort(shap_value_means)[-5:] # Last 3 (highest values)

        # Get the top 3 feature names and values
        top_x_features_mean = feature_column[top_x_indices_mean]
        top_x_values_mean = shap_value_means[top_x_indices_mean]

        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar(np.flip(top_x_features_mean), np.flip(top_x_values_mean))
        plt.xlabel('Feature')
        plt.ylabel('Absolute Mean SHAP Value')
        plt.title(f'Top {5} Features by Absolute Mean SHAP Value')
        plt.xticks(rotation=45, ha='right')
        plt.show()


        return (max_shap_feature_per_user, top_3_values)

# run_classification models
def run_classification(
    feature_column, target_column, dataset, random_state=42
):
    """
    Enhanced classification analysis with multiple models and comprehensive reporting.
    """
    X_train, X_test, y_train, y_test, train_id, test_id = preprocess_features(feature_column, target_column, dataset)

    models = [
        # Core Models
        (
            LogisticRegression(class_weight="balanced", max_iter=200),
            "Logistic Regression",
        ),
        (RandomForestClassifier(random_state=random_state), "Random Forest"),
        (
            lgb.LGBMClassifier(
                objective="binary",
                verbose=-1,
                force_row_wise=True,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
            "LightGBM",
        ),
        (BalancedRandomForestClassifier(random_state=random_state), "Balanced RF"),
        # Gradient Boosting Family
        (
            XGBClassifier(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
            ),
            "XGBoost",
        ),
        (
            CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3, silent=True
            ),
            "CatBoost",
        ),
        (
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=6
            ),
            "HistGB",
        ),
        (
            RUSBoostClassifier(
                n_estimators=200, learning_rate=0.05, random_state=random_state
            ),
            "RUSBoost",
        ),
    ]

    # Model evaluation framework
    model_results = []
    roc_curves = []
    model_predictions = {}

    def evaluate_model(model, name):
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            dump(model, filename=f"{name}.joblib")
            train_time_end = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            model_predictions[name] = y_proba

            # Calculate metrics
            metrics = {
                "model": name,
                "roc_auc": roc_auc_score(y_test, y_proba),
                "accuracy": accuracy_score(y_test, y_pred),
                "train_time": train_time_end - start_time,
                "predict_time": (time.time() - train_time_end) / X_test.shape[0]
            }

            # Add classification report metrics
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            for k, v in clf_report["weighted avg"].items():
                metrics[k] = v
            model_results.append(metrics)

            # Print model report
            print(f"\n\033[1m{name} Results\033[0m")
            print(
                f"ROC-AUC: {metrics['roc_auc']:.3f} | Accuracy: {metrics['accuracy']:.3f}"
            )
            print(f"Training Time: {metrics['train_time']:.1f}s | Predicting Time: {metrics['predict_time']:.6f}s")
            print(classification_report(y_test, y_pred))

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves.append((fpr, tpr, name, metrics["roc_auc"]))

            # Plot Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{name} - Confusion Matrix")
            plt.show()

        except Exception as e:
            print(f"\n\033[91mError in {name}: {str(e)}\033[0m")

    # Execute all models
    for model, name in models:
        evaluate_model(model, name)
        # if name in ["LightGBM", "XGBoost", "HistGB", "CatBoost"]:
            # evaluate_model(model, name)

    # Results analysis
    results_df = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)

    if model_predictions:
        y_proba_ensemble = np.max(np.array(list(model_predictions.values())), axis=0)
        y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)

        # Evaluate ensemble performance
        ensemble_metrics = {
            "model": "Ensemble (Max Prob)",
            "roc_auc": roc_auc_score(y_test, y_proba_ensemble),
            "accuracy": accuracy_score(y_test, y_pred_ensemble),
        }

        # Add ensemble results to dataframe
        results_df = pd.DataFrame(model_results)
        results_df = pd.concat(
            [results_df, pd.DataFrame([ensemble_metrics])], ignore_index=True
        ).sort_values("roc_auc", ascending=False)
    ensemble_clf_report = classification_report(y_test, y_pred_ensemble, output_dict=True)

    # Convert the classification report metrics into the ensemble_metrics dictionary
    ensemble_metrics.update({
        "accuracy": accuracy_score(y_test, y_pred_ensemble),
        "precision": ensemble_clf_report["weighted avg"]["precision"],
        "recall": ensemble_clf_report["weighted avg"]["recall"],
        "f1-score": ensemble_clf_report["weighted avg"]["f1-score"],
    })

    # Add ensemble results to dataframe
    results_df = pd.concat(
        [results_df, pd.DataFrame([ensemble_metrics])], ignore_index=True
    ).sort_values("roc_auc", ascending=False)

    # Print classification report for ensemble
    print(f"\n\033[1mEnsemble Model Results\033[0m")
    print(f"ROC-AUC: {ensemble_metrics['roc_auc']:.3f} | Accuracy: {ensemble_metrics['accuracy']:.3f}")
    print(f"Precision: {ensemble_metrics['precision']:.3f} | Recall: {ensemble_metrics['recall']:.3f} | F1-score: {ensemble_metrics['f1-score']:.3f}")
    print(classification_report(y_test, y_pred_ensemble))

    print("\n\033[1m" + "=" * 40 + " FINAL RESULTS " + "=" * 40 + "\033[0m")
    print(
        results_df[
            [
                "model",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1-score",
                "train_time",
                "predict_time"
            ]
        ]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )

    # Plot all AUC-ROC curves
    plt.figure(figsize=(8, 6))
    roc_curves_sorted = sorted(roc_curves, key=lambda x: x[3], reverse=True)
    for fpr, tpr, name, auc in roc_curves_sorted:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    # Compute and plot ensemble ROC
    if model_predictions:
        fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_proba_ensemble)
        plt.plot(
            fpr_ensemble,
            tpr_ensemble,
            label=f"Ensemble (AUC = {ensemble_metrics['roc_auc']:.3f})",
            linestyle="--",
            color="black",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve for All Models (With Ensemble)")
    plt.legend()
    plt.show()
    # # Plot all AUC-ROC curves
    # plt.figure(figsize=(8, 6))
    # roc_curves_sorted = sorted(roc_curves, key=lambda x: x[3], reverse=True)
    # for fpr, tpr, name, auc in roc_curves_sorted:
    #     plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    # plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("AUC-ROC Curve for All Models")
    # plt.legend()
    # plt.show()
    #return user_data


def get_best_features(
    feature_column,
    target_column,
    dataset,
    n_features=50,
    test_size=0.2,
    random_state=42,
):
    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
    dataset = dataset.dropna()

    def preprocess(dataset):
        # Replace infinite values
        dataset = dataset.replace([np.inf, -np.inf], np.nan)

        # Drop or fill NaN values
        dataset = dataset.fillna(
            dataset.median()
        )  # Or use mean, or a specific strategy

        return dataset
    
    # Define features and target
    X = dataset[feature_column]
    X = preprocess(X)
    y = dataset[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test, train_id, test_id = preprocess_features(feature_column, target_column, dataset)

    # Feature analysis
    feature_correlations = X.corrwith(y)
    best_features = []

    for model in [
        RandomForestClassifier(),
        BalancedRandomForestClassifier(),
        XGBClassifier(
            objective="binary:logistic",
        ),
    ]:
        try:
            m = model.fit(X_train, y_train)
            importance = (
                pd.DataFrame(
                    {
                        "Feature": feature_column,
                        "Importance": m.feature_importances_,
                        "Correlation": feature_correlations,
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(n_features)
            )
            highest_importances = importance["Feature"]
            highest_importances = importance["Feature"]
            best_features.append(highest_importances)

            print(f"\nTop Features ({model.__class__.__name__}):")
            print(importance.to_string(index=False))
        except AttributeError:
            continue

    return best_features


def optimize_xgb_params(X_train, X_val, y_train, y_val, max_evals=100):
    """
    Optimize XGBoost hyperparameters using Hyperopt with TPE algorithm.
    Uses separate validation set for parameter tuning to prevent overfitting.

    Parameters:
    -----------
    X_train : array-like
        Training features
    X_val : array-like
        Validation features for hyperparameter tuning
    y_train : array-like
        Training target
    y_val : array-like
        Validation target for hyperparameter tuning
    max_evals : int, default=100
        Number of optimization iterations

    Returns:
    --------
    dict : Best parameters found
    float : Best validation score achieved
    XGBClassifier : Best model
    """
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.04), np.log(0.12)),
        "max_depth": hp.choice("max_depth", range(5, 11)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 7, 1),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "n_estimators": hp.choice("n_estimators", range(50, 500, 50)),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(1.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(1.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(1.0)),
    }

    def objective(params):
        # Create and train XGBoost classifier with current parameters
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            **params,
        )

        # Train on training set
        xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        val_pred = xgb.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_pred)

        return {"loss": -val_score, "status": STATUS_OK}

    trials = Trials()
    print("Starting hyperparameter optimization...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1,
    )

    # Convert best parameters
    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_depth": best["max_depth"] + 3,
        "min_child_weight": int(best["min_child_weight"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "n_estimators": [100, 200, 300, 400, 500][best["n_estimators"]],
        "gamma": float(best["gamma"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
    }

    # Train final model with best parameters
    best_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        **best_params,
    )

    # Train with early stopping using validation set
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Get best validation score
    best_score = -min(trials.losses())

    # Print results
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest validation ROC-AUC score: {best_score:.4f}")

    # Plot optimization history
    scores = [-trial["result"]["loss"] for trial in trials.trials]
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title("Hyperopt Optimization History")
    plt.xlabel("Iteration")
    plt.ylabel("Validation ROC-AUC Score")
    plt.show()

    return best_params, best_score, best_model


def run_classification_plotly(feature_column, target_column, dataset, random_state=42):
    """
    Enhanced classification analysis with multiple models and comprehensive reporting using Plotly visualizations.
    """
    X_train, X_test, y_train, y_test, train_id, test_id = preprocess_features(
        feature_column, target_column, dataset
    )

    models = [
        (
            LogisticRegression(
                class_weight="balanced", max_iter=200, random_state=random_state
            ),
            "Logistic Regression",
        ),
        (RandomForestClassifier(random_state=random_state), "Random Forest"),
        (
            lgb.LGBMClassifier(
                objective="binary",
                verbose=-1,
                force_row_wise=True,
                random_state=random_state,
            ),
            "LightGBM",
        ),
        (BalancedRandomForestClassifier(random_state=random_state), "Balanced RF"),
        (
            XGBClassifier(
                learning_rate=0.08,
                max_depth=7,
                n_estimators=100,
                objective="binary:logistic",
                random_state=random_state,
            ),
            "XGBoost",
        ),
        (CatBoostClassifier(silent=True, random_state=random_state), "CatBoost"),
        (HistGradientBoostingClassifier(random_state=random_state), "HistGB"),
        (RUSBoostClassifier(random_state=random_state), "RUSBoost"),
    ]

    model_results = []
    roc_curves = []

    def plot_classification_report(clf_report, model_name):
        """Create heatmap visualization of classification report"""
        # Extract class-specific scores (excluding 'accuracy' and averages)
        classes = ['0.0', '1.0']  # For binary classification
        metrics = ["precision", "recall", "f1-score"]

        z = []
        for cls in classes:
            row = []
            for metric in metrics:
                row.append(clf_report[cls][metric])
            z.append(row)

        # Add macro avg and weighted avg
        for avg_type in ["macro avg", "weighted avg"]:
            row = []
            for metric in metrics:
                row.append(clf_report[avg_type][metric])
            z.append(row)

        # Update labels to include averages
        classes = classes + ["macro avg", "weighted avg"]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=metrics,
                y=classes,
                colorscale="Blues",
                text=np.round(z, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=True,
            )
        )

        fig.update_layout(
            title=f"Classification Report - {model_name}",
            xaxis_title="Metrics",
            yaxis_title="Classes",
            height=400,
        )
        fig.show()

    def plot_confusion_matrix(cm, model_name):
        """Create interactive confusion matrix visualization"""
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=True,
            )
        )

        fig.update_layout(
            title=f"Confusion Matrix - {model_name}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=400,
        )
        fig.show()

    def evaluate_model(model, name):
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time_end = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name,
                "roc_auc": roc_auc_score(y_test, y_proba),
                "accuracy": accuracy_score(y_test, y_pred),
                "train_time": train_time_end - start_time,
                "predict_time": (time.time() - train_time_end) / X_test.shape[0],
            }

            clf_report = classification_report(y_test, y_pred, output_dict=True)
            for k, v in clf_report["weighted avg"].items():
                metrics[k] = v
            model_results.append(metrics)

            print(f"\n\033[1m{name} Results\033[0m")
            print(
                f"ROC-AUC: {metrics['roc_auc']:.3f} | Accuracy: {metrics['accuracy']:.3f}"
            )
            print(
                f"Training Time: {metrics['train_time']:.1f}s | Predicting Time: {metrics['predict_time']:.6f}s"
            )

            # Plot classification report
            plot_classification_report(clf_report, name)

            # Get ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves.append((fpr, tpr, name, metrics["roc_auc"]))

            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, name)

        except Exception as e:
            print(f"\n\033[91mError in {name}: {str(e)}\033[0m")

    # Execute all models
    for model, name in models:
        user_data = evaluate_model(model, name)

    # Create final results visualization
    results_df = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)

    # Plot final results as a parallel coordinates plot
    fig = px.parallel_coordinates(
        results_df,
        dimensions=["roc_auc", "accuracy", "precision", "recall", "f1-score"],
        color="model",
        title="Model Performance Comparison",
    )
    fig.update_layout(height=600)
    fig.show()

    # Create bar plot for training and prediction times
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Training Time", "Prediction Time (per sample)")
    )

    fig.add_trace(
        go.Bar(x=results_df["model"], y=results_df["train_time"], name="Training Time"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=results_df["model"], y=results_df["predict_time"], name="Prediction Time"
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500, title_text="Model Timing Comparison", showlegend=False
    )
    fig.show()

    # Plot all ROC curves
    fig = go.Figure()
    for fpr, tpr, name, auc in roc_curves:
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC = {auc:.3f})", mode="lines")
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random",
            mode="lines",
            line=dict(dash="dash", color="gray"),
        )
    )

    fig.update_layout(
        title="ROC Curves for All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=600,
        showlegend=True,
    )
    fig.show()

    # Print final results table
    print("\n\033[1m" + "=" * 40 + " FINAL RESULTS " + "=" * 40 + "\033[0m")
    print(
        results_df[
            [
                "model",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1-score",
                "train_time",
                "predict_time",
            ]
        ]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )

class WeightedEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)

    def predict_proba(self, X):
        ensemble_proba = np.zeros((X.shape[0]))
        for name, weight in self.weights.items():
            model = self.models[name]
            proba = model.predict_proba(X)[:, 1]
            ensemble_proba += weight * proba
        return np.vstack((1 - ensemble_proba, ensemble_proba)).T
    
def run_classification2(feature_column, target_column, dataset, random_state=42):
    """
    Enhanced classification analysis with multiple models and comprehensive reporting.
    Includes ensemble model saving and optimizations for maximizing AUC-ROC.
    """
    X_train, X_test, y_train, y_test, train_id, test_id = preprocess_features(
        feature_column, target_column, dataset
    )

    # Hyperparameter optimization settings for key models
    models = [
        # Core Models
        (
            LogisticRegression(
                class_weight="balanced", max_iter=1000, C=0.1, solver="liblinear"
            ),
            "Logistic Regression",
        ),
        (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
            ),
            "Random Forest",
        ),
        (
            lgb.LGBMClassifier(
                objective="binary",
                verbose=-1,
                force_row_wise=True,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
            "LightGBM",
        ),
        (
            BalancedRandomForestClassifier(n_estimators=200, random_state=random_state),
            "Balanced RF",
        ),
        # Gradient Boosting Family
        (
            XGBClassifier(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                scale_pos_weight=sum(y_train == 0)
                / sum(y_train == 1),  # Handle class imbalance
            ),
            "XGBoost",
        ),
        (
            CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3, silent=True
            ),
            "CatBoost",
        ),
        (
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=6
            ),
            "HistGB",
        ),
        (
            RUSBoostClassifier(
                n_estimators=200, learning_rate=0.05, random_state=random_state
            ),
            "RUSBoost",
        ),
    ]

    # Model evaluation framework
    model_results = []
    roc_curves = []
    model_predictions = {}
    fitted_models = {}  # Store fitted models for ensemble creation

    def evaluate_model(model, name):
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            fitted_models[name] = model  # Store fitted model
            dump(model, filename=f"{name}.joblib")
            train_time_end = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            model_predictions[name] = y_proba

            # Calculate metrics
            metrics = {
                "model": name,
                "roc_auc": roc_auc_score(y_test, y_proba),
                "accuracy": accuracy_score(y_test, y_pred),
                "train_time": train_time_end - start_time,
                "predict_time": (time.time() - train_time_end) / X_test.shape[0],
            }

            # Add classification report metrics
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            for k, v in clf_report["weighted avg"].items():
                metrics[k] = v
            model_results.append(metrics)

            # Print model report
            print(f"\n\033[1m{name} Results\033[0m")
            print(
                f"ROC-AUC: {metrics['roc_auc']:.3f} | Accuracy: {metrics['accuracy']:.3f}"
            )
            print(
                f"Training Time: {metrics['train_time']:.1f}s | Predicting Time: {metrics['predict_time']:.6f}s"
            )
            print(classification_report(y_test, y_pred))

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves.append((fpr, tpr, name, metrics["roc_auc"]))

            # Plot Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{name} - Confusion Matrix")
            plt.show()

        except Exception as e:
            print(f"\n\033[91mError in {name}: {str(e)}\033[0m")

    # Execute all models
    for model, name in models:
        # evaluate_model(model, name)
        if name in ["LightGBM", "XGBoost", "CatBoost"]:
            evaluate_model(model, name)

    # Results analysis
    results_df = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)

    # Create and save the ensemble model
    if model_predictions:
        # Create a more sophisticated ensemble - weighted average based on individual model performance
        weights = {
            name: score
            for name, score in zip(
                [model[1] for model in models if model[1] in model_predictions.keys()],
                [
                    results_df[results_df["model"] == name]["roc_auc"].values[0]
                    for name in model_predictions.keys()
                ],
            )
        }

        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate weighted ensemble predictions
        y_proba_ensemble = np.zeros_like(list(model_predictions.values())[0])
        for name, weight in normalized_weights.items():
            y_proba_ensemble += weight * model_predictions[name]

        y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)

        # Evaluate ensemble performance
        ensemble_metrics = {
            "model": "Weighted Ensemble",
            "roc_auc": roc_auc_score(y_test, y_proba_ensemble),
            "accuracy": accuracy_score(y_test, y_pred_ensemble),
        }

        # Create the ensemble model
        ensemble_model = WeightedEnsembleModel(fitted_models, normalized_weights)

        # Save the ensemble model
        dump(ensemble_model, filename="WeightedEnsemble.joblib")
        print("\n\033[1mEnsemble model saved as 'WeightedEnsemble.joblib'\033[0m")

        # Add ensemble results to dataframe
        results_df = pd.DataFrame(model_results)
        ensemble_clf_report = classification_report(
            y_test, y_pred_ensemble, output_dict=True
        )

        # Convert the classification report metrics into the ensemble_metrics dictionary
        ensemble_metrics.update(
            {
                "accuracy": accuracy_score(y_test, y_pred_ensemble),
                "precision": ensemble_clf_report["weighted avg"]["precision"],
                "recall": ensemble_clf_report["weighted avg"]["recall"],
                "f1-score": ensemble_clf_report["weighted avg"]["f1-score"],
            }
        )

        # Add ensemble results to dataframe
        results_df = pd.concat(
            [results_df, pd.DataFrame([ensemble_metrics])], ignore_index=True
        ).sort_values("roc_auc", ascending=False)

        # Print classification report for ensemble
        print(f"\n\033[1mEnsemble Model Results\033[0m")
        print(
            f"ROC-AUC: {ensemble_metrics['roc_auc']:.3f} | Accuracy: {ensemble_metrics['accuracy']:.3f}"
        )
        print(
            f"Precision: {ensemble_metrics['precision']:.3f} | Recall: {ensemble_metrics['recall']:.3f} | F1-score: {ensemble_metrics['f1-score']:.3f}"
        )
        print(classification_report(y_test, y_pred_ensemble))

    print("\n\033[1m" + "=" * 40 + " FINAL RESULTS " + "=" * 40 + "\033[0m")
    print(
        results_df[
            [
                "model",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1-score",
                "train_time",
                "predict_time",
            ]
        ]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )

    # Plot all AUC-ROC curves
    plt.figure(figsize=(8, 6))
    roc_curves_sorted = sorted(roc_curves, key=lambda x: x[3], reverse=True)
    for fpr, tpr, name, auc in roc_curves_sorted:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    # Compute and plot ensemble ROC
    if model_predictions:
        fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_proba_ensemble)
        plt.plot(
            fpr_ensemble,
            tpr_ensemble,
            label=f"Weighted Ensemble (AUC = {ensemble_metrics['roc_auc']:.3f})",
            linestyle="--",
            color="black",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve for All Models (With Ensemble)")
    plt.legend()
    plt.show()

    return (
        results_df,
        fitted_models,
        ensemble_model if "ensemble_model" in locals() else None,
    )
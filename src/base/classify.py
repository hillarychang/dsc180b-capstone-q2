import pandas as pd
import numpy as np
import warnings
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def preprocess_features(feature_column, target_column, dataset, test_size = 0.2, random_state = 42):
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

    # Define features and targett
    X = dataset[feature_column]
    X = preprocess(X)
    y = dataset[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Balance and scale data
    resampler = SMOTETomek(random_state=random_state)
    X_train, y_train = resampler.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# run_classification models
def run_classification(
    X_train, X_test, y_train, y_test, random_state=42
):
    """
    Enhanced classification analysis with multiple models and comprehensive reporting.
    """

    # Configure models in standardized format
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
        ("lgbm", lgb.LGBMClassifier()),
    ]

    models = [
        # Core Models
        (
            LogisticRegression(class_weight="balanced", max_iter=200),
            "Logistic Regression",
        ),
        (RandomForestClassifier(random_state=random_state), "Random Forest"),
        (lgb.LGBMClassifier(objective="binary", force_row_wise=True), "LightGBM"),
        (BalancedRandomForestClassifier(random_state=random_state), "Balanced RF"),
        # Gradient Boosting Family
        (
            XGBClassifier(
                learning_rate=0.08,
                max_depth=7,
                n_estimators=100,
                objective="binary:logistic",
            ),
            "XGBoost",
        ),
        (CatBoostClassifier(silent=True), "CatBoost"),
        (HistGradientBoostingClassifier(), "HistGB"),
        (RUSBoostClassifier(random_state=random_state), "RUSBoost"),
    ]

    # Model evaluation framework
    model_results = []

    def evaluate_model(model, name):
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "model": name,
                "roc_auc": roc_auc_score(y_test, y_proba),
                "accuracy": accuracy_score(y_test, y_pred),
                "train_time": time.time() - start_time,
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
            print(f"Training Time: {metrics['train_time']:.1f}s")
            print(classification_report(y_test, y_pred))

            # Plot AUC-ROC score
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.3f})")

        except Exception as e:
            print(f"\n\033[91mError in {name}: {str(e)}\033[0m")

    # Execute all models
    for model, name in models:
        evaluate_model(model, name)

    # Results analysis
    results_df = pd.DataFrame(model_results).sort_values("roc_auc", ascending=False)

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
            ]
        ]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve for All Models")
    plt.legend()
    plt.show()


def get_best_features(
    X_train,
    X_test,
    y_train,
    y_test,
    n_features=50,
    test_size=0.2,
    random_state=42,
):
    # Feature analysis
    feature_correlations = X_train.corrwith(y_train)
    best_features = []

    for model in [
        RandomForestClassifier(),
        BalancedRandomForestClassifier(),
        XGBClassifier(),
    ]:
        try:
            m = model.fit(X_train, y_train)
            importance = (
                pd.DataFrame(
                    {
                        "Feature": X_train.columns,
                        "Importance": m.feature_importances_,
                        "Correlation": feature_correlations,
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(n_features)
            )
            highest_importances = importance["Feature"]
            best_features.append(highest_importances)

            print(f"\nTop Features ({model.__class__.__name__}):")
            print(importance.to_string(index=False))
        except AttributeError:
            continue
    

    return best_features


def optimize_xgb_params(X_train, y_train, max_evals=100):
    """
    Optimize XGBoost hyperparameters using Hyperopt with TPE algorithm.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    max_evals : int, default=100
        Number of optimization iterations

    Returns:
    --------
    dict : Best parameters found
    float : Best score achieved
    XGBClassifier : Best model
    """

    # Define the search space
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.3)),
        "max_depth": hp.choice("max_depth", range(3, 11)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 7, 1),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400, 500]),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(1.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(1.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(1.0)),
    }

    def objective(params):
        # Create XGBoost classifier with current parameters
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            **params,
        )

        # Perform cross-validation
        score = cross_val_score(
            xgb, X_train, y_train, scoring="roc_auc", cv=5, n_jobs=-1
        ).mean()

        # Return negative score for minimization
        return {"loss": -score, "status": STATUS_OK}

    # Store training history
    trials = Trials()

    # Run optimization
    print("Starting hyperparameter optimization...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1,
    )

    # Convert best parameters to proper format
    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_depth": best["max_depth"] + 3,  # Adjust for hp.choice indexing
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
    best_model.fit(X_train, y_train)

    # Get best score
    best_score = -min(trials.losses())

    # Print results
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"\nBest ROC-AUC score: {best_score:.4f}")

    # Plot optimization history
    scores = [-trial["result"]["loss"] for trial in trials.trials]
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title("Hyperopt Optimization History")
    plt.xlabel("Iteration")
    plt.ylabel("ROC-AUC Score")
    plt.show()

    return best_params, best_score, best_model
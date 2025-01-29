import pandas as pd
import numpy as np
import warnings
import time
import pandas as pd
import numpy as np

# Scikit-Learn Components
from sklearn.model_selection import train_test_split, GridSearchCV
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


def run_classification(
    feature_column, target_column, dataset, test_size=0.2, random_state=42
):
    """
    Enhanced classification analysis with multiple models and comprehensive reporting.
    """
    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
    dataset = dataset.dropna()

    # Define features and target
    X = dataset[feature_column]
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
        (lgb.LGBMClassifier(), "LightGBM"),
        (BalancedRandomForestClassifier(random_state=random_state), "Balanced RF"),
        # Gradient Boosting Family
        (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), "XGBoost"),
        (CatBoostClassifier(silent=True), "CatBoost"),
        (HistGradientBoostingClassifier(), "HistGB"),
        # SVM Variants
        (
            SVC(class_weight="balanced", probability=True, random_state=random_state),
            "SVM-RBF",
        ),
        (
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
            "SVM-Linear",
        ),
        (
            GridSearchCV(
                SVC(
                    class_weight="balanced", probability=True, random_state=random_state
                ),
                param_grid={"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]},
                cv=3,
                scoring="roc_auc",
            ),
            "Tuned SVM",
        ),
        # Neural Networks
        (
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                random_state=random_state,
            ),
            "MLP",
        ),
        # Advanced Ensembles
        (
            StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(),
                stack_method="predict_proba",
            ),
            "Stacking Ensemble",
        ),
        # Imbalance Specialists
        (
            EasyEnsembleClassifier(n_estimators=10, random_state=random_state),
            "EasyEnsemble",
        ),
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

    # Feature analysis
    feature_correlations = X.corrwith(y)

    print("\n\033[1m" + "=" * 40 + " FEATURE ANALYSIS " + "=" * 40 + "\033[0m")
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
                        "Feature": feature_column,
                        "Importance": m.feature_importances_,
                        "Correlation": feature_correlations,
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(10)
            )

            print(f"\nTop Features ({model.__class__.__name__}):")
            print(importance.to_string(index=False))

        except AttributeError:
            continue

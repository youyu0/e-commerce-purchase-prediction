"""
@Function                       @Description

ensure_output_dir               Make sure the directory exist
print_split_stats               Show positive/negative distribution of the dataset
compute_scale_pos_weight        Compute scale_pos_weight for XGBoost
predict_scores                  Get score
evaluate_predictions            Calculate every metric scores
find_best_threshold             Find best threshold
add_behavior_features           Do feature engineering
build_training_dataset          Generate final training dataset
group_train_val_test_split      Split dataset by user_session
make_preprocessor               Create a preprocessor for XGBoost pipeline
build_xgb_pipeline              XGBoost pipeline
tune_xgboost                    Auto Fine-tuned XGBoost model hyperparameter
main                            Main function

"""
import time
import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


INPUT_PATH = "2019-Oct.csv"
OUTPUT_DIR = "./output"
OUTPUT_TRAIN_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5
N_ITER_SEARCH = 16

PRIMARY_METRIC = "average_precision"   # PR-AUC
THRESHOLD_OBJECTIVE = "f1"             # "f1" or "recall"
MIN_PRECISION_CONSTRAINT = None        # if want to consider Precision while choosing threshold
MAX_ROWS = None                        # set integer for debugging if needed


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_split_stats(name, y):
    y = pd.Series(y)
    pos = int(y.sum())
    total = len(y)
    neg = total - pos
    rate = pos / total if total > 0 else 0.0
    print(f"{name:<10} -> n={total:,}, pos={pos:,}, neg={neg:,}, pos_rate={rate:.4f}")


def compute_scale_pos_weight(y):
    y = pd.Series(y)
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    if pos == 0:
        return 1.0
    return neg / pos


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Model does not support predict_proba.")


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, objective="f1", min_precision=None):
    grid = np.linspace(0.01, 0.99, 99)

    records = []
    best_threshold = 0.50
    best_score = -np.inf

    for thr in grid:
        metrics_dict = evaluate_predictions(y_true, y_prob, threshold=thr)

        if min_precision is not None and metrics_dict["precision"] < min_precision:
            score = -np.inf
        else:
            if objective == "recall":
                score = metrics_dict["recall"]
            else:
                score = metrics_dict["f1"]

        metrics_dict["objective_score"] = float(score)
        records.append(metrics_dict)

        if score > best_score:
            best_score = score
            best_threshold = thr

    threshold_df = pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)
    return float(best_threshold), threshold_df


def add_behavior_features(df):
    df = df.copy()

    df["user_session"] = df["user_session"].fillna("__missing_session__").astype(str)
    df["event_type"] = df["event_type"].fillna("unknown").astype(str)
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").fillna(-1).astype("int64")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

    df = df.dropna(subset=["event_time"]).copy()

    df["is_view"] = (df["event_type"] == "view").astype("int32")
    df["is_cart"] = (df["event_type"] == "cart").astype("int32")
    df["is_remove"] = (df["event_type"] == "remove_from_cart").astype("int32")
    df["is_purchase"] = (df["event_type"] == "purchase").astype("int32")

    df = df.sort_values(["user_session", "event_time"]).reset_index(drop=True)

    df["prior_session_view_count"] = (
        df.groupby("user_session")["is_view"].cumsum() - df["is_view"]
    ).fillna(0).astype("int32")

    df["prior_session_cart_count"] = (
        df.groupby("user_session")["is_cart"].cumsum() - df["is_cart"]
    ).fillna(0).astype("int32")

    df["prior_session_remove_count"] = (
        df.groupby("user_session")["is_remove"].cumsum() - df["is_remove"]
    ).fillna(0).astype("int32")

    df["prior_same_product_view_count"] = (
        df.groupby(["user_session", "product_id"])["is_view"].cumsum() - df["is_view"]
    ).fillna(0).astype("int32")

    df["prior_same_product_cart_count"] = (
        df.groupby(["user_session", "product_id"])["is_cart"].cumsum() - df["is_cart"]
    ).fillna(0).astype("int32")

    df["prior_same_product_remove_count"] = (
        df.groupby(["user_session", "product_id"])["is_remove"].cumsum() - df["is_remove"]
    ).fillna(0).astype("int32")

    df["prior_session_event_count"] = (
        df.groupby("user_session").cumcount()
    ).fillna(0).astype("int32")

    session_start = df.groupby("user_session")["event_time"].transform("min")
    df["seconds_from_session_start"] = (
        (df["event_time"] - session_start).dt.total_seconds()
    ).fillna(0).astype("float32")

    return df


def build_training_dataset(df):
    df = add_behavior_features(df)

    df_targets = df.loc[df["event_type"].isin(["cart", "purchase"])].copy()

    df_targets = df_targets.drop_duplicates(
        subset=["event_type", "product_id", "price", "user_id", "user_session"]
    )

    df_targets["is_purchased"] = (df_targets["event_type"] == "purchase").astype("int32")

    df_targets["is_purchased"] = (
        df_targets.groupby(["user_session", "product_id"])["is_purchased"]
        .transform("max")
        .astype("int32")
    )

    df_targets = df_targets.loc[df_targets["event_type"] == "cart"].copy()
    df_targets = df_targets.drop_duplicates(["user_session", "product_id", "is_purchased"])

    df_targets["event_weekday"] = df_targets["event_time"].dt.weekday.astype("int32")
    df_targets["brand"] = df_targets["brand"].fillna("unknown").astype(str)
    df_targets["category_code"] = df_targets["category_code"].fillna("unknown.unknown").astype(str)
    df_targets["price"] = pd.to_numeric(df_targets["price"], errors="coerce").fillna(0).astype("float32")

    category_split = df_targets["category_code"].str.split(".", n=2, expand=True)
    df_targets["category_code_level1"] = category_split[0].fillna("unknown").astype(str)
    df_targets["category_code_level2"] = category_split[1].fillna("unknown").astype(str)

    keep_cols = [
        "event_time",
        "event_type",
        "product_id",
        "price",
        "user_id",
        "user_session",
        "brand",
        "category_code",
        "category_code_level1",
        "category_code_level2",
        "event_weekday",
        "is_purchased",
        "prior_session_view_count",
        "prior_session_cart_count",
        "prior_session_remove_count",
        "prior_session_event_count",
        "prior_same_product_view_count",
        "prior_same_product_cart_count",
        "prior_same_product_remove_count",
        "seconds_from_session_start",
    ]

    return df_targets[keep_cols].copy()


def group_train_val_test_split(X, y, groups, test_size=0.15, val_size=0.15, random_state=42):
    gss_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_val_idx, test_idx = next(gss_1.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx].copy()
    y_train_val = y.iloc[train_val_idx].copy()
    groups_train_val = groups.iloc[train_val_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    relative_val_size = val_size / (1.0 - test_size)

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state + 1
    )
    train_idx_rel, val_idx_rel = next(gss_2.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx_rel].copy()
    y_train = y_train_val.iloc[train_idx_rel].copy()
    groups_train = groups_train_val.iloc[train_idx_rel].copy()

    X_val = X_train_val.iloc[val_idx_rel].copy()
    y_val = y_train_val.iloc[val_idx_rel].copy()
    groups_val = groups_train_val.iloc[val_idx_rel].copy()

    assert set(groups_train).isdisjoint(set(groups_val))
    assert set(groups_train).isdisjoint(set(groups_test))
    assert set(groups_val).isdisjoint(set(groups_test))

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test


def make_preprocessor(cat_cols, num_cols):
    # For features with text type, first fill missing value & transform to numeric
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    # For features with numeric type, fill missing value with median
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    prep = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        verbose_feature_names_out=False
    )
    return prep


def build_xgb_pipeline(cat_cols, num_cols, scale_pos_weight):
    prep = make_preprocessor(cat_cols, num_cols)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
    )

    return Pipeline([
        ("prep", prep),
        ("model", model),
    ])


def tune_xgboost(X_train, y_train, groups_train, cat_cols, num_cols):
    base_spw = compute_scale_pos_weight(y_train)
    print(f"\nscale_pos_weight from training set = {base_spw:.4f}")
    xgb_pipe = build_xgb_pipeline(cat_cols, num_cols, base_spw)

    param_dist = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__gamma": [0.0, 0.1, 0.3, 0.5],
        "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "model__reg_lambda": [1.0, 3.0, 5.0, 10.0],
        "model__scale_pos_weight": [base_spw * 0.75, base_spw, base_spw * 1.25],
    }

    cv = GroupKFold(n_splits=CV_FOLDS)

    search = RandomizedSearchCV(
        estimator=xgb_pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=PRIMARY_METRIC,
        cv=cv,
        refit=True,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train, groups=groups_train)

    best_pipe = search.best_estimator_
    best_pipe.named_steps["model"].set_params(n_jobs=-1)

    return search, best_pipe


def main():
    ensure_output_dir()

    df = pd.read_csv(INPUT_PATH, nrows=MAX_ROWS)

    df_targets = build_training_dataset(df)
    df_targets.to_csv(OUTPUT_TRAIN_PATH, index=False)

    print("\nFull class distribution:")
    print_split_stats("full", df_targets["is_purchased"])

    feature_cols = [
        "brand",
        "price",
        "event_weekday",
        "category_code_level1",
        "category_code_level2",
        "prior_session_view_count",
        "prior_session_cart_count",
        "prior_session_remove_count",
        "prior_session_event_count",
        "prior_same_product_view_count",
        "prior_same_product_cart_count",
        "prior_same_product_remove_count",
        "seconds_from_session_start",
    ]

    # identify features with text type or numeric; this is useful for XGBoost pipeline
    cat_cols = [
        "brand",
        "category_code_level1",
        "category_code_level2",
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df_targets[feature_cols].copy()
    y = df_targets["is_purchased"].astype(int).copy()
    groups = df_targets["user_session"].astype(str).copy()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        groups_train, groups_val, groups_test
    ) = group_train_val_test_split(
        X=X,
        y=y,
        groups=groups,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nSplit distribution:")
    print_split_stats("train", y_train)
    print_split_stats("val", y_val)
    print_split_stats("test", y_test)

    print("\nTuning XGBoost with GroupKFold...")
    xgb_search, best_xgb_pipe = tune_xgboost(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        cat_cols=cat_cols,
        num_cols=num_cols
    )

    print("\nBest XGBoost CV PR-AUC:", xgb_search.best_score_)
    print("Best XGBoost params:")
    print(json.dumps(xgb_search.best_params_, indent=2))

    val_prob = predict_scores(best_xgb_pipe, X_val)
    best_threshold, threshold_df = find_best_threshold(
        y_true=y_val,
        y_prob=val_prob,
        objective=THRESHOLD_OBJECTIVE,
        min_precision=MIN_PRECISION_CONSTRAINT
    )

    # threshold_path = os.path.join(OUTPUT_DIR, "threshold_sweep_validation.csv")
    # threshold_df.to_csv(threshold_path, index=False)
    print(f"Chosen threshold ({THRESHOLD_OBJECTIVE}) = {best_threshold:.2f}")

    # re-train with test + validation set
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    best_xgb_pipe.fit(X_dev, y_dev)

    print("\nFinal Test Performance:")
    test_prob = predict_scores(best_xgb_pipe, X_test)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=best_threshold)
    print(json.dumps(test_metrics, indent=2))
    test_predictions = X_test.copy()
    test_predictions["user_session"] = groups_test.values
    test_predictions["product_id"] = df_targets.loc[X_test.index, "product_id"].values
    test_predictions["y_true"] = y_test.values
    test_predictions["y_prob"] = test_prob
    test_predictions["y_pred"] = (test_prob >= best_threshold).astype(int)
    pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    test_predictions.to_csv(pred_path, index=False)
    print(f"\nSaved test predictions to: {pred_path}")


if __name__ == "__main__":
    
    """Overall Process
    
    1. Feature Engineering
    2. Build the training dataset
    3. Split train/validate/test dataset
    4. Compute scale_pos_weight dealing with class imbalance
    5. Auto fine-tune XGBoost model
        - Compute scale_pos_weight dealing with class imbalance
        - Build XGBoost pipeline
        - Use RandomizedSearchCV to do auto fine-tune (based on PR-AUC)
    6. Find best threshold
        - Start from threshold=0.5
        - Currently, we only find the threshold with best F1 score
    7. Train again with train + validation set
    
    """
    print("Start...")
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"\nExecute time: {end - start:.6f} s")
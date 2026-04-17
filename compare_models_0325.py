import time
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


from predict_v2_0324 import (
    INPUT_PATH, OUTPUT_DIR, MAX_ROWS, TEST_SIZE, VAL_SIZE, RANDOM_STATE, CV_FOLDS, 
    PRIMARY_METRIC, THRESHOLD_OBJECTIVE, MIN_PRECISION_CONSTRAINT,
    ensure_output_dir, build_training_dataset, print_split_stats,
    group_train_val_test_split, make_preprocessor, compute_scale_pos_weight,
    predict_scores, evaluate_predictions, find_best_threshold
)


N_ITER_SEARCH_COMPARE = 16

def build_lgbm_pipeline(cat_cols, num_cols, scale_pos_weight):
    prep = make_preprocessor(cat_cols, num_cols)
    model = LGBMClassifier(
        objective="binary",
        class_weight={0: 1.0, 1: scale_pos_weight},
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1 
    )
    return Pipeline([("prep", prep), ("model", model)])

def build_rf_pipeline(cat_cols, num_cols):
    prep = make_preprocessor(cat_cols, num_cols)
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", prep), ("model", model)])

def tune_model(pipeline, param_dist, X_train, y_train, groups_train, model_name="Model"):
    print(f"\n[{model_name}] K-Fold Tuning...")
    cv = GroupKFold(n_splits=CV_FOLDS)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH_COMPARE, 
        scoring=PRIMARY_METRIC,
        cv=cv,
        refit=True,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search.fit(X_train, y_train, groups=groups_train)
    print(f"[{model_name}] BEST CV PR-AUC: {search.best_score_:.4f}")
    return search.best_estimator_

def main():
    ensure_output_dir()
    df = pd.read_csv(INPUT_PATH, nrows=MAX_ROWS)
    df_targets = build_training_dataset(df)
    
    feature_cols = [
        "brand", "price", "event_weekday", "category_code_level1", "category_code_level2",
        "prior_session_view_count", "prior_session_cart_count", "prior_session_remove_count",
        "prior_session_event_count", "prior_same_product_view_count", "prior_same_product_cart_count",
        "prior_same_product_remove_count", "seconds_from_session_start"
    ]
    cat_cols = ["brand", "category_code_level1", "category_code_level2"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df_targets[feature_cols].copy()
    y = df_targets["is_purchased"].astype(int).copy()
    groups = df_targets["user_session"].astype(str).copy()

    (X_train, X_val, X_test, y_train, y_val, y_test,
     groups_train, groups_val, groups_test) = group_train_val_test_split(
        X, y, groups, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    base_spw = compute_scale_pos_weight(y_train)
    
    # === 1. XGBoost ===
    xgb_pipe = Pipeline([
        ("prep", make_preprocessor(cat_cols, num_cols)),
        ("model", XGBClassifier(objective="binary:logistic", tree_method="hist", scale_pos_weight=base_spw, random_state=RANDOM_STATE, n_jobs=1))
    ])
    xgb_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8],
    }
    
    # === 2. LightGBM ===
    lgbm_pipe = build_lgbm_pipeline(cat_cols, num_cols, base_spw)
    lgbm_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8, -1],
        "model__num_leaves": [15, 31, 63],
    }
    
    # === 3. Random Forest ===
    rf_pipe = build_rf_pipeline(cat_cols, num_cols)
    rf_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [6, 10, 15, None],
        "model__min_samples_split": [2, 5, 10]
    }

    models_to_test = {
        "Random Forest": (rf_pipe, rf_params),
        "LightGBM": (lgbm_pipe, lgbm_params),
        "XGBoost": (xgb_pipe, xgb_params),
    }

    results = []

    
    for name, (pipe, params) in models_to_test.items():
        best_pipe = tune_model(pipe, params, X_train, y_train, groups_train, model_name=name)
        
        
        val_prob = predict_scores(best_pipe, X_val)
        best_threshold, _ = find_best_threshold(y_val, val_prob, objective=THRESHOLD_OBJECTIVE, min_precision=MIN_PRECISION_CONSTRAINT)
        
       
        test_prob = predict_scores(best_pipe, X_test)
        test_metrics = evaluate_predictions(y_test, test_prob, threshold=best_threshold)
        
        results.append({
            "Model": name,
            "Best Threshold": best_threshold,
            "Test PR-AUC": test_metrics["pr_auc"],
            "Test ROC-AUC": test_metrics["roc_auc"],
            "Test F1": test_metrics["f1"],
            "Test Precision": test_metrics["precision"],
            "Test Recall": test_metrics["recall"]
        })

    # 產出比較圖表
    df_results = pd.DataFrame(results).sort_values("Test PR-AUC", ascending=False)
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(df_results.to_markdown(index=False))
    
    out_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nRESULT SAVE TO: {out_path}")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f": {time.perf_counter() - start:.2f} 秒")

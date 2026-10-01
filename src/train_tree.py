from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
TEST_SIZE = 0.20
TARGET = "SalePrice"
# These are deliberately small and interpretable for the visual demo.
FEATURES = ["OverallQual", "GrLivArea", "YearBuilt", "GarageCars", "TotalBsmtSF", "FullBath"]
LABELS = {
    "OverallQual": "OverallQual (chất lượng)",
    "GrLivArea": "GrLivArea (diện tích ở)",
    "YearBuilt": "YearBuilt (năm xây dựng)",
    "GarageCars": "GarageCars (chỗ gara)",
    "TotalBsmtSF": "TotalBsmtSF (diện tích tầng hầm)",
    "FullBath": "FullBath (phòng tắm đầy đủ)",
}


def tree_to_json(model: DecisionTreeRegressor, feature_names: list[str]) -> dict:
    t = model.tree_
    def visit(node: int) -> dict:
        is_leaf = t.children_left[node] == t.children_right[node]
        payload = {
            "id": int(node), "samples": int(t.n_node_samples[node]),
            "value": float(t.value[node][0][0]), "impurity": float(t.impurity[node]),
            "leaf": bool(is_leaf),
        }
        if not is_leaf:
            payload.update({
                "feature": feature_names[t.feature[node]],
                "threshold": float(t.threshold[node]),
                "left": visit(t.children_left[node]),
                "right": visit(t.children_right[node]),
            })
        return payload
    return visit(0)


def main() -> None:
    df = pd.read_csv(DATA)
    missing_before = int(df[FEATURES + [TARGET]].isna().sum().sum())
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    medians = X.median()
    X = X.fillna(medians)
    missing_after = int(X.isna().sum().sum())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=8, random_state=SEED)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        "model": "Decision Tree Regression",
        "target": TARGET,
        "features": FEATURES,
        "feature_labels": LABELS,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "source_rows": int(len(df)), "source_columns": int(len(df.columns)),
        "missing_before_selected": missing_before, "missing_after_selected": missing_after,
        "max_depth": int(model.get_depth()), "node_count": int(model.tree_.node_count),
        "leaf_count": int(model.get_n_leaves()), "train_seconds": round(train_seconds, 4),
        "train": {"MAE": round(float(mean_absolute_error(y_train, pred_train)), 2), "RMSE": round(float(np.sqrt(mean_squared_error(y_train, pred_train))), 2), "R2": round(float(r2_score(y_train, pred_train)), 4)},
        "test": {"MAE": round(float(mean_absolute_error(y_test, pred_test)), 2), "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 2), "R2": round(float(r2_score(y_test, pred_test)), 4)},
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (ART / "tree.json").write_text(json.dumps(tree_to_json(model, FEATURES), ensure_ascii=False, indent=2), encoding="utf-8")
    (ART / "defaults.json").write_text(json.dumps({k: float(v) for k, v in medians.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    pred = pd.DataFrame({"actual": y_test.values, "predicted": pred_test})
    pred["error"] = pred["predicted"] - pred["actual"]
    pred["absolute_error"] = pred["error"].abs()
    pred.head(120).to_csv(ART / "test_predictions.csv", index=False)
    # Full tree diagram: the actual fitted tree, not a mock illustration.
    plt.figure(figsize=(24, 12), dpi=150)
    plot_tree(model, feature_names=[LABELS[x] for x in FEATURES], filled=True, rounded=True, impurity=False, proportion=False, precision=0, fontsize=7)
    plt.title("Decision Tree Regression — cây thực tế sau khi train")
    plt.tight_layout()
    plt.savefig(ART / "decision_tree.png", bbox_inches="tight")
    plt.close()
    # Actual-vs-predicted chart from the held-out test set.
    plt.figure(figsize=(8, 7), dpi=150)
    plt.scatter(y_test, pred_test, alpha=0.55, s=18, color="#2563eb", edgecolors="none")
    lo, hi = float(min(y_test.min(), pred_test.min())), float(max(y_test.max(), pred_test.max()))
    plt.plot([lo, hi], [lo, hi], "--", color="#ef4444", label="Dự đoán hoàn hảo")
    plt.xlabel("Giá thực tế (SalePrice)"); plt.ylabel("Giá dự đoán")
    plt.title("Test set: giá thực tế và giá dự đoán"); plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(ART / "actual_vs_predicted.png", bbox_inches="tight"); plt.close()
    # Text representation for reproducible explanation in README/UI.
    (ART / "tree_rules.txt").write_text(export_text(model, feature_names=[LABELS[x] for x in FEATURES], decimals=0), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

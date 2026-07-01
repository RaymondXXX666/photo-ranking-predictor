from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from lightgbm import LGBMRanker


DATA_DIR = Path("processed")
ID_COLS = ["shoot_id", "scene_id", "image_id"]


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")
    return train_df, valid_df, test_df


def add_relevance_label(df):
    """
    LightGBM ranker wants relevance label where larger = better.
    Our rank is smaller = better.

    Example:
    scene_size = 5
    rank 1 -> relevance 4
    rank 2 -> relevance 3
    rank 5 -> relevance 0
    """
    df = df.copy()

    if "scene_size" not in df.columns:
        df["scene_size"] = df.groupby(["shoot_id", "scene_id"])["image_id"].transform("count")

    df["relevance"] = df["scene_size"] - df["rank"]

    return df


def get_feature_cols(df):
    drop_cols = ID_COLS + ["rank", "relevance"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def sort_by_group(df):
    """
    LightGBM needs rows sorted by query group.
    Here each query group = one scene.
    """
    return df.sort_values(["shoot_id", "scene_id", "rank"]).reset_index(drop=True)


def get_group_sizes(df):
    return df.groupby(["shoot_id", "scene_id"]).size().values


def pairwise_accuracy(group):
    if len(group) < 2:
        return np.nan

    correct = 0
    total = 0

    true_ranks = group["rank"].values
    pred_scores = group["pred_score"].values

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            true_order = true_ranks[i] < true_ranks[j]
            pred_order = pred_scores[i] > pred_scores[j]

            if true_order == pred_order:
                correct += 1

            total += 1

    return correct / total if total > 0 else np.nan


def top1_accuracy(group):
    pred_best = group.sort_values("pred_score", ascending=False).iloc[0]
    true_best_rank = group["rank"].min()
    return 1.0 if pred_best["rank"] == true_best_rank else 0.0


def evaluate_predictions(df, name):
    scene_metrics = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        if len(group) < 2:
            continue

        true_rank = group["rank"].values
        pred_score = group["pred_score"].values

        if np.std(pred_score) == 0:
            spearman = np.nan
            kendall = np.nan
        else:
            spearman = spearmanr(true_rank, -pred_score).correlation
            kendall = kendalltau(true_rank, -pred_score).correlation

        scene_metrics.append({
            "shoot_id": shoot_id,
            "scene_id": scene_id,
            "n_images": len(group),
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_acc": pairwise_accuracy(group),
            "top1_acc": top1_accuracy(group),
        })

    metrics_df = pd.DataFrame(scene_metrics)

    print(f"\n{name} scene-level metrics")
    print("-" * 50)
    print("Scenes evaluated:", len(metrics_df))
    print("Mean Spearman:", metrics_df["spearman"].mean(skipna=True))
    print("Mean Kendall:", metrics_df["kendall"].mean(skipna=True))
    print("Mean Pairwise Acc:", metrics_df["pairwise_acc"].mean(skipna=True))
    print("Top-1 Accuracy:", metrics_df["top1_acc"].mean(skipna=True))

    return metrics_df


def main():
    train_df, valid_df, test_df = load_data()

    train_df = add_relevance_label(train_df)
    valid_df = add_relevance_label(valid_df)
    test_df = add_relevance_label(test_df)

    train_df = sort_by_group(train_df)
    valid_df = sort_by_group(valid_df)
    test_df = sort_by_group(test_df)

    feature_cols = get_feature_cols(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["relevance"]
    group_train = get_group_sizes(train_df)

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["relevance"]
    group_valid = get_group_sizes(valid_df)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)
    print("Number of features:", len(feature_cols))
    print("Train groups:", len(group_train))
    print("Valid groups:", len(group_valid))
    
    max_relevance = int(train_df["relevance"].max())
    label_gain = list(range(max_relevance + 1))

    print("Max relevance:", max_relevance)
    print("Label gain length:", len(label_gain))

    model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    label_gain=label_gain,
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[1, 3, 5],
    )

    for split_name, df in [("Valid", valid_df), ("Test", test_df)]:
        result_df = df[ID_COLS + ["rank", "scene_size", "relevance"]].copy()
        result_df["pred_score"] = model.predict(df[feature_cols])

        metrics_df = evaluate_predictions(result_df, split_name)

        pred_path = DATA_DIR / f"{split_name.lower()}_predictions_lgbm_ranker.csv"
        metrics_path = DATA_DIR / f"{split_name.lower()}_scene_metrics_lgbm_ranker.csv"

        result_df.to_csv(pred_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        print(f"\nSaved predictions to {pred_path}")
        print(f"Saved metrics to {metrics_path}")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = DATA_DIR / "feature_importance_lgbm_ranker.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nTop 30 LightGBM feature importances:")
    print(importance_df.head(30).to_string(index=False))
    print(f"\nSaved feature importance to {importance_path}")


if __name__ == "__main__":
    main()
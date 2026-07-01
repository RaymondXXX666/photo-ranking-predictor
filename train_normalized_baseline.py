from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path("processed")

ID_COLS = ["shoot_id", "scene_id", "image_id"]
TARGET_COL = "quality_score"


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")
    return train_df, valid_df, test_df


def add_quality_score(df):
    df = df.copy()

    # clean_data.py 已经算过 scene_size，并且 build_features.py 保留了它
    if "scene_size" not in df.columns:
        df["scene_size"] = df.groupby(["shoot_id", "scene_id"])["image_id"].transform("count")

    df["quality_score"] = 1.0 - (df["rank"] - 1) / (df["scene_size"] - 1)

    return df


def get_feature_cols(df):
    drop_cols = ID_COLS + ["rank", "quality_score", "scene_size"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def pairwise_accuracy(group):
    if len(group) < 2:
        return np.nan

    correct = 0
    total = 0

    true_ranks = group["rank"].values
    pred_scores = group["pred_score"].values

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            # rank 越小越好
            true_order = true_ranks[i] < true_ranks[j]

            # pred_score 越大越好
            pred_order = pred_scores[i] > pred_scores[j]

            if true_order == pred_order:
                correct += 1

            total += 1

    return correct / total if total > 0 else np.nan


def top1_accuracy(group):
    pred_best = group.sort_values("pred_score", ascending=False).iloc[0]
    true_best_rank = group["rank"].min()
    return 1.0 if pred_best["rank"] == true_best_rank else 0.0


def evaluate_by_scene(df, name):
    scene_metrics = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        if len(group) < 2:
            continue

        true_rank = group["rank"].values
        pred_score = group["pred_score"].values

        # 如果预测全一样，Spearman/Kendall 没意义
        if np.std(pred_score) == 0:
            spearman = np.nan
            kendall = np.nan
        else:
            # pred_score 越大越好；rank 越小越好，所以用 -pred_score
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

    print(f"\n{name} scene-level ranking metrics")
    print("-" * 50)
    print("Scenes evaluated:", len(metrics_df))
    print("Mean Spearman:", metrics_df["spearman"].mean(skipna=True))
    print("Mean Kendall:", metrics_df["kendall"].mean(skipna=True))
    print("Mean Pairwise Acc:", metrics_df["pairwise_acc"].mean(skipna=True))
    print("Top-1 Accuracy:", metrics_df["top1_acc"].mean(skipna=True))

    return metrics_df


def evaluate_regression(df, name):
    mae = mean_absolute_error(df["quality_score"], df["pred_score"])
    rmse = np.sqrt(mean_squared_error(df["quality_score"], df["pred_score"]))

    print(f"\n{name} normalized regression metrics")
    print("-" * 50)
    print("MAE:", mae)
    print("RMSE:", rmse)


def train_and_evaluate(model, model_name, train_df, valid_df, test_df, feature_cols):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    model.fit(X_train, y_train)

    for split_name, df in [("Valid", valid_df), ("Test", test_df)]:
        X = df[feature_cols]

        result_df = df[ID_COLS + ["rank", "scene_size", "quality_score"]].copy()
        result_df["pred_score"] = model.predict(X)

        evaluate_regression(result_df, split_name)
        scene_metrics = evaluate_by_scene(result_df, split_name)

        pred_path = DATA_DIR / f"{split_name.lower()}_predictions_normalized_{model_name}.csv"
        metrics_path = DATA_DIR / f"{split_name.lower()}_scene_metrics_normalized_{model_name}.csv"

        result_df.to_csv(pred_path, index=False)
        scene_metrics.to_csv(metrics_path, index=False)

        print(f"\nSaved predictions to {pred_path}")
        print(f"Saved scene metrics to {metrics_path}")


def main():
    train_df, valid_df, test_df = load_data()

    train_df = add_quality_score(train_df)
    valid_df = add_quality_score(valid_df)
    test_df = add_quality_score(test_df)

    feature_cols = get_feature_cols(train_df)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)
    print("Number of features:", len(feature_cols))

    print("\nQuality score check:")
    print(train_df[["rank", "scene_size", "quality_score"]].head(10))

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=3,
            random_state=42,
        ),
    }

    for model_name, model in models.items():
        train_and_evaluate(
            model=model,
            model_name=model_name,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )


if __name__ == "__main__":
    main()
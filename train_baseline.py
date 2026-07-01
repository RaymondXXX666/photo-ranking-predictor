from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_DIR = Path("processed")

ID_COLS = ["shoot_id", "scene_id", "image_id"]
TARGET_COL = "rank"


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    return train_df, valid_df, test_df


def get_xy(df):
    drop_cols = ID_COLS + [TARGET_COL]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y, feature_cols


def pairwise_accuracy(group):
    """
    在一个 scene 里面，看任意两张图的相对顺序是否预测正确。
    rank 越小越好，所以 pred_rank 越小也应该越好。
    """
    if len(group) < 2:
        return np.nan

    correct = 0
    total = 0

    true_ranks = group["rank"].values
    pred_scores = group["pred_rank"].values

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            true_order = true_ranks[i] < true_ranks[j]
            pred_order = pred_scores[i] < pred_scores[j]

            if true_order == pred_order:
                correct += 1

            total += 1

    return correct / total if total > 0 else np.nan


def top1_accuracy(group):
    """
    每个 scene 里，预测最好的那张是否真的是 rank=1。
    """
    if len(group) == 0:
        return np.nan

    pred_best = group.sort_values("pred_rank", ascending=True).iloc[0]
    return 1.0 if pred_best["rank"] == group["rank"].min() else 0.0


def evaluate_by_scene(df, name):
    scene_metrics = []

    for scene_id, group in df.groupby("scene_id"):
        if len(group) < 2:
            continue

        true_rank = group["rank"].values
        pred_rank = group["pred_rank"].values

        spearman = spearmanr(true_rank, pred_rank).correlation
        kendall = kendalltau(true_rank, pred_rank).correlation

        scene_metrics.append({
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

    if len(metrics_df) == 0:
        print("No valid scenes for evaluation.")
        return metrics_df

    print("Scenes evaluated:", len(metrics_df))
    print("Mean Spearman:", metrics_df["spearman"].mean())
    print("Mean Kendall:", metrics_df["kendall"].mean())
    print("Mean Pairwise Acc:", metrics_df["pairwise_acc"].mean())
    print("Top-1 Accuracy:", metrics_df["top1_acc"].mean())

    return metrics_df


def evaluate_regression(df, name):
    mae = mean_absolute_error(df["rank"], df["pred_rank"])
    rmse = np.sqrt(mean_squared_error(df["rank"], df["pred_rank"]))

    print(f"\n{name} regression metrics")
    print("-" * 50)
    print("MAE:", mae)
    print("RMSE:", rmse)


def train_and_evaluate_model(model, model_name, train_df, valid_df, test_df, feature_cols):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    model.fit(X_train, y_train)

    for name, df in [("Valid", valid_df), ("Test", test_df)]:
        X = df[feature_cols]

        result_df = df[ID_COLS + [TARGET_COL]].copy()
        result_df["pred_rank"] = model.predict(X)

        evaluate_regression(result_df, name)
        scene_metrics = evaluate_by_scene(result_df, name)

        output_path = DATA_DIR / f"{name.lower()}_predictions_{model_name}.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved predictions to {output_path}")

        metrics_path = DATA_DIR / f"{name.lower()}_scene_metrics_{model_name}.csv"
        scene_metrics.to_csv(metrics_path, index=False)
        print(f"Saved scene metrics to {metrics_path}")

    return model


def main():
    train_df, valid_df, test_df = load_data()

    X_train, y_train, feature_cols = get_xy(train_df)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)
    print("Number of features:", len(feature_cols))

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
    }

    trained_models = {}

    for model_name, model in models.items():
        trained_models[model_name] = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )


if __name__ == "__main__":
    main()
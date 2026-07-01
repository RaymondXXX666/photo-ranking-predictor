from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


DATA_DIR = Path("processed")
ID_COLS = ["shoot_id", "scene_id", "image_id"]


def add_quality_score(df):
    df = df.copy()

    if "scene_size" not in df.columns:
        df["scene_size"] = df.groupby(["shoot_id", "scene_id"])["image_id"].transform("count")

    df["quality_score"] = 1.0 - (df["rank"] - 1) / (df["scene_size"] - 1)

    return df


def main():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    train_df = add_quality_score(train_df)

    drop_cols = ID_COLS + ["rank", "quality_score"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["quality_score"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_path = DATA_DIR / "feature_importance_random_forest.csv"
    importance_df.to_csv(output_path, index=False)

    print("Top 30 features:")
    print(importance_df.head(30).to_string(index=False))

    print()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
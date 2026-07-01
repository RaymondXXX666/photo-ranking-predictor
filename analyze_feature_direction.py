from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


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
    train_df = add_quality_score(train_df)

    drop_cols = ID_COLS + ["rank", "quality_score"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    rows = []

    for feature in feature_cols:
        corr, p_value = spearmanr(train_df[feature], train_df["quality_score"])

        rows.append({
            "feature": feature,
            "spearman_corr_with_quality": corr,
            "p_value": p_value,
        })

    direction_df = pd.DataFrame(rows)
    direction_df = direction_df.sort_values(
        "spearman_corr_with_quality",
        ascending=False
    )

    output_path = DATA_DIR / "feature_direction_spearman.csv"
    direction_df.to_csv(output_path, index=False)

    print("Most positively correlated features:")
    print(direction_df.head(20).to_string(index=False))

    print("\nMost negatively correlated features:")
    print(direction_df.tail(20).sort_values(
        "spearman_corr_with_quality"
    ).to_string(index=False))

    print()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
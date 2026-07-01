from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


DATA_DIR = Path("processed")
ID_COLS = ["shoot_id", "scene_id", "image_id"]


def make_pairwise_dataset(df, feature_cols):
    X_pairs = []
    y_pairs = []

    for (_, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        X = group[feature_cols].values
        ranks = group["rank"].values

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                xi = X[i]
                xj = X[j]

                rank_i = ranks[i]
                rank_j = ranks[j]

                # label = 1 means i is better than j
                label = int(rank_i < rank_j)

                X_pairs.append(xi - xj)
                y_pairs.append(label)

                # reverse pair
                X_pairs.append(xj - xi)
                y_pairs.append(1 - label)

    return np.array(X_pairs), np.array(y_pairs)


def main():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")

    drop_cols = ID_COLS + ["rank"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train, y_train = make_pairwise_dataset(train_df, feature_cols)

    print("Pairwise train shape:", X_train.shape)
    print("Number of original features:", len(feature_cols))

    model = GradientBoostingClassifier(
        random_state=42,
    )

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 40 pairwise feature importances:")
    print(importance_df.head(40).to_string(index=False))

    output_path = DATA_DIR / "pairwise_gb_feature_importance.csv"
    importance_df.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
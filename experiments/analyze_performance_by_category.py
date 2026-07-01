from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("processed")


def image_level_pair_correctness(preds):
    rows = []

    for (shoot_id, scene_id), group in preds.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            correct = 0
            total = 0

            for j in range(len(group)):
                if i == j:
                    continue

                true_i_better = group.loc[i, "rank"] < group.loc[j, "rank"]
                pred_i_better = group.loc[i, "pred_score"] > group.loc[j, "pred_score"]

                if true_i_better == pred_i_better:
                    correct += 1

                total += 1

            rows.append({
                "shoot_id": group.loc[i, "shoot_id"],
                "scene_id": group.loc[i, "scene_id"],
                "image_id": group.loc[i, "image_id"],
                "rank": group.loc[i, "rank"],
                "image_pairwise_acc": correct / total if total > 0 else np.nan,
            })

    return pd.DataFrame(rows)


def main():
    preds = pd.read_csv(DATA_DIR / "test_predictions_pairwise_gradient_boosting_classifier.csv")
    features = pd.read_csv(DATA_DIR / "test_features.csv")

    image_acc = image_level_pair_correctness(preds)

    category_cols = [c for c in features.columns if c.startswith("category_")]

    merged = image_acc.merge(
        features[["shoot_id", "scene_id", "image_id", "rank"] + category_cols],
        on=["shoot_id", "scene_id", "image_id", "rank"],
        how="left",
    )

    rows = []

    for cat in category_cols:
        subset = merged[merged[cat] == 1]

        if len(subset) == 0:
            continue

        rows.append({
            "category": cat,
            "n_images": len(subset),
            "mean_image_pairwise_acc": subset["image_pairwise_acc"].mean(),
        })

    summary = pd.DataFrame(rows).sort_values("mean_image_pairwise_acc", ascending=False)

    print(summary.to_string(index=False))

    output_path = DATA_DIR / "performance_by_category.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
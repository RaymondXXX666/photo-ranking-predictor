from pathlib import Path
import pandas as pd

DATA_DIR = Path("processed")


def main():
    preds_path = DATA_DIR / "test_predictions_pairwise_gradient_boosting_classifier.csv"
    preds = pd.read_csv(preds_path)

    rows = []

    for (shoot_id, scene_id), group in preds.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                rank_i = group.loc[i, "rank"]
                rank_j = group.loc[j, "rank"]

                score_i = group.loc[i, "pred_score"]
                score_j = group.loc[j, "pred_score"]

                true_order = rank_i < rank_j
                pred_order = score_i > score_j

                rows.append({
                    "shoot_id": shoot_id,
                    "scene_id": scene_id,
                    "rank_i": rank_i,
                    "rank_j": rank_j,
                    "rank_gap": abs(rank_i - rank_j),
                    "correct": int(true_order == pred_order),
                })

    pair_df = pd.DataFrame(rows)

    summary = (
        pair_df.groupby("rank_gap")
        .agg(
            n_pairs=("correct", "count"),
            pairwise_acc=("correct", "mean"),
        )
        .reset_index()
        .sort_values("rank_gap")
    )

    print(summary.to_string(index=False))

    output_path = DATA_DIR / "pairwise_accuracy_by_rank_gap.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
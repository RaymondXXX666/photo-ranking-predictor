from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_DIR = Path("processed")
ID_COLS = ["shoot_id", "scene_id", "image_id"]


def make_pairwise_dataset(df, feature_cols):
    X_pairs = []
    y_pairs = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
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

                label = int(rank_i < rank_j)

                X_pairs.append(xi - xj)
                y_pairs.append(label)

                X_pairs.append(xj - xi)
                y_pairs.append(1 - label)

    return np.array(X_pairs), np.array(y_pairs)


def safe_corr(func, a, b):
    try:
        value = func(a, b).correlation
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def evaluate_scene_ranking(df, feature_cols, model):
    scene_metrics = []
    pred_rows = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        X = group[feature_cols].values
        ranks = group["rank"].values

        scores = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue

                diff = (X[i] - X[j]).reshape(1, -1)
                prob_i_beats_j = model.predict_proba(diff)[0, 1]
                scores[i] += prob_i_beats_j

        pred_order = np.argsort(-scores)
        true_order = np.argsort(ranks)

        # Convert scores so higher score means better, rank lower means better.
        spearman = safe_corr(spearmanr, -ranks, scores)
        kendall = safe_corr(kendalltau, -ranks, scores)

        correct_pairs = 0
        total_pairs = 0

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                true_i_better = ranks[i] < ranks[j]
                pred_i_better = scores[i] > scores[j]

                if true_i_better == pred_i_better:
                    correct_pairs += 1

                total_pairs += 1

        pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        top1_acc = int(pred_order[0] == true_order[0])

        scene_metrics.append({
            "shoot_id": shoot_id,
            "scene_id": scene_id,
            "n_images": len(group),
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_acc": pairwise_acc,
            "top1_acc": top1_acc,
        })

        for idx in range(len(group)):
            pred_rows.append({
                "shoot_id": shoot_id,
                "scene_id": scene_id,
                "image_id": group.loc[idx, "image_id"],
                "rank": group.loc[idx, "rank"],
                "pred_score": scores[idx],
            })

    metrics_df = pd.DataFrame(scene_metrics)
    preds_df = pd.DataFrame(pred_rows)

    summary = {
        "mean_spearman": metrics_df["spearman"].mean(),
        "mean_kendall": metrics_df["kendall"].mean(),
        "mean_pairwise_acc": metrics_df["pairwise_acc"].mean(),
        "top1_acc": metrics_df["top1_acc"].mean(),
        "n_scenes": len(metrics_df),
    }

    return summary, metrics_df, preds_df


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * 60)
    print("Scenes evaluated:", summary["n_scenes"])
    print("Mean Spearman:", summary["mean_spearman"])
    print("Mean Kendall:", summary["mean_kendall"])
    print("Mean Pairwise Acc:", summary["mean_pairwise_acc"])
    print("Top-1 Accuracy:", summary["top1_acc"])


def main():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    drop_cols = ID_COLS + ["rank"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)
    print("Number of features:", len(feature_cols))

    X_train, y_train = make_pairwise_dataset(train_df, feature_cols)

    print("Pairwise train shape:", X_train.shape)
    print("Pairwise label mean:", y_train.mean())

    param_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [1, 3, 5],
        "subsample": [0.8, 1.0],
    }

    all_results = []
    best_model = None
    best_params = None
    best_valid_summary = None
    best_score = -1

    keys = list(param_grid.keys())

    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))

        model = GradientBoostingClassifier(
            random_state=42,
            **params,
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_prob = model.predict_proba(X_train)[:, 1]

        train_acc = accuracy_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_prob)

        valid_summary, _, _ = evaluate_scene_ranking(valid_df, feature_cols, model)

        # Main selection metric: validation pairwise accuracy.
        # Tie-breaker: top-1 accuracy.
        selection_score = (
            valid_summary["mean_pairwise_acc"]
            + 0.05 * valid_summary["top1_acc"]
        )

        row = {
            **params,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "valid_spearman": valid_summary["mean_spearman"],
            "valid_kendall": valid_summary["mean_kendall"],
            "valid_pairwise_acc": valid_summary["mean_pairwise_acc"],
            "valid_top1_acc": valid_summary["top1_acc"],
            "selection_score": selection_score,
        }

        all_results.append(row)

        print(
            params,
            "valid_pairwise=",
            round(valid_summary["mean_pairwise_acc"], 4),
            "valid_top1=",
            round(valid_summary["top1_acc"], 4),
            "train_auc=",
            round(train_auc, 4),
        )

        if selection_score > best_score:
            best_score = selection_score
            best_model = model
            best_params = params
            best_valid_summary = valid_summary

    results_df = pd.DataFrame(all_results).sort_values(
        ["selection_score", "valid_pairwise_acc", "valid_top1_acc"],
        ascending=False,
    )

    results_path = DATA_DIR / "pairwise_gb_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("Best params selected by validation:")
    print(best_params)
    print_summary("Best validation summary", best_valid_summary)

    test_summary, test_metrics, test_preds = evaluate_scene_ranking(test_df, feature_cols, best_model)
    print_summary("Test summary for selected model", test_summary)

    test_metrics.to_csv(DATA_DIR / "test_scene_metrics_pairwise_gb_tuned.csv", index=False)
    test_preds.to_csv(DATA_DIR / "test_predictions_pairwise_gb_tuned.csv", index=False)

    print(f"\nSaved tuning results to {results_path}")
    print("Saved tuned test metrics to processed/test_scene_metrics_pairwise_gb_tuned.csv")
    print("Saved tuned test predictions to processed/test_predictions_pairwise_gb_tuned.csv")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_DIR = Path("processed")
ID_COLS = ["shoot_id", "scene_id", "image_id"]


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_features.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid_features.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")
    return train_df, valid_df, test_df


def get_feature_cols(df):
    drop_cols = ID_COLS + ["rank"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def make_enhanced_pair_features(feat_a, feat_b):
    diff = feat_a - feat_b
    abs_diff = np.abs(diff)

    return np.concatenate([
        diff,
        abs_diff,
        feat_a,
        feat_b,
    ])


def make_pairwise_data(df, feature_cols):
    """
    For each scene, create enhanced pairwise samples:
    x = [A - B, abs(A - B), A, B]
    y = 1 if A has better rank than B else 0

    rank smaller = better.
    """
    X_pairs = []
    y_pairs = []
    meta_pairs = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row_i = group.iloc[i]
                row_j = group.iloc[j]

                feat_i = row_i[feature_cols].values.astype(float)
                feat_j = row_j[feature_cols].values.astype(float)

                rank_i = row_i["rank"]
                rank_j = row_j["rank"]

                if rank_i == rank_j:
                    continue

                # sample 1: i compared with j
                X_pairs.append(make_enhanced_pair_features(feat_i, feat_j))
                y_pairs.append(1 if rank_i < rank_j else 0)

                meta_pairs.append({
                    "shoot_id": shoot_id,
                    "scene_id": scene_id,
                    "image_a": row_i["image_id"],
                    "image_b": row_j["image_id"],
                    "rank_a": rank_i,
                    "rank_b": rank_j,
                })

                # sample 2: reverse comparison
                X_pairs.append(make_enhanced_pair_features(feat_j, feat_i))
                y_pairs.append(1 if rank_j < rank_i else 0)

                meta_pairs.append({
                    "shoot_id": shoot_id,
                    "scene_id": scene_id,
                    "image_a": row_j["image_id"],
                    "image_b": row_i["image_id"],
                    "rank_a": rank_j,
                    "rank_b": rank_i,
                })

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)
    meta_df = pd.DataFrame(meta_pairs)

    return X_pairs, y_pairs, meta_df


def score_scene_with_pairwise_model(model, group, feature_cols):
    """
    For a scene, compare every image against every other image.
    Each image gets a score based on how often it is predicted to beat others.
    """
    group = group.reset_index(drop=True)
    n = len(group)

    scores = np.zeros(n)

    features = group[feature_cols].values.astype(float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pair_features = make_enhanced_pair_features(features[i], features[j])
            prob_i_beats_j = model.predict_proba(pair_features.reshape(1, -1))[0, 1]
            scores[i] += prob_i_beats_j

    result = group[ID_COLS + ["rank"]].copy()
    result["pred_score"] = scores

    return result


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


def evaluate_predictions(pred_df, name):
    scene_metrics = []

    for (shoot_id, scene_id), group in pred_df.groupby(["shoot_id", "scene_id"]):
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


def predict_split(model, df, feature_cols):
    all_scene_preds = []

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        if len(group) < 2:
            continue

        scene_pred = score_scene_with_pairwise_model(model, group, feature_cols)
        all_scene_preds.append(scene_pred)

    return pd.concat(all_scene_preds, ignore_index=True)


def train_and_evaluate(model, model_name, train_df, valid_df, test_df, feature_cols):
    print("\n" + "=" * 80)
    print(f"Training pairwise model: {model_name}")
    print("=" * 80)

    X_train_pairs, y_train_pairs, _ = make_pairwise_data(train_df, feature_cols)

    print("Pairwise train shape:", X_train_pairs.shape)
    print("Pairwise label mean:", y_train_pairs.mean())

    model.fit(X_train_pairs, y_train_pairs)

    train_pair_pred = model.predict(X_train_pairs)
    train_pair_prob = model.predict_proba(X_train_pairs)[:, 1]

    print("\nPairwise training metrics")
    print("-" * 50)
    print("Pairwise train accuracy:", accuracy_score(y_train_pairs, train_pair_pred))

    try:
        print("Pairwise train AUC:", roc_auc_score(y_train_pairs, train_pair_prob))
    except Exception:
        pass

    for split_name, split_df in [("Valid", valid_df), ("Test", test_df)]:
        pred_df = predict_split(model, split_df, feature_cols)
        metrics_df = evaluate_predictions(pred_df, split_name)

        pred_path = DATA_DIR / f"{split_name.lower()}_predictions_pairwise_{model_name}.csv"
        metrics_path = DATA_DIR / f"{split_name.lower()}_scene_metrics_pairwise_{model_name}.csv"

        pred_df.to_csv(pred_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        print(f"\nSaved predictions to {pred_path}")
        print(f"Saved metrics to {metrics_path}")


def main():
    train_df, valid_df, test_df = load_data()

    feature_cols = get_feature_cols(train_df)

    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test:", test_df.shape)
    print("Number of features:", len(feature_cols))

    models = {
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=800,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "gradient_boosting_classifier": GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.015,
            max_depth=2,
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
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import spearmanr, kendalltau

from feature_builder import build_features_from_uploaded_json


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "final_pairwise_gb_model.joblib"
FEATURE_COLS_PATH = BASE_DIR / "model" / "final_feature_columns.json"


app = FastAPI(title="Wedding Photo Ranking Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load(MODEL_PATH)

with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)


def safe_corr(func, a, b):
    try:
        result = func(a, b)
        value = result.correlation
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def prepare_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df


def evaluate_scene_ranking(df: pd.DataFrame):
    df = prepare_feature_df(df)

    scene_metrics = []
    predictions = []

    total_images = int(len(df))
    total_scenes = int(df["scene_id"].nunique())
    ranked_images = int(df["rank"].notna().sum())
    unranked_images = int(df["rank"].isna().sum())

    for (shoot_id, scene_id), group in df.groupby(["shoot_id", "scene_id"]):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        # ------------------------------------------------------------
        # Prediction for all images in the scene
        # ------------------------------------------------------------
        X = group[FEATURE_COLS].values
        scores = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue

                diff = (X[i] - X[j]).reshape(1, -1)
                prob_i_beats_j = model.predict_proba(diff)[0, 1]
                scores[i] += prob_i_beats_j

        predicted_order = np.argsort(-scores)

        for idx in range(len(group)):
            rank_value = group.loc[idx].get("rank", None)

            predictions.append({
                "shoot_id": str(shoot_id),
                "scene_id": str(scene_id),
                "image_id": str(group.loc[idx, "image_id"]),
                "true_rank": None if pd.isna(rank_value) else float(rank_value),
                "pred_score": float(scores[idx]),
                "predicted_position": int(np.where(predicted_order == idx)[0][0] + 1),
            })

        # ------------------------------------------------------------
        # Evaluation only on images with valid rank labels
        # ------------------------------------------------------------
        eval_group = group[group["rank"].notna()].reset_index(drop=True)

        # Evaluation requires at least two valid ranked images.
        if len(eval_group) < 2:
            continue

        X_eval = eval_group[FEATURE_COLS].values
        ranks = eval_group["rank"].values

        eval_scores = np.zeros(len(eval_group), dtype=float)

        for i in range(len(eval_group)):
            for j in range(len(eval_group)):
                if i == j:
                    continue

                diff = (X_eval[i] - X_eval[j]).reshape(1, -1)
                prob_i_beats_j = model.predict_proba(diff)[0, 1]
                eval_scores[i] += prob_i_beats_j

        eval_predicted_order = np.argsort(-eval_scores)
        true_order = np.argsort(ranks)

        correct_pairs = 0
        total_pairs = 0

        for i in range(len(eval_group)):
            for j in range(i + 1, len(eval_group)):
                true_i_better = ranks[i] < ranks[j]
                pred_i_better = eval_scores[i] > eval_scores[j]

                if true_i_better == pred_i_better:
                    correct_pairs += 1

                total_pairs += 1

        pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        top1_acc = int(eval_predicted_order[0] == true_order[0])

        spearman = safe_corr(spearmanr, -ranks, eval_scores)
        kendall = safe_corr(kendalltau, -ranks, eval_scores)

        scene_metrics.append({
            "shoot_id": str(shoot_id),
            "scene_id": str(scene_id),
            "n_images": int(len(eval_group)),
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_acc": float(pairwise_acc),
            "top1_acc": int(top1_acc),
        })

    metrics_df = pd.DataFrame(scene_metrics)

    if len(metrics_df) == 0:
        summary = {
            "mode": "prediction_only",
            "total_images": total_images,
            "total_scenes": total_scenes,
            "ranked_images": ranked_images,
            "unranked_images": unranked_images,
            "n_scenes_evaluated": 0,
            "mean_spearman": None,
            "mean_kendall": None,
            "mean_pairwise_acc": None,
            "top1_acc": None,
            "message": "No complete rank labels found. Returned predictions only.",
        }
    else:
        summary = {
            "mode": "evaluation",
            "total_images": total_images,
        "total_scenes": total_scenes,
        "ranked_images": ranked_images,
        "unranked_images": unranked_images,
            "n_scenes_evaluated": int(len(metrics_df)),
            "mean_spearman": float(metrics_df["spearman"].mean()),
            "mean_kendall": float(metrics_df["kendall"].mean()),
            "mean_pairwise_acc": float(metrics_df["pairwise_acc"].mean()),
            "top1_acc": float(metrics_df["top1_acc"].mean()),
        }

    return {
        "summary": summary,
        "scene_metrics": scene_metrics,
        "predictions": predictions,
    }


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": len(FEATURE_COLS),
    }


@app.post("/evaluate")
async def evaluate_json(file: UploadFile = File(...)):
    content = await file.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON file.",
        }

    feature_df = build_features_from_uploaded_json(data)

    if len(feature_df) == 0:
        return {
            "error": "No image records found in uploaded JSON.",
        }

    result = evaluate_scene_ranking(feature_df)

    return result

@app.post("/debug-ranks")
async def debug_ranks(file: UploadFile = File(...)):
    content = await file.read()
    data = json.loads(content)

    feature_df = build_features_from_uploaded_json(data)

    rank_counts = {}
    for key, value in feature_df["rank"].value_counts(dropna=False).items():
        if pd.isna(key):
            clean_key = "NaN"
        else:
            clean_key = str(float(key))

        rank_counts[clean_key] = int(value)

    non_null_ranks = feature_df["rank"].dropna()

    return {
        "n_rows": int(len(feature_df)),
        "rank_counts": rank_counts,
        "min_rank": None if non_null_ranks.empty else float(non_null_ranks.min()),
        "n_rank_null": int(feature_df["rank"].isna().sum()),
        "n_rank_zero": int((feature_df["rank"] == 0).sum()),
    }
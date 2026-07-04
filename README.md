# Wedding Photo Ranking Predictor

This repository contains my take-home assessment submission for a scene-level wedding photo ranking task.

The goal is to predict the preferred ordering of photos within each scene using the provided face detection metadata. I approached the task as a **within-scene learning-to-rank problem** rather than a global classification or raw-rank regression problem, because rank values are only meaningful relative to other photos in the same scene.

The final system uses a **pairwise Gradient Boosting model** trained on feature differences between images from the same scene. The submission includes feature extraction, model training, evaluation, error analysis, feature ablation, a FastAPI backend, and a deployed web demo.

- Live demo: https://photo-ranking-predictor.vercel.app/
- Repository: https://github.com/RaymondXXX666/photo-ranking-predictor

---

## Executive Summary

This project predicts the relative ranking of wedding photos within each scene. The key modeling decision is that `rank` is not a global photo-quality label: rank 1 only means the best image inside its own scene.

I therefore formulate the task as a within-scene ranking problem rather than ordinary regression. The final model uses pairwise comparisons between images from the same scene, then aggregates pairwise win probabilities into a final scene-level ordering.

The solution includes:

- data cleaning for noisy face-detection records
- face-level to image-level feature aggregation
- handling of missing ranks, no-face images, and multi-face images
- shoot-level train/validation/test split to reduce leakage
- baseline experiments including raw rank regression, normalized rank regression, pairwise ranking, and LightGBM ranker
- scene-level ranking evaluation using Pairwise Accuracy, Top-1 Accuracy, Top-3 Coverage, Spearman, and Kendall Tau
- a deployed demo for prediction and evaluation

I compared raw rank regression, normalized rank regression, pairwise ranking, LightGBM ranker, and heuristic/domain features. The model learns useful ranking signals from face detection metadata, especially focus, eye state, subject confidence, and main-subject quality.

## Final Test Results

The final model was evaluated on a held-out shoot-level test split.

| Model | Pairwise Accuracy | Top-1 Accuracy | Best in Top 3 | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|
| Pairwise Gradient Boosting | 82.02% | 77.06% | 95.00% on 20 eligible scenes | 0.6617 | 0.6405 |

`Best in Top 3` is only calculated for scenes with more than 3 ranked candidates, because Top-3 is trivial for scenes with 2–3 images.

---

## Project Structure

~~~text
.
├── README.md
├── scripts/
│   ├── build_features.py
│   └── train_pairwise_baseline.py
├── experiments/
│   └── analyze_pairwise_by_rank_gap.py
├── backend/
│   ├── app.py
│   ├── feature_builder.py
│   ├── model/
│   │   ├── final_pairwise_gb_model.joblib
│   │   └── final_feature_columns.json
│   └── frontend/
│       └── app/
├── processed/
└── data/
~~~

Main components:

- `scripts/build_features.py` — builds image-level features from the provided face detection metadata.
- `scripts/train_pairwise_baseline.py` — trains and evaluates pairwise ranking models.
- `experiments/analyze_pairwise_by_rank_gap.py` — analyzes pairwise accuracy by true rank gap.
- `backend/app.py` — FastAPI backend for upload, prediction, and evaluation.
- `backend/feature_builder.py` — backend feature extraction used by the live demo.
- `backend/frontend/` — Next.js frontend for the deployed demo and embedded project report.

---

## Problem Framing

The provided labels are scene-relative ranks. A rank of `1` means the best photo within a particular scene, not the best photo globally across the dataset.

Because of this, I treated the task as a **scene-level ranking problem**. For each scene, the model compares pairs of images and learns which image should be preferred.

Instead of predicting an absolute score globally, the final model predicts pairwise preferences:

~~~text
Given image A and image B from the same scene:
predict whether A should rank higher than B.
~~~

At inference time, each image receives a relative tournament score by summing its predicted pairwise win probabilities against other images in the same scene.

---

## Approach Summary

I compared several formulations and feature/model variants before selecting the final approach.

| Experiment / Variant | Description | Result / Observation | Decision |
|---|---|---|---|
| Raw-rank regression | Predicted absolute rank values directly. | Less suitable because ranks are only meaningful within each scene. | Rejected |
| Normalized-rank regression | Predicted rank normalized by scene size. | Better than raw-rank regression, but still pointwise and less aligned with the ranking task. | Rejected |
| Pairwise Random Forest | Learned pairwise preferences from within-scene feature differences. | Strong baseline, but weaker than Gradient Boosting on the held-out test set. | Baseline |
| Pairwise Gradient Boosting | Learned pairwise preferences from within-scene feature differences. | Best overall held-out performance. | Selected |
| Robust worst-case quality features | Added main-subject and group-level quality risk features. | Improved generalization and matched common photo-culling failure modes. | Kept |
| Composite quality penalty | Tested hand-weighted combinations of focus, eye-state, and pose penalties. | Reduced held-out generalization compared with more atomic quality features. | Rejected |

---

## Feature Engineering

The model uses image-level features extracted from the provided face detection metadata.

Feature groups include:

- Scene/category metadata
- Face count and selected-face count
- Face focus and confidence statistics
- Face area and position statistics
- Eye-state probabilities, including open, closed, midblink, covered, glasses, partially open, and barely open states
- Main-subject specific quality features
- Group-level eye and focus quality features
- Robust worst-case features, such as:
  - maximum main-subject eye penalty
  - maximum main-subject focus penalty
  - group minimum open-eye probability
  - group maximum closed-eye probability
  - group maximum covered-eye probability

The final feature set keeps the original **continuous probability and quality features**. Threshold-based features are added only as lightweight warning signals; they do not replace the continuous metadata.

---

## Feature Ablation

I treated feature engineering as an empirical question rather than assuming every manually designed feature would help.

| Feature Set | Description | Held-out Test Result | Decision |
|---|---|---|---|
| Continuous metadata features | Focus scores, confidence, eye-state probabilities, face size, position, and subject metadata. | Kept as the base signal. These preserve the original probability granularity. | Kept |
| Continuous + threshold warning features | Added warning features for common culling failure modes such as closed eyes, covered eyes, low focus, and poor main-subject quality. | Best final result: 82.02% pairwise accuracy, 77.06% Top-1 accuracy, 0.6617 Spearman, and 0.6405 Kendall. | Selected |
| Without threshold warning features | Removed warning features while retaining continuous probability and quality features. | Dropped to 80.60% pairwise accuracy, 76.15% Top-1 accuracy, 0.6400 Spearman, and 0.6120 Kendall. | Rejected |
| Composite quality penalty | Tested hand-weighted combinations of focus, eye-state, and pose penalties. | Reduced held-out generalization compared with keeping more atomic quality signals. | Rejected |

This suggests that lightweight domain-informed warning features were useful in this dataset, but overly rigid hand-composed scoring rules could hurt generalization.

---

## Evaluation Metrics

The project reports several scene-level ranking metrics.

| Metric | Meaning |
|---|---|
| Pairwise Accuracy | How often the model orders two images correctly within the same scene. |
| Top-1 Accuracy | Whether the model selects the same best image as the ground-truth rank 1 image. |
| Best in Top 3 | Whether the true best image appears in the model's top 3 recommendations. Only evaluated for scenes with more than 3 ranked candidates. |
| Spearman Correlation | Rank-order correlation between predicted and true scene-level ordering. |
| Kendall Correlation | Pairwise rank correlation between predicted and true ordering. |

---

## Rank Gap Error Analysis

I analyzed pairwise accuracy by the true rank gap between two images.

| True Rank Gap | Example Comparisons | Pairs | Pairwise Accuracy |
|---:|---|---:|---:|
| 1 | Rank 1 vs 2, 2 vs 3, 3 vs 4 | 189 | 78.31% |
| 2 | Rank 1 vs 3, 2 vs 4, 3 vs 5 | 80 | 83.75% |
| 3 | Rank 1 vs 4, 2 vs 5, 3 vs 6 | 35 | 91.43% |
| 4 | Rank 1 vs 5, 2 vs 6, 3 vs 7 | 15 | 93.33% |
| 5 | Rank 1 vs 6, 2 vs 7, 3 vs 8 | 7 | 85.71% |
| 6 | Rank 1 vs 7, 2 vs 8 | 3 | 100.00% |
| 7 | Rank 1 vs 8 | 1 | 100.00% |

The model is strongest at separating clearly good and clearly weak images. Adjacent-rank comparisons are the hardest, which matches the nature of wedding photo culling: rank 1 vs rank 2 may depend on subtle expression, eye-state, focus, or photographer preference differences.

---

## Running the Training Pipeline

From the project root:

~~~bash
python scripts/build_features.py
python scripts/train_pairwise_baseline.py
~~~

The training script writes predictions, metrics, the final model, and feature columns to `processed/`.

Example outputs:

~~~text
processed/final_pairwise_gb_model.joblib
processed/final_feature_columns.json
processed/test_predictions_pairwise_gradient_boosting_classifier.csv
processed/test_scene_metrics_pairwise_gradient_boosting_classifier.csv
~~~

To update the backend model artifacts after retraining:

~~~bash
cp processed/final_pairwise_gb_model.joblib backend/model/final_pairwise_gb_model.joblib
cp processed/final_feature_columns.json backend/model/final_feature_columns.json
~~~

---

## Running the Backend Locally

From the `backend/` directory:

~~~bash
uvicorn app:app --reload
~~~

The API provides:

- `GET /` — health check
- `POST /evaluate` — upload a `face_detections.json` file, generate predicted rankings, and return evaluation metrics when ranks are available
- `POST /debug-ranks` — inspect rank-label parsing

---

## Running the Frontend Locally

From the `backend/frontend/` directory:

~~~bash
npm install
npm run dev
~~~

Then open:

~~~text
http://localhost:3000
~~~

---

## Notes on Prediction Scores

The displayed scores are **relative pairwise tournament scores** within each scene. They are used for ordering images inside the same scene and are not calibrated probabilities across different scenes.

Very similar images may receive tied or near-tied scores, especially in small scenes with only two or three candidates.

---

## Design Decisions

| Decision | Reason |
|---|---|
| Pairwise ranking instead of raw rank regression | Ranks are scene-relative, not global labels |
| Shoot-level split | Reduces leakage from visually similar photos in the same wedding |
| Gradient Boosting classifier | Suitable for small tabular data and fast inference |
| Preserve continuous face probabilities | Avoids discarding signal from eye-state, focus, and confidence scores |
| Add threshold warning flags | Provides interpretable auxiliary indicators without replacing continuous features |
| Evaluate with Top-1, Top-3, pairwise accuracy, Spearman, Kendall | Metrics match the product goal of selecting the best images within a scene |

## Limitations and Future Work

- The validation and test sets are each based on a single held-out shoot. This avoids image-level leakage across scenes from the same shoot, but the final numbers may still be sensitive to the style and difficulty of those particular shoots. With more data, I would run leave-one-shoot-out or grouped cross-validation across shoots.
- The current model does not use explicit scene-level normalization. Since training is pairwise within each scene, shared scene-level offsets are partially reduced by feature differencing, but explicit scene-normalized features could be tested in a future iteration.
- Some quality features use hand-defined thresholds. These are used as interpretable warning features, not as hard ranking rules.
- The model relies on upstream face detection metadata and does not directly inspect raw image pixels. Future work could combine metadata features with lightweight visual embeddings for composition, emotion, and overall image aesthetics.
- The current demo focuses on metadata-based ranking and evaluation. A production system would also need stronger input validation, monitoring, user feedback loops, and calibration across more diverse shoots.

---

## Summary

This project demonstrates an end-to-end applied ML workflow for a realistic ranking task:

1. Reframing scene-relative labels as a learning-to-rank problem.
2. Building interpretable features from face detection metadata.
3. Comparing multiple modelling formulations.
4. Validating feature engineering with ablation.
5. Evaluating model behavior with ranking metrics and rank-gap error analysis.
6. Deploying the result as a working web demo.


<img width="1154" height="901" alt="image" src="https://github.com/user-attachments/assets/38d0412d-9587-4899-817e-6e17701ed0bb" />

<img width="1111" height="707" alt="image" src="https://github.com/user-attachments/assets/da2f87a9-138c-44f6-b9de-3c401eefdab9" />

<img width="1533" height="893" alt="image" src="https://github.com/user-attachments/assets/9e91c5c5-757f-49aa-8b14-e831e19ddb0e" />



# Photo Ranking Dataset - Data Structure Documentation

## Live Demo Link: https://photo-ranking-predictor.vercel.app

# Wedding Photo Ranking Predictor

This repository contains my take-home assessment submission for a scene-level wedding photo ranking task.

The goal is to predict the preferred ordering of photos within each scene using the provided face detection metadata. I approached the task as a **within-scene learning-to-rank problem** rather than a global classification or raw-rank regression problem, because rank values are only meaningful relative to other photos in the same scene.

The final system uses a **pairwise Gradient Boosting model** trained on feature differences between images from the same scene. The submission includes feature extraction, model training, evaluation, error analysis, feature ablation, a FastAPI backend, and a deployed web demo.

- Live demo: https://photo-ranking-predictor.vercel.app/
- Repository: https://github.com/RaymondXXX666/photo-ranking-predictor

---

## Final Test Results

The final model was evaluated on a held-out shoot-level test split.

| Model | Pairwise Accuracy | Top-1 Accuracy | Best in Top 3 | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|
| Pairwise Gradient Boosting | 82.02% | 77.06% | 95.00% on 20 eligible scenes | 0.6617 | 0.6405 |

`Best in Top 3` is only calculated for scenes with more than 3 ranked candidates, because Top-3 is trivial for scenes with 2–3 images.

---

## Project Structure

```text
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

<img width="1154" height="901" alt="image" src="https://github.com/user-attachments/assets/38d0412d-9587-4899-817e-6e17701ed0bb" />

<img width="1111" height="707" alt="image" src="https://github.com/user-attachments/assets/da2f87a9-138c-44f6-b9de-3c401eefdab9" />

<img width="1533" height="893" alt="image" src="https://github.com/user-attachments/assets/9e91c5c5-757f-49aa-8b14-e831e19ddb0e" />



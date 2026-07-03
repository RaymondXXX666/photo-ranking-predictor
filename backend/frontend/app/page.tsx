"use client";

import { useState, type ReactNode } from "react";

type Summary = {
  mode: string;
  total_images: number;
  total_scenes: number;
  ranked_images: number;
  unranked_images: number;
  n_scenes_evaluated: number;
  mean_spearman: number | null;
  mean_kendall: number | null;
  mean_pairwise_acc: number | null;
  top1_acc: number | null;
  top3_contains_best?: number | null;
  top3_eligible_scenes?: number;
  message?: string;
};

type SceneMetric = {
  shoot_id: string;
  scene_id: string;
  n_images: number;
  spearman: number;
  kendall: number;
  pairwise_acc: number;
  top1_acc: number;
  top3_contains_best?: number | null;
};

type Prediction = {
  shoot_id: string;
  scene_id: string;
  image_id: string;
  true_rank: number | null;
  pred_score: number;
  predicted_position: number;
};

type EvaluationResult = {
  summary: Summary;
  scene_metrics: SceneMetric[];
  predictions: Prediction[];
};

function formatPercent(value: number | null) {
  if (value === null || value === undefined) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value: number | null) {
  if (value === null || value === undefined) return "N/A";
  return value.toFixed(4);
}

export default function Home() {
  const [mode, setMode] = useState<"predict" | "evaluate">("predict");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleEvaluate() {
    if (!file) {
      setError("Please choose a JSON file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const apiBaseUrl = "https://photo-ranking-predictor.onrender.com";

      const response = await fetch(`${apiBaseUrl}/evaluate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <section className="mb-10">
          <p className="mb-3 text-sm uppercase tracking-[0.3em] text-neutral-400">
            Wedding Photo Ranking
          </p>

          <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
            Photo Ranking Predictor
          </h1>
          <h2 className="mb-4 max-w-4xl text-2xl font-semibold tracking-tight text-neutral-200 md:text-3xl">
            Scene-level learning-to-rank for wedding photo culling.
          </h2>

          <p className="max-w-4xl text-lg text-neutral-300">
            Upload a face_detections.json file to predict within-scene photo rankings.
            The system builds interpretable image-level features from face detection
            metadata, compares images pairwise within each scene, and surfaces the
            strongest candidates for review. When ground-truth ranks are available, the
            demo also evaluates ranking quality using pairwise accuracy, Top-1 accuracy,
            rank correlation, and larger-scene Top-3 coverage.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a
              href="#demo"
              className="rounded-xl bg-white px-5 py-3 font-semibold text-neutral-950 transition hover:bg-neutral-200"
            >
              Try Demo
            </a>
            <a
              href="#readme"
              className="rounded-xl border border-neutral-700 px-5 py-3 font-semibold text-neutral-200 transition hover:border-neutral-400"
            >
              Read Method
            </a>
          </div>
          <p className="max-w-3xl text-lg text-neutral-300">
            Prediction mode ranks all images, including images without ground-truth labels.
            Evaluation mode calculates metrics only on images with valid rank labels.
          </p>
        </section>

        <section id="demo" className="mb-10 rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow-xl">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="mb-6 grid gap-3 md:grid-cols-2">
              <button
                onClick={() => setMode("predict")}
                className={`rounded-xl border px-5 py-4 text-left transition ${mode === "predict"
                  ? "border-white bg-white text-neutral-950"
                  : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:border-neutral-500"
                  }`}
              >
                <p className="font-semibold">Predict Rankings</p>
                <p className="mt-1 text-sm opacity-75">
                  Upload a JSON file and generate predicted photo rankings for each scene.
                </p>
              </button>

              <button
                onClick={() => setMode("evaluate")}
                className={`rounded-xl border px-5 py-4 text-left transition ${mode === "evaluate"
                  ? "border-white bg-white text-neutral-950"
                  : "border-neutral-700 bg-neutral-950 text-neutral-300 hover:border-neutral-500"
                  }`}
              >
                <p className="font-semibold">Evaluate Labeled JSON</p>
                <p className="mt-1 text-sm opacity-75">
                  If the JSON contains ground-truth ranks, calculate ranking metrics.
                </p>
              </button>
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium text-neutral-300">
                Upload JSON file
              </label>

              <input
                type="file"
                accept=".json,application/json"
                onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null);
                  setError("");
                }}
                className="block w-full cursor-pointer rounded-lg border border-neutral-700 bg-neutral-950 text-sm text-neutral-300 file:mr-4 file:border-0 file:bg-neutral-100 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-neutral-950 hover:file:bg-white"
              />

              {file && (
                <p className="mt-2 text-sm text-neutral-400">
                  Selected: {file.name}
                </p>
              )}
            </div>

            <button
              onClick={handleEvaluate}
              disabled={loading}
              className="rounded-xl bg-white px-6 py-3 font-semibold text-neutral-950 transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading
                ? mode === "evaluate"
                  ? "Evaluating..."
                  : "Predicting..."
                : mode === "evaluate"
                  ? "Evaluate JSON"
                  : "Predict Rankings"}
            </button>
          </div>

          {error && (
            <div className="mt-4 rounded-lg border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-200">
              {error}
            </div>
          )}
        </section>

        {result && (
          <>
            <section className="mb-10">
              <h2 className="mb-4 text-2xl font-semibold">Upload Summary</h2>

              <div className="grid gap-4 md:grid-cols-4">
                <MetricCard
                  label="Total Images"
                  value={result.summary.total_images.toString()}
                />
                <MetricCard
                  label="Total Scenes"
                  value={result.summary.total_scenes.toString()}
                />
                <MetricCard
                  label="Ranked Images"
                  value={result.summary.ranked_images.toString()}
                />
                <MetricCard
                  label="Unranked Images"
                  value={result.summary.unranked_images.toString()}
                />
              </div>
            </section>
            {mode === "evaluate" && (
              <>
                <section className="mb-4">
                  <h2 className="mb-2 text-2xl font-semibold">Evaluation Summary</h2>
                  <p className="max-w-4xl text-sm text-neutral-400">
                    Metrics are calculated only on scenes with valid ground-truth ranks. Best in
                    Top 3 is only evaluated on scenes with more than 3 ranked candidates.
                  </p>
                </section>

                <section className="mb-10 grid gap-4 md:grid-cols-3 lg:grid-cols-6">
                  <MetricCard
                    label="Evaluated Scenes"
                    value={result.summary.n_scenes_evaluated.toString()}
                  />
                  <MetricCard
                    label="Pairwise Acc"
                    value={formatPercent(result.summary.mean_pairwise_acc)}
                  />
                  <MetricCard
                    label="Top-1 Acc"
                    value={formatPercent(result.summary.top1_acc)}
                  />
                  <MetricCard
                    label="Best in Top 3"
                    value={formatPercent(result.summary.top3_contains_best ?? null)}
                    note={
                      result.summary.top3_eligible_scenes !== undefined
                        ? `${result.summary.top3_eligible_scenes} eligible scenes with n > 3`
                        : "Only for scenes with n > 3"
                    }
                  />
                  <MetricCard
                    label="Spearman"
                    value={formatNumber(result.summary.mean_spearman)}
                  />
                  <MetricCard
                    label="Kendall"
                    value={formatNumber(result.summary.mean_kendall)}
                  />
                </section>

                <section className="mb-10 rounded-2xl border border-neutral-800 bg-neutral-900 p-6">
                  <h2 className="mb-4 text-2xl font-semibold">Scene Metrics</h2>

                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-left text-sm">
                      <thead className="border-b border-neutral-800 text-neutral-400">
                        <tr>
                          <th className="py-3 pr-4">Scene</th>
                          <th className="py-3 pr-4">Images</th>
                          <th className="py-3 pr-4">Pairwise Acc</th>
                          <th className="py-3 pr-4">Top-1</th>
                          <th className="py-3 pr-4">Spearman</th>
                          <th className="py-3 pr-4">Kendall</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.scene_metrics.slice(0, 30).map((scene) => (
                          <tr
                            key={`${scene.shoot_id}-${scene.scene_id}`}
                            className="border-b border-neutral-800/60"
                          >
                            <td className="py-3 pr-4 font-medium">
                              {scene.scene_id}
                            </td>
                            <td className="py-3 pr-4">{scene.n_images}</td>
                            <td className="py-3 pr-4">
                              {formatPercent(scene.pairwise_acc)}
                            </td>
                            <td className="py-3 pr-4">
                              {scene.top1_acc === 1 ? "Correct" : "Wrong"}
                            </td>
                            <td className="py-3 pr-4">
                              {formatNumber(scene.spearman)}
                            </td>
                            <td className="py-3 pr-4">
                              {formatNumber(scene.kendall)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="mt-4 text-sm text-neutral-500">
                    Showing first 30 scenes.
                  </p>
                </section>
              </>
            )}

            <section className="rounded-2xl border border-neutral-800 bg-neutral-900 p-6">
              <h2 className="mb-4 text-2xl font-semibold">
                Predicted Rankings
              </h2>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left text-sm">
                  <thead className="border-b border-neutral-800 text-neutral-400">
                    <tr>
                      <th className="py-3 pr-4">Scene</th>
                      <th className="py-3 pr-4">Image ID</th>
                      {mode === "evaluate" && (
                        <th className="py-3 pr-4">True Rank</th>
                      )}
                      <th className="py-3 pr-4">Predicted Position</th>
                      <th className="py-3 pr-4">Score</th>
                    </tr>
                  </thead>

                  <tbody>
                    {result.predictions
                      .filter((pred) => mode === "predict" || pred.true_rank !== null)
                      .slice(0, 50)
                      .map((pred, index) => (
                        <tr
                          key={`${pred.scene_id}-${pred.image_id}-${index}`}
                          className="border-b border-neutral-800/60"
                        >
                          <td className="py-3 pr-4 font-medium">
                            {pred.scene_id}
                          </td>
                          <td className="max-w-xs truncate py-3 pr-4 text-neutral-400">
                            {pred.image_id}
                          </td>
                          {mode === "evaluate" && (
                            <td className="py-3 pr-4">
                              {pred.true_rank ?? "N/A"}
                            </td>
                          )}
                          <td className="py-3 pr-4">
                            {pred.predicted_position}
                          </td>
                          <td className="py-3 pr-4">
                            {pred.pred_score.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-4 text-sm text-neutral-500">
                Showing first 50 predictions.
              </p>
            </section>
          </>
        )}
      </div>
      <section
        id="readme"
        className="mt-16 rounded-2xl border border-neutral-800 bg-neutral-900 p-8"
      >
        <p className="mb-3 text-sm uppercase tracking-[0.3em] text-neutral-500">
          Project README
        </p>

        <h2 className="mb-6 text-3xl font-bold">Wedding Photo Ranking System</h2>

        <div className="space-y-8 text-neutral-300">
          <ReadmeBlock title="Overview">
            <p>
              This project builds a system to predict within-scene wedding photo
              rankings from face detection metadata. Given a face_detections.json file,
              the system extracts image-level features, applies a trained pairwise
              ranking model, and returns predicted rankings for each scene. The final decision is based on
              the trade-off between dataset size, model performance, computational efficiency, and latency sensitivity.
            </p>
            <p>
              After multiple rounds of experimental comparisons, I ultimately selected the <strong>Tuned Pairwise GradientBoostingClassifier</strong> as the final model. The Tuned Pairwise GradientBoostingClassifier performed best on the validation set in terms of <strong>Pairwise Accuracy</strong> and <strong>Top-1 Accuracy</strong>. We believe these two metrics better reflect the model’s ranking capabilities in this scenarios.
              Although LightGBM Ranker has advantages in training efficiency and native support for ranking tasks, its overall performance is slightly inferior to that of pairwise methods given the current dataset size. I made the final selection based on empirical results from the validation and test sets.
              Since the dataset contains a large number of scenes with only <strong>2–3</strong> photos, NDCG has limited discriminative power; therefore, I used Pairwise Accuracy and scene-level ranking correlation metrics (Spearman and Kendall) as the primary evaluation metrics. Feature importance analysis and error case analysis also indicate that the model effectively captures key quality signals such as the midblink probability and focus score of the primary face.
            </p>

            <p>
              The most useful signals were
              focus quality, eye-state probabilities, main-subject bad-eye indicators, and
              worst-case group eye features. I also tested manually weighted composite
              quality penalties, but removed them after ablation showed weaker held-out
              generalization.
            </p>
          </ReadmeBlock>

          <ReadmeBlock title="Problem Definition">
            <p>
              The ranking task is scene-relative: rank 1 is the best image within a
              scene, not globally across the entire wedding shoot. Because of this,
              the final system uses a pairwise ranking approach instead of directly
              predicting raw rank values.
            </p>
          </ReadmeBlock>

          <ReadmeBlock title="Input Data">
            <p>
              The input is a JSON file containing image-level metadata, detected faces,
              bounding boxes, focus scores, eye-state probabilities, subject importance
              scores, pose/orientation values, and group-level metrics.
            </p>
          </ReadmeBlock>

          <ReadmeBlock title="Feature Engineering">
            <ul className="list-disc space-y-2 pl-6">
              <li>Scene, category, face count, selected-face count, and group-pose features.</li>
              <li>Aggregate face quality features such as focus, confidence, face area, and face position.</li>
              <li>Eye-state probability features including open, closed, midblink, covered, glasses, partially-open, and barely-open states.</li>
              <li>Main-subject specific features, especially main face focus, open-eye probability, closed-eye probability, covered-eye probability, and pose.</li>
              <li>Separate main-subject and non-main-subject bad-eye counts and ratios, because main-subject failures are more important for photo culling.</li>
              <li>Robust worst-case features such as maximum main-subject eye penalty, maximum main-subject focus penalty, group minimum open-eye probability, and group maximum closed/covered-eye probability.</li>
              <li>Composite hand-weighted quality penalties were tested but removed after ablation because they improved validation performance less reliably than atomic worst-case features on the held-out test set.</li>
            </ul>
          </ReadmeBlock>

          <ReadmeBlock title="Modeling Approach">
            <p>
              The final model is a pairwise Gradient Boosting classifier. For every
              pair of ranked images in the same scene, the model learns whether image A
              should rank above image B using the feature difference vector A - B.
              At inference time, each image receives a score by summing its predicted
              probability of beating every other image in the same scene.
            </p>
          </ReadmeBlock>

          <ReadmeBlock title="Evaluation Metrics">
            <ul className="list-disc space-y-2 pl-6">
              <li>
                <strong>Pairwise Accuracy:</strong> how often the model orders image pairs correctly.
              </li>
              <li>
                <strong>Top-1 Accuracy:</strong> whether the model selects the same best image as the ground truth.
              </li>
              <li>
                <strong>Best in Top 3:</strong> whether the photographer-selected best image
                appears within the model&apos;s top 3 recommendations. This is only calculated
                for scenes with more than 3 ranked candidates.
              </li>
              <li>
                <strong>Spearman Correlation:</strong> rank-order correlation between predicted and true ordering.
              </li>
              <li>
                <strong>Kendall Correlation:</strong> pairwise rank correlation between predicted and true ordering.
              </li>
            </ul>
          </ReadmeBlock>

          <ReadmeBlock title="Final Results">
            <p>
              The final Pairwise Gradient Boosting model was evaluated on the held-out test
              split. The Top-3 metric is reported only on scenes with more than 3 ranked
              candidates.
            </p>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm">
                <thead className="border-b border-neutral-800 text-neutral-400">
                  <tr>
                    <th className="py-3 pr-4">Model</th>
                    <th className="py-3 pr-4">Pairwise Accuracy</th>
                    <th className="py-3 pr-4">Top-1 Accuracy</th>
                    <th className="py-3 pr-4">Best in Top 3</th>
                    <th className="py-3 pr-4">Spearman</th>
                    <th className="py-3 pr-4">Kendall</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Pairwise Gradient Boosting</td>
                    <td className="py-3 pr-4">82.02%</td>
                    <td className="py-3 pr-4">77.06%</td>
                    <td className="py-3 pr-4">95.00% on 20 eligible scenes</td>
                    <td className="py-3 pr-4">0.6617</td>
                    <td className="py-3 pr-4">0.6405</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </ReadmeBlock>

          <ReadmeBlock title="Experimentation & Iteration">
            <p>
              I treated this as an iterative ranking problem rather than a one-shot model
              selection task. Several baselines and feature variants were tested to
              understand which formulation best matched the scene-relative nature of the
              labels.
            </p>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm">
                <thead className="border-b border-neutral-800 text-neutral-400">
                  <tr>
                    <th className="py-3 pr-4">Experiment</th>
                    <th className="py-3 pr-4">Motivation</th>
                    <th className="py-3 pr-4">Outcome</th>
                    <th className="py-3 pr-4">Decision</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Raw rank regression</td>
                    <td className="py-3 pr-4">
                      Simple baseline that directly predicts the original rank value.
                    </td>
                    <td className="py-3 pr-4">Weak performance because rank scale varies by scene.</td>
                    <td className="py-3 pr-4">Rejected</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Normalized rank regression</td>
                    <td className="py-3 pr-4">
                      Converts rank into a scene-relative quality score.
                    </td>
                    <td className="py-3 pr-4">Large improvement over raw rank regression.</td>
                    <td className="py-3 pr-4">Useful baseline</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Pairwise ranking</td>
                    <td className="py-3 pr-4">
                      Models whether one image should rank above another within the same scene.
                    </td>
                    <td className="py-3 pr-4">Best overall formulation.</td>
                    <td className="py-3 pr-4">Selected</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Main-subject eye/focus features</td>
                    <td className="py-3 pr-4">
                      Main-subject quality should matter more than non-main faces.
                    </td>
                    <td className="py-3 pr-4">Improved held-out ranking performance.</td>
                    <td className="py-3 pr-4">Kept</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Robust worst-case quality features</td>
                    <td className="py-3 pr-4">
                      Added atomic main-subject and group-level quality signals such as maximum
                      eye penalty, maximum focus penalty, group minimum open-eye probability, and
                      group maximum closed/covered-eye probability.
                    </td>
                    <td className="py-3 pr-4">
                      Improved held-out test performance and aligned well with wedding photo
                      culling failure modes.
                    </td>
                    <td className="py-3 pr-4">Kept</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Composite quality penalty</td>
                    <td className="py-3 pr-4">
                      Tested a hand-weighted quality score combining focus, eye-state, and pose
                      penalties.
                    </td>
                    <td className="py-3 pr-4">
                      Validation performance was less reliable on the held-out test set, suggesting
                      the hand-tuned composite score introduced overfitting.
                    </td>
                    <td className="py-3 pr-4">Rejected</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Enhanced pair representation</td>
                    <td className="py-3 pr-4">
                      Tested richer pair features such as A-B, absolute difference, A, and B.
                    </td>
                    <td className="py-3 pr-4">Did not improve held-out performance.</td>
                    <td className="py-3 pr-4">Rejected</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">LightGBM ranker</td>
                    <td className="py-3 pr-4">
                      Tested a dedicated learning-to-rank model.
                    </td>
                    <td className="py-3 pr-4">Slightly underperformed the pairwise Gradient Boosting model. Even the computational complexity is ligher.</td>
                    <td className="py-3 pr-4">Rejected</td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">Composition features</td>
                    <td className="py-3 pr-4">
                      Added center distance, face-size heuristics, and front-facing ratios.
                    </td>
                    <td className="py-3 pr-4">
                      Did not improve generalization, likely because composition preference is scene-dependent.
                    </td>
                    <td className="py-3 pr-4">Rejected</td>
                  </tr>

                  <tr>
                    <td className="py-3 pr-4 font-medium">Hyperparameter tuning</td>
                    <td className="py-3 pr-4">
                      Searched Gradient Boosting settings using validation performance.
                    </td>
                    <td className="py-3 pr-4">
                      Validation improved slightly, but test performance did not improve.
                    </td>
                    <td className="py-3 pr-4">Default model retained</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </ReadmeBlock>

          <ReadmeBlock title="Rank Gap Analysis">
            <p>
              Pairwise accuracy improved as the true rank gap became larger. This means
              the model was much better at separating clearly different images, while
              adjacent-rank comparisons such as rank 1 vs rank 2 remained the hardest.
            </p>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm">
                <thead className="border-b border-neutral-800 text-neutral-400">
                  <tr>
                    <th className="py-3 pr-4">True Rank Gap</th>
                    <th className="py-3 pr-4">Example Pair Comparisons</th>
                    <th className="py-3 pr-4">Pairs</th>
                    <th className="py-3 pr-4">Pairwise Accuracy</th>
                    <th className="py-3 pr-4">Interpretation</th>
                  </tr>
                </thead>

                <tbody>
                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">1</td>
                    <td className="py-3 pr-4">Rank 1 vs 2, 2 vs 3, 3 vs 4</td>
                    <td className="py-3 pr-4">189</td>
                    <td className="py-3 pr-4">78.31%</td>
                    <td className="py-3 pr-4">
                      Hardest bucket; adjacent ranks are often visually similar or
                      subjective.
                    </td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">2</td>
                    <td className="py-3 pr-4">Rank 1 vs 3, 2 vs 4, 3 vs 5</td>
                    <td className="py-3 pr-4">80</td>
                    <td className="py-3 pr-4">83.75%</td>
                    <td className="py-3 pr-4">
                      Easier than adjacent-rank comparisons as quality differences become
                      clearer.
                    </td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">3</td>
                    <td className="py-3 pr-4">Rank 1 vs 4, 2 vs 5, 3 vs 6</td>
                    <td className="py-3 pr-4">35</td>
                    <td className="py-3 pr-4">91.43%</td>
                    <td className="py-3 pr-4">
                      Clearer separation between stronger and weaker images.
                    </td>
                  </tr>

                  <tr className="border-b border-neutral-800/60">
                    <td className="py-3 pr-4 font-medium">4</td>
                    <td className="py-3 pr-4">Rank 1 vs 5, 2 vs 6, 3 vs 7</td>
                    <td className="py-3 pr-4">15</td>
                    <td className="py-3 pr-4">93.33%</td>
                    <td className="py-3 pr-4">
                      Usually much easier to separate because the rank difference is
                      larger.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </ReadmeBlock>

          <ReadmeBlock title="Error Analysis">
            <p>
              The model performs best when there are clear differences in focus, eye
              state, or main-subject quality. It struggles more on adjacent-rank pairs,
              where the difference between rank 1 and rank 2 may be subjective or
              depend on visual information not present in the metadata.
            </p>

            <p>
              I found that feature engineering had to be validated rather than assumed. Some manually designed features, such as composite quality penalties, reduced held-out performance and were removed. In contrast, threshold-based warning features for common failure modes such as closed eyes, covered eyes, and poor main-subject quality improved test performance when used alongside continuous focus and eye-state probabilities.

              This suggests that the model benefits from a combination of continuous metadata signals and lightweight domain-informed warning features, while overly rigid hand-composed scoring rules can hurt generalization.
            </p>
          </ReadmeBlock>

          <ReadmeBlock title="How to Use This Demo">
            <ol className="list-decimal space-y-2 pl-6">
              <li>Choose Predict Rankings to rank all images in the uploaded JSON file.</li>
              <li>Choose Evaluate Labeled JSON if the file contains valid ground-truth ranks.</li>
              <li>Upload a face_detections.json file.</li>
              <li>Review the predicted rankings and scene-level evaluation metrics.</li>
            </ol>
          </ReadmeBlock>
        </div>
      </section>
    </main>
  );
}

function MetricCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-neutral-900 p-5">
      <p className="mb-2 text-sm text-neutral-400">{label}</p>
      <p className="text-2xl font-bold">{value}</p>
      {note && <p className="mt-2 text-xs text-neutral-500">{note}</p>}
    </div>
  );
}

function ReadmeBlock({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section>
      <h3 className="mb-3 text-xl font-semibold text-neutral-100">{title}</h3>
      <div className="leading-7 text-neutral-300">{children}</div>
    </section>
  );
}
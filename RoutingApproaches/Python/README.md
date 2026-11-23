# Routing approaches (Python)

Task: given an IRC chat, score from 0 to 1 whether a candidate (someone who spoke in the last 20 lines, other
than the speaker) replies to the latest message. Data is the irc-disentanglement dataset in `../../data/irc-data`.

## Deliverables

| File | What it is |
|------|------------|
| `ml_train.py` | Trains the Keras model: hyperparameters, checkpoints (`weights/checkpoints/`), final weights and the training curve (`weights/ml_training_curve.png`). The features and layers are in `machine_learning.py` |
| `ml_inference.py` | Loads `weights/ml.weights.h5` and scores `test_sample.txt`. `--eval` writes `ml_eval.csv` |
| `weights/ml.weights.h5` | The trained weights (features only, so no token map is needed) |
| `llm_harness.py` | Every prompt (`PROMPTS`) and LLM (`MODELS`) tried, and the prompt log metrics. Responses are saved in `llm/runs/` |
| `llm_inference.py` | The best prompt (v0) on the best LLM (gemma4:e4b) for `test_sample.txt`. `--eval` writes `llm_eval.csv` |
| `eval.py` | Every number in the results table, from the five CSVs, on the full test set and the human evaluation subset |
| `ground_truth.csv` | The full test set: lines 1000-1499 of the 10 test logs, and who replied to each |
| `human_eval.csv` | Human confidences that stig_ replies, for lines 1000-1049 of 2005-07-06_14 |
| `symbolic_eval.csv`, `ml_eval.csv`, `llm_eval.csv` | One row per test point (30,221) with each approach's score. Written with `--eval` by `evaluate_symbolic.py`, `ml_inference.py` and `llm_inference.py` |
| `test_sample.txt` | The test set sample, scored for crimsun by `symbolic.py`, `ml_inference.py` and `llm_inference.py` |

## Other files

- `machine_learning.py`: the ML model's features, architecture and data formatting
- `symbolic.py`: the symbolic approach
- `util.py`: loading the dataset into points
- `feature_sweep.py`, `layer_sweep.py`: hyperparameter sweeps, written up in `models/models.md` with every model trained
- `evaluate_machine_learning.py`, `evaluate_symbolic.py`: metrics on the dev split
- `history_viewer.py`, `architecture_diagram.py`: plots

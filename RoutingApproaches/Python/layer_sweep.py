import json
from pathlib import Path

import numpy as np

import machine_learning as ml
from feature_sweep import formatted_data, history_json, run, setup_model, summary
from machine_learning import FEATURE_COUNT, USE_TEXT

"""
Trains the model with 4 down to 1 hidden dense layers, on all FEATURE_COUNT features, and scores
each one on the validation split. Layers are removed from the current (64, 32) + (32, 16) setup.

Reuses the data cached by feature_sweep.py. Each run is saved to out_dir/<n>_layers/ with its
weights, history and token map. To load one, pass its feature_layers/head_layers to get_model_architecture.

Usage: python layer_sweep.py [out_dir] [data_cache] [--no-text]
    --no-text  trains without the text branch (USE_TEXT = False). The cache's text input is dropped
"""

# hidden layers -> (feature_layers, head_layers)
CONFIGS = {
    4: ((64, 32), (32, 16)),
    3: ((64,),    (32, 16)),
    2: ((64,),    (32,)),
    1: ((),       (32,)),
}

if __name__ == "__main__":
    from sys import argv

    text = "--no-text" not in argv
    argv = [a for a in argv if a != "--no-text"]

    # get_model_architecture reads this at build time
    ml.USE_TEXT = text

    out_dir = Path(argv[1] if len(argv) > 1 else "models/layer_sweep")
    cache   = Path(argv[2] if len(argv) > 2 else f"models/feature_sweep/{'data_text' if USE_TEXT else 'data'}.npz")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    results      = json.loads(results_path.read_text()) if results_path.exists() else {}

    model = setup_model()
    data  = formatted_data(model, cache)

    if not text:
        data = ({"features": data[0]["features"]}, data[1], {"features": data[2]["features"]}, data[3])

    print(f"Text branch {'on' if text else 'off'}, inputs {sorted(data[0])}")
    print(f"Train points {len(data[1])}, valid points {len(data[3])}, label imbalance {np.mean(data[1]):.4f}")

    for layers, (feature_layers, head_layers) in CONFIGS.items():
        if str(layers) in results:
            print(f"=== Skipping {layers} layers (already done) ===")
            continue

        print(f"=== {layers} layers: features {feature_layers}, head {head_layers} ===")
        result, trained, history = run(model, data, FEATURE_COUNT, feature_layers=feature_layers, head_layers=head_layers)
        results[str(layers)] = {"feature_layers": list(feature_layers), "head_layers": list(head_layers), **result}

        run_dir = out_dir / f"{layers}_layers"
        run_dir.mkdir(exist_ok=True)
        trained.save_weights(run_dir / "routing.weights.h5")
        (run_dir / "history.json").write_text(history_json(history))
        model.save_token_map(str(run_dir / "token_map.json"))
        results_path.write_text(json.dumps(results, indent=2))

        print(f"=== {layers} layers: {summary(result)} ===")

import itertools, json, math, random, subprocess, shutil, time
from pathlib import Path

BASE_SCRIPT = "train_siglip_heads_enhanced.py"
BASE_OUTPUT = Path("outputs/hparam_runs")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

N_RANDOM = 30            # how many random samples
FINALISTS = 5            # top-k to retrain longer
SEED = 42
random.seed(SEED)

search_space = {
    "lr":        (5e-4, 5e-3),       # log uniform
    "dropout":   [0.0, 0.1, 0.2],
    "hidden_mult": [0.5, 1.0, 1.5],
    "proj_dim":  [256, 512],
    "label_smoothing": [0.0, 0.05, 0.1],
    "warmup_frac": [0.0, 0.05],
    "use_pos_weight_balancing": [False, True]
}

def sample_point():
    def log_uniform(lo, hi):
        import math, random
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    return {
        "lr": log_uniform(*search_space["lr"]),
        "dropout": random.choice(search_space["dropout"]),
        "hidden_mult": random.choice(search_space["hidden_mult"]),
        "proj_dim": random.choice(search_space["proj_dim"]),
        "label_smoothing": random.choice(search_space["label_smoothing"]),
        "warmup_frac": random.choice(search_space["warmup_frac"]),
        "use_pos_weight_balancing": random.choice(search_space["use_pos_weight_balancing"]),
    }

def run_trial(idx, params, epochs=20):
    run_name = f"trial_{idx:03d}"
    out_dir = BASE_OUTPUT / run_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    args = [
        "python", BASE_SCRIPT,
        "--image_embeddings", "outputs/image_embeddings.pt",
        "--text_embeddings", "outputs/text_embeddings.pt",
        "--output_dir", str(BASE_OUTPUT),
        "--run_name", run_name,
        "--epochs", str(epochs),
        "--batch_size", "64",
        "--proj_dim", str(params["proj_dim"]),
        "--hidden_mult", str(params["hidden_mult"]),
        "--dropout", str(params["dropout"]),
        "--lr", f"{params['lr']:.6f}",
        "--label_smoothing", str(params["label_smoothing"]),
        "--warmup_frac", str(params["warmup_frac"]),
        "--log_interval", "1",
        "--save_interval", str(epochs)  # only final save
    ]
    if params["use_pos_weight_balancing"]:
        args.append("--use_pos_weight_balancing")
    # Mixed precision can speed up:
    args.append("--amp")
    print("Launching:", " ".join(args))
    start = time.time()
    subprocess.run(args, check=True)
    dur = time.time() - start

    metrics_path = out_dir / f"retrieval_metrics_epoch_{epochs}.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {}
    metrics["duration_sec"] = dur
    metrics["params"] = params
    metrics["run_name"] = run_name

    # composite score
    loss = metrics.get("avg_train_loss", float("inf"))
    mean_r10 = metrics.get("mean_R@10", 0.0)
    alpha = 0.1
    metrics["score"] = mean_r10 - alpha * loss
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    return metrics

results = []
for i in range(N_RANDOM):
    params = sample_point()
    try:
        metrics = run_trial(i, params, epochs=20)
        print(f"[{i}] score={metrics['score']:.4f} mean_R@10={metrics.get('mean_R@10')} loss={metrics.get('avg_train_loss')}")
        results.append(metrics)
    except subprocess.CalledProcessError as e:
        print(f"Trial {i} failed:", e)

# Rank trials
results_sorted = sorted(results, key=lambda m: m["score"], reverse=True)
(Path("outputs/hparam_runs/summary_all.json")
 .write_text(json.dumps(results_sorted, indent=2)))

print("Top 5:")
for r in results_sorted[:5]:
    print(r["run_name"], r["score"], r["params"])

# Retrain finalists longer
final_models = []
for j, r in enumerate(results_sorted[:FINALISTS]):
    params = r["params"]
    long_epochs = 40
    run_name = r["run_name"] + "_long"
    params_copy = dict(params)
    metrics_long = run_trial(1000 + j, params_copy, epochs=long_epochs)
    final_models.append(metrics_long)

(Path("outputs/hparam_runs/finalists.json")
 .write_text(json.dumps(final_models, indent=2)))

# Pick global best
all_candidates = results_sorted + final_models
best = sorted(all_candidates, key=lambda m: m["score"], reverse=True)[0]
print("BEST MODEL:", best["run_name"], best["score"], best["params"])

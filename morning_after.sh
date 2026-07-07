#!/usr/bin/env bash
# Run this after the overnight GraphSAGE training (runs/graphsage_scarce) finishes.
# Re-evaluates on the test set, regenerates the training-evolution GIF, and
# points app.py + predict.py at the new best checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=== checking training finished ==="
if pgrep -f "src.train --mode real --model graphsage" > /dev/null; then
    echo "GraphSAGE training is still running -- wait for it to finish before running this script."
    exit 1
fi
tail -5 runs/graphsage_scarce/losses.csv

echo "=== Step 6: force-based evaluation on test set ==="
python3 -m src.evaluate --checkpoint checkpoints/graphsage_scarce_best.pt --out checkpoints/eval_results_graphsage.json
python3 - <<'PYEOF'
import json
mlp = json.load(open("checkpoints/eval_results_mlp.json"))
gs = json.load(open("checkpoints/eval_results_graphsage.json"))
print("\n=== MLP vs GraphSAGE (test set) ===")
for k in ["cl_rel_err", "cl_spearman", "cd_rel_err", "cd_spearman"]:
    print(f"{k:15s}  mlp={mlp[k]:.4f}   graphsage={gs[k]:.4f}")
PYEOF

echo "=== Step 8.1: regenerate training-evolution GIF for GraphSAGE ==="
python3 scripts/make_evolution_gif.py --run-name graphsage_scarce --fps 15 --out plots/training_evolution_graphsage.gif

echo "=== pointing app.py / predict.py at the new checkpoint ==="
sed -i.bak 's/CHECKPOINT = ".*"/CHECKPOINT = "checkpoints\/graphsage_scarce_best.pt"/' src/app_core.py
rm -f src/app_core.py.bak
sed -i.bak 's/default="checkpoints\/mlp_scarce_best.pt"/default="checkpoints\/graphsage_scarce_best.pt"/' predict.py
rm -f predict.py.bak

echo "=== done. Update PROGRESS.md with the comparison numbers above, then: ==="
echo "  git add -A && git commit -m 'Step 9: GraphSAGE results, switch default checkpoint'"
echo "  streamlit run app.py   # now serving GraphSAGE predictions"

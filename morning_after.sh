#!/usr/bin/env bash
# DEPRECATED (2026-07-11): this was written for the original Step 9 overnight
# GraphSAGE run on the `scarce` task, before the M4 Pro scale-up work. Both
# scale-up training runs (mlp_full_fullres_v4, graphsage_full_64k) have long
# since finished, and the current default checkpoint in src/app_core.py /
# predict.py (the full-resolution MLP -- deliberately not the higher-scoring
# GraphSAGE, see PROGRESS.md 2026-07-11 "multi-page website" entry for why)
# reflects a considered decision, not a stale default.
#
# The old version of this script below used `sed` to blindly overwrite that
# checkpoint back to graphsage_scarce_best.pt, which would silently undo that
# decision if run today. Refusing to run rather than risk that.
#
# If you need to re-evaluate a checkpoint or regenerate an evolution GIF, use
# the underlying tools directly instead:
#   python3 -m src.evaluate --checkpoint checkpoints/<name>_best.pt --out checkpoints/eval_results_<name>.json
#   python3 scripts/make_evolution_gif.py --run-name <name> --fps 15 --out plots/training_evolution_<name>.gif
# If you need to (re)train + auto-resume on crash, use scripts/train_supervised.sh.
echo "morning_after.sh is deprecated and refuses to run -- see the comment at the top of this file." >&2
exit 1

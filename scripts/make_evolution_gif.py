"""Stitch plots/evolution/<run>/*.png into an animated GIF (Step 8.1)."""
import argparse
import glob
import os

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--run-name", default="mlp_scarce")
p.add_argument("--fps", type=float, default=15)
p.add_argument("--out", default="plots/training_evolution.gif")
args = p.parse_args()

frame_paths = sorted(glob.glob(f"plots/evolution/{args.run_name}/epoch_*.png"))
print(f"found {len(frame_paths)} frames")
assert len(frame_paths) > 1, "need at least 2 frames to make a GIF"

frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
duration_ms = int(1000 / args.fps)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
print(f"saved {args.out} ({len(frames)} frames @ {args.fps} fps)")

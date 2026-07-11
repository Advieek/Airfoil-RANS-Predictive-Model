"""Export the trained MLP to ONNX for Netron visualization (Step 8.2).
MPS export is flaky per BUILD_SPEC -- move to CPU first."""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.models import build_model

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="checkpoints/mlp_full_fullres_v4_best.pt")
p.add_argument("--out", default="checkpoints/model.onnx")
args = p.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model_kwargs = ckpt.get("model_kwargs", {"in_dim": 7, "out_dim": 4})
model = build_model(ckpt["model_name"], **model_kwargs)
model.load_state_dict(ckpt["model_state"])
model.to("cpu").eval()
print(f"exporting {ckpt['model_name']} {model_kwargs} from {args.checkpoint}")

dummy = torch.randn(1000, 7)
torch.onnx.export(
    model,
    dummy,
    args.out,
    input_names=["point_features"],
    output_names=["predictions"],
    dynamic_axes={"point_features": {0: "n_points"}, "predictions": {0: "n_points"}},
    opset_version=18,
    external_data=False,  # single self-contained .onnx file -- simpler for static/web hosting
)
print(f"exported {args.out}")

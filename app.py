"""Streamlit live demo: pick an airfoil, drag AoA/Re sliders, watch the
predicted flow field and Cl/Cd update in real time (Step 8.3).

Performance: geometry (shapely point-cloud generation) is cached per airfoil
via @st.cache_data; the model is cached via @st.cache_resource. Slider moves
only rebuild the 2-column inlet-velocity block (src.geometry.apply_inflow)
and run one forward pass -- no geometry recomputation.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.app_core import CHECKPOINT, load_geometry, predict_and_integrate
from src.evaluate import load_model_from_checkpoint
from src.geometry import check_envelope

st.set_page_config(page_title="Airfoil RANS Surrogate", layout="wide")


@st.cache_resource
def get_model():
    return load_model_from_checkpoint(CHECKPOINT, device="cpu")


@st.cache_data
def get_geometry(airfoil_key, dat_bytes=None):
    return load_geometry(airfoil_key, dat_bytes=dat_bytes)


st.title("Airfoil RANS Surrogate — live demo")

with st.sidebar:
    st.header("Airfoil")
    source = st.radio("Source", ["NACA 4-digit", "Upload .dat"])
    dat_bytes = None
    if source == "NACA 4-digit":
        naca_code = st.text_input("NACA code", value="2412", max_chars=4)
        airfoil_key = naca_code
    else:
        uploaded = st.file_uploader("Selig .dat file", type=["dat", "txt"])
        dat_bytes = uploaded.getvalue() if uploaded is not None else None
        airfoil_key = uploaded.name if uploaded is not None else None

    st.header("Flow conditions")
    aoa = st.slider("Angle of attack (deg)", -5.0, 15.0, 5.0, step=0.5)
    reynolds = st.slider("Reynolds number", 2_000_000, 6_000_000, 4_000_000, step=100_000)

if airfoil_key is None:
    st.info("Upload a .dat file to begin.")
    st.stop()

for w in check_envelope(reynolds, aoa):
    st.warning(w)

model, stats, ckpt = get_model()
geo = get_geometry(airfoil_key, dat_bytes=dat_bytes)
cloud, pred, cd, cl = predict_and_integrate(geo, reynolds, aoa, model, stats)

col1, col2 = st.columns(2)
col1.metric("Predicted Cl", f"{cl:.3f}")
col2.metric("Predicted Cd", f"{cd:.4f}")

field = st.radio("Field", ["pressure", "velocity magnitude"], horizontal=True)
if field == "pressure":
    values = pred[:, 2]
    label = "pressure"
else:
    values = np.linalg.norm(pred[:, :2], axis=1)
    label = "|velocity|"

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(cloud["position"][:, 0], cloud["position"][:, 1], c=values, s=3, cmap="viridis")
ax.set_aspect("equal")
ax.set_title(f"predicted {label} — {airfoil_key}, Re={reynolds:.2e}, AoA={aoa} deg")
fig.colorbar(sc, ax=ax)
st.pyplot(fig)

st.caption(
    "Model: per-point MLP trained on AirfRANS 'scarce' task. "
    "Cd is known to be unreliable for this baseline (see PROGRESS.md) -- "
    "context-free per-point predictions can't resolve the near-wall gradients drag depends on."
)

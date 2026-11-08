"""Build model inputs from data/nhanes_week.npz -> data/inputs.npz (CNN channels C, rhythm features F)."""
import sys; sys.path.insert(0, "src")
import numpy as np
from wearcnn_guard.features import cnn_input, rhythm_features
d = np.load("data/nhanes_week.npz")
C = cnn_input(d["X"], d["M"]); F = rhythm_features(d["X"], d["M"])
np.savez_compressed("data/inputs.npz", C=C, F=F)
print("C", C.shape, "F", F.shape)

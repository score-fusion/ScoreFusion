import numpy as np
import matplotlib.pyplot as plt

# ===== load npz =====
data = np.load("./cache/1_149146_flair.npz")
vol = data[data.files[0]]
# import pdb;pdb.set_trace()

# handle shape (1, 192, 192) -> (192, 192)
if vol.ndim == 3:
    vol = vol[0]

# normalize
vol = vol.astype(np.float32)
vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

slice_img = vol

# save png (no show)
plt.imsave("flair894_slice_mid.png", slice_img, cmap="gray")

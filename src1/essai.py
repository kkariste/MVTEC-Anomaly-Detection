#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:29:02 2025

@author: lechat
"""

#%%
from skimage.util import view_as_windows

def generate_anomaly_map1(image, lbp_transformer, ocsvm_model, upscale=True):
    # 1. Convertir en NDG
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Image LBP
    lbp = local_binary_pattern(
        image,
        lbp_transformer.n_points,
        lbp_transformer.radius,
        lbp_transformer.method
    )

    # 3. Params
    patch_size = lbp_transformer.patch_size
    step = lbp_transformer.step or patch_size[0]
    n_bins = lbp_transformer.n_points + 2 if lbp_transformer.method == 'uniform' else 2**lbp_transformer.n_points

    # 4. Découpage
    patches = view_as_windows(lbp, patch_size, step=step)
    h, w = patches.shape[:2]
    scores = np.zeros((h, w))

    # 5. Score par patch
    for i in range(h):
        for j in range(w):
            patch = patches[i, j]
            hist = np.histogram(patch, bins=n_bins, range=(0, n_bins))[0]
            hist = hist / (hist.sum() + 1e-6)
            score = ocsvm_model.decision_function([hist])[0]
            scores[i, j] = -score

    # 6. Interpolation optionnelle
    if upscale:
        target_size = (image.shape[1], image.shape[0])  # (width, height)
        scores = cv2.resize(scores.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)

    # Normalisation [0,1], PB si tous les scores sont identiques
    scores -= scores.min()
    if scores.max() > 0:  # Éviter division par zéro
        scores /= scores.max()

    return scores




#%%
import matplotlib.pyplot as plt

from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure
import skimage
import cv2
path = "../data/tile/test/glue_strip/000.png"
path1 = "../data/grid/train/good/000.png"
image= skimage.io.imread(path, as_gray=True)
print(image.dtype)

#image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

plt.imshow(lbp, cmap='gray')
plt.show()

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(32, 32),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
)

print(fd.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

#%%
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Patch uniforme 3x3
patch = np.array([
    [100, 100, 100],
    [100, 100, 100],
    [100, 100, 100]
], dtype=np.uint8)

# Calcul du LBP avec method='uniform'
lbp_result = local_binary_pattern(patch, P=8, R=1, method='uniform')
#hist, _ = np.hist(lbp_result, range=(0, 10))
print(lbp_result)

fig = plt.figure(figsize=(10, 4))

# Affichage LBP image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(lbp_result, cmap='gray')
ax1.set_title("LBP (uniform)")
ax1.axis('off')

# Histogramme
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(lbp_result.ravel(), bins=np.arange(11) - 0.5, range=(0, 10), edgecolor='black')
ax2.set_title("Histogramme des valeurs LBP")
ax2.set_xlabel("Valeur LBP")
ax2.set_ylabel("Fréquence")
ax2.set_xticks(range(10))

plt.tight_layout()
plt.show()
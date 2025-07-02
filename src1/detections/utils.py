import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math
from PIL import Image
from datasets.mvtecad import DatasetSplit, MVTecDataset

def set_seed(SEED_VALUE: int = 42):
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

def afficher_images_test(dataset: MVTecDataset) -> None:
    """
    Affiche une image et son masque pour chaque type d'anomalie, 
    en utilisant directement les données chargées par le dataset.
    """
    if dataset.split != DatasetSplit.TEST:
        raise ValueError("Cette fonction est conçue pour être utilisée uniquement avec le split TEST.")

    classname = dataset.classnames_to_use[0]

    # Regrouper les indices par type d’anomalie
    anomalies = {}
    for idx, sample in enumerate(dataset.data_to_iterate):
        _, anomaly, _, _ = sample
        if anomaly not in anomalies:
            anomalies[anomaly] = []
        anomalies[anomaly].append(idx)

    n = len(anomalies)
    plt.figure(figsize=(4 * n, 6))

    for i, (anomaly, indices) in enumerate(anomalies.items()):
        idx = random.choice(indices)
        sample = dataset[idx]
        img = sample["image"]
        mask = sample["mask"]

        # Ligne 1 : image
        plt.subplot(2, n, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"{anomaly}")

        # Ligne 2 : masque
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(mask, cmap="gray")
        plt.axis("off")

    plt.suptitle(f"Classe : {classname}", fontsize=16)
    plt.tight_layout()
    plt.show()


def show_train_set(train_set, n_images=4):
    """
    Affiche les n_images premières images et masques du train_set.
    """
    plt.figure(figsize=(4 * n_images, 4))
    for i in range(n_images):
        sample_train = train_set[i]
        
        # Ligne 1 : image
        plt.subplot(2, n_images, i + 1)
        plt.imshow(sample_train["image"], cmap='gray')
        plt.axis("off")
        plt.title(f"{sample_train['image_name']}")
        
        # Ligne 2 : masque
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(sample_train["mask"], cmap='gray')
        plt.axis("off")

    plt.suptitle(f"Classe : {sample_train['classname']}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_patches(patches: np.ndarray):
    """
    Affiche tous les patchs contenus dans un tableau NumPy.
    patches doit avoir la forme (N, H, W) ou (N, H, W, C)
    """
    n_patches = len(patches)
    n_cols = math.ceil(math.sqrt(n_patches))
    n_rows = math.ceil(n_patches / n_cols)

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    for i, ax in enumerate(grid):
        if i >= n_patches:
            ax.axis('off')
            continue
        patch = patches[i]
        if patch.ndim == 2:
            ax.imshow(patch, cmap='gray')
        else:
            ax.imshow(patch)
        ax.axis('off')

    plt.show()



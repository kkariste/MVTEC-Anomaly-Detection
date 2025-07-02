import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
from patchify import patchify
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


method_lbp=['default',
			'ror',
			'uniform',
			'var']


class MVTecDescriptor:
    """
    Classe pour transformer une image, extraire des patchs, 
    calculer des LBP et leurs histogrammes.
    """

    def __init__(self, patch_size: tuple[int, int],  step: int = None) -> None:
        self.patch_size = patch_size
        self.step = step
        
    #@staticmethod
    #def transform(image: np.ndarray , size: tuple[int, int]) -> np.ndarray:
        #image = np.array(image_pil)
        #if len(image.shape) == 3:
            #image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        #image = image / 255.0
        #image = cv.resize(image, size, interpolation=cv.INTER_AREA)
        #return image

    def patchify_image(self, image: np.ndarray, step: int = None) -> np.ndarray:
        if step is None:
            step = self.patch_size[0]
            print(f"Step utilisé : {step}")
        else:
            step = self.step
        return patchify(image, self.patch_size, step=step)

    @staticmethod
    def lpb_histogram(image_patch: np.ndarray, 
                      numPoints: int, 
                      radius: int, 
                      method: str) -> np.ndarray:
        lbp = local_binary_pattern(image_patch, numPoints, radius, method)

        if method == 'uniform':
            n_bins = numPoints + 2
        else:
            n_bins = 2**numPoints

        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist, lbp


    def compute_lbp_and_histograms(self, image: np.ndarray, numPoints: int = 8, radius: int = 1, method: str = 'uniform', step: int = None):
        if len(image.shape) == 3:  # Si RGB (H, W, 3)
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        patches = self.patchify_image(image, step=self.step)
        lbp_patches = []
        histograms = []
        for row in patches:
            lbp_row = []
            for patch in row:
                patch = patch.squeeze()
                hist, lbp = self.lpb_histogram(patch, numPoints, radius, method)
                lbp_row.append(lbp)
                histograms.append(hist)
            lbp_patches.append(lbp_row)
        return np.array(lbp_patches), np.array(histograms)





class MVTecOCSVM:
    """
    Détecteur d'anomalies basé sur One-Class SVM avec descripteurs LBP.
    """
    def __init__(self, kernel: str = 'rbf', gamma: float = 0.01, nu: float = 0.01,
                 patch_size=(32, 32), step: int=None, lbp_radius=1, lbp_points=8, lbp_method='uniform'):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu

        self.patch_size = patch_size
        self.step = step
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_method = lbp_method

        self.descriptor = MVTecDescriptor(patch_size=self.patch_size, step=self.step )
        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)

    def extract_features(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Applique LBP + histogrammes sur chaque image et renvoie un tableau de descripteurs.
        """
        all_histograms = []
        for img in images:
            _, histograms = self.descriptor.compute_lbp_and_histograms(
                img, 
                numPoints=self.lbp_points,
                radius=self.lbp_radius,
                method=self.lbp_method
            )
            all_histograms.extend(histograms)  # Ajouter chaque histogramme de patch
        return np.array(all_histograms)

    def train(self, images_train: list[np.ndarray]) -> None:
        features = self.extract_features(images_train)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        return features
        
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prédit l'anomalie pour tous les patchs d'une image.
        Retourne les prédictions et les scores.
        """
        _, histograms = self.descriptor.compute_lbp_and_histograms(
            image,
            numPoints=self.lbp_points,
            radius=self.lbp_radius,
            method=self.lbp_method
        )
        features_scaled = self.scaler.transform(histograms)
        predictions = self.model.predict(features_scaled)  # +1 (normal) ou -1 (anomalie)
        scores = self.model.decision_function(features_scaled)  # scores de confiance
        return predictions, scores

    
    
    
    def compute_anomaly_map(self, image: np.ndarray, scores: np.ndarray, step: int = None) -> np.ndarray:
        H, W = image.shape
        ph, pw = self.patch_size
        if step is None:
            step = ph

        anomaly_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.uint8)

        n_patches_h = (H - ph) // step + 1
        n_patches_w = (W - pw) // step + 1

        idx = 0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * step
                x = j * step
                value = -scores[idx]  # plus le score est négatif, plus c'est anormal
                anomaly_map[y:y+ph, x:x+pw] += value
                count_map[y:y+ph, x:x+pw] += 1
                idx += 1

        count_map[count_map == 0] = 1
        anomaly_map /= count_map

        # Normalisation [0,1]
        anomaly_map -= anomaly_map.min()
        if anomaly_map.max() > 0:
            anomaly_map /= anomaly_map.max()

        return anomaly_map


        
        
class PatchViewer:
    """
    Classe d'affichage visuel : patchs, LBP, histogrammes
    """

    @staticmethod
    def _plot_in_grid(images: np.ndarray, cmap: str = 'gray'):
        n_images = len(images)
        n_cols = math.ceil(math.sqrt(n_images))
        n_rows = math.ceil(n_images / n_cols)

        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

        for i, ax in enumerate(grid):
            if i >= n_images:
                ax.axis('off')
                continue
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.axis('off')

        plt.show()

    @staticmethod
    def show_patches(patches: np.ndarray):
        if patches.ndim == 4:
            patches = patches.reshape(-1, *patches.shape[2:])
        PatchViewer._plot_in_grid(patches, cmap='gray')

    @staticmethod
    def show_lbp_images(lbp_patches: np.ndarray):
        if lbp_patches.ndim == 4:
            lbp_patches = lbp_patches.reshape(-1, *lbp_patches.shape[2:])
        PatchViewer._plot_in_grid(lbp_patches, cmap='gray')

    """@staticmethod
    def show_histograms(histograms: np.ndarray):
        n = len(histograms)
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
        bins = histograms.shape[1]

        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.3)

        for i, ax in enumerate(grid):
            if i >= n:
                ax.axis('off')
                continue
            ax.bar(np.arange(bins), histograms[i], width=1.0, edgecolor='black')
            ax.set_xlim([0, bins])
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.set_title(f"Hist{i}")
        
        #plt.tight_layout()
        plt.show()"""
        
    @staticmethod
    def _plot_bars_in_grid(data: np.ndarray):
        """
        Affiche une liste de vecteurs (histogrammes) sous forme de barplots en grille.
        data : np.ndarray de forme (n_items, n_bins)
        """
        n = data.shape[0]
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
        bins = data.shape[1]

        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.3)

        for i, ax in enumerate(grid):
            if i >= n:
                ax.axis('off')
                continue
            ax.bar(np.arange(bins), data[i], width=1.0, edgecolor='black')
            ax.set_xlim([0, bins])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Hist {i+1}")

        plt.show()

    # Exemple d'appel pour afficher tes histogrammes
    @staticmethod
    def show_histograms(histograms: np.ndarray):
        PatchViewer._plot_bars_in_grid(histograms)

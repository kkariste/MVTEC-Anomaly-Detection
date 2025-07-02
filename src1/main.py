# %% 
from datasets.mvtecAD import DatasetSplit, MVTecDataset
from detections.utils import afficher_images_test, set_seed, show_train_set
from detections.detector import MVTecDescriptor, PatchViewer,  MVTecOCSVM
from metrics import compute_pixelwise_retrieval_metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# %%
# Set seed for reproductibility
set_seed()

# %%
# Explore one trainset class for MVTec Dataclass
classname = 'carpet'
train_set = MVTecDataset(
        source="../data",
        classname=classname,
        split=DatasetSplit.TRAIN,
        train_val_split=0.8,
        shuffle=True
    )

val_set = MVTecDataset(
    source="../data",
    classname=classname,
    split=DatasetSplit.VAL,       
    train_val_split=0.8,
    shuffle=True
)

n_images = 4
#show_train_set(train_set, n_images)

# %%

STEP = 5

# Compute and plot the first single image trainset patches, LBP and histogram
sample_train = train_set[1]

descriptor = MVTecDescriptor(patch_size=(32, 32), step=STEP)
"""patches_train = descriptor.patchify_image(sample_train['image'], step=STEP)
lbp_patches_train, histograms_train = descriptor.compute_lbp_and_histograms(sample_train['image'])

# Plotting
PatchViewer.show_patches(patches_train)
PatchViewer.show_lbp_images(lbp_patches_train)
PatchViewer.show_histograms(histograms_train)
"""
# %%
# Do the same for the test set
test_set = MVTecDataset(
        source="../data",
        classname=classname,
        split=DatasetSplit.TEST,
        train_val_split=1.0,
        shuffle=False
    )

#print(sample_test['image'].shape)
#Afficher quelques images de test et leur masque
"""afficher_images_test(test_set)

sample_test = test_set[89]

patches_test = descriptor.patchify_image(sample_test['image'], step=STEP)
lbp_patches_test, histograms_test = descriptor.compute_lbp_and_histograms(sample_test['image'])

PatchViewer.show_patches(patches_test)
PatchViewer.show_lbp_images(lbp_patches_test)
PatchViewer.show_histograms(histograms_test)
"""
# %%
images_train = [sample["image"] for sample in train_set]
images_train = [cv.resize(images_train[i], (256, 256), interpolation=cv.INTER_AREA) for i in range (len(images_train))]
images_val = [sample["image"] for sample in val_set]
images_test = [cv.resize(images_val[i], (256, 256), interpolation=cv.INTER_AREA) for i in range (len(images_val))]
ground_truth_masks = [sample["mask"] for sample in val_set]

#%%

# Entraîner le modèle

# Grilles d'hyperparamètres
radii = [1, 2, 3, 4]
points = [8, 11, 16]
nus = [0.001, 0.01, 0.1, 0.15, 0.20, 0.25]
gammas = [0.001, 0.01, 0.1, 0.15, 0.20, 0.25]

# Métrique personnalisée pour OCSVM (à minimiser)
def custom_ocsvm_score(detector, X):
    """Score basé sur la variance des décisions pour données normales (plus stable = meilleur)"""
    _, scores = detector.predict(X)
    return -np.var(scores)  # On veut minimiser la variance des scores normaux

best_score = float('inf')  # On minimise la métrique
best_params = {}

for radius in radii:
    for point in points:
        for nu in nus:
            for gamma in gammas:
                print(f"Test: radius={radius}, points={point}, nu={nu}, gamma={gamma}")
                
                # Initialiser le détecteur
                detector = MVTecOCSVM(
                    patch_size=(32, 32),
                    step=STEP,
                    lbp_radius=radius,
                    lbp_points=point,
                    lbp_method='uniform',
                    gamma=gamma,
                    nu=nu
                )
                
                # Entraînement sur images saines
                detector.train(images_train)

                # Évaluation sur val set (uniquement normal)
                try:
                    current_score = 0
                    for img in images_val:  # images_val doit contenir uniquement des normales
                        _, scores = detector.predict(img)
                        current_score += custom_ocsvm_score(detector, img)  # Score par image
                    
                    mean_score = current_score / len(images_val)

                    if mean_score < best_score:  # On cherche le score minimal
                        best_score = mean_score
                        best_params = {
                            "radius": radius,
                            "points": point,
                            "nu": nu,
                            "gamma": gamma,
                            "score": best_score
                        }
                        print(f"Nouveau meilleur score: {best_score:.4f}")
                
                except Exception as e:
                    print(f"Erreur avec {radius}/{point}/{nu}/{gamma}: {str(e)}")
                    continue

print("\nMeilleurs hyperparamètres trouvés :")
print(best_params)
#%%
best_params = {}
best_params['radius'] = 1
best_params['points'] = 8
best_params['gamma'] = 0.01
best_params['nu'] = 0.15
model = MVTecOCSVM(
    patch_size=(32, 32),
    step=STEP,
    lbp_radius=best_params['radius'],
    lbp_points=best_params['points'],
    lbp_method='uniform',
    gamma=best_params['gamma'],
    nu=best_params['nu']
)

# Entraînement
features = model.train(images_train)

#%%
# Données de test
images_test = [sample['image'] for sample in test_set]
images_test = [cv.resize(images_test[i], (256, 256), interpolation=cv.INTER_AREA) for i in range (len(images_test))]
gt_masks = [sample['mask'] for sample in test_set]
images_test = [cv.resize(gt_masks[i], (256, 256), interpolation=cv.INTER_AREA) for i in range (len(gt_masks))]
# Binarisation des masques ground truth
gt_masks = [(mask > 0).astype(np.uint8) for mask in gt_masks]
anomalies_map = []

#%%
# Prédiction
for img in images_test:
    predictions, scores = model.predict(img)
    anomaly_map = model.compute_anomaly_map(img, scores)
    anomalies_map.append(anomaly_map)
#%%
def show_results(images, anomaly_maps, gt_masks, idx=0):
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(images[idx], cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1,3,2)
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(anomaly_maps[idx], sigma=8)
    plt.imshow(smoothed, cmap='jet', vmin=0, vmax=1)
    plt.title("Carte d'anomalie")
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(gt_masks[idx], cmap='gray')
    plt.title("Masque de vérité")
    plt.axis('off')

    plt.show()

# Exemple pour afficher la première image
show_results(images_test, anomalies_map, gt_masks, idx=10)

#%%

# Évaluation
results = compute_pixelwise_retrieval_metrics(anomalies_map, gt_masks)

print("AUROC:", results["auroc"])
print("Optimal threshold:", results["optimal_threshold"])
print("Optimal FPR:", results["optimal_fpr"])
print("Optimal FNR:", results["optimal_fnr"])
#%%
print(type(gt_masks[0]))
#print(features.shape)
#print(features[:, 0].shape)

"""import math
# features.shape = (n_patches, n_composantes)
n_composantes = features.shape[1]

n_cols = 4
n_rows = math.ceil(n_composantes / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
axes = axes.ravel()

for i in range(n_composantes):
    axes[i].hist(features[:, i], bins=10)
    axes[i].set_title(f"Composante {i}")

plt.tight_layout()
plt.show()

#%%
images_test = [sample["image"] for sample in test_set]
X_lbp_test = detector.train(images_test)
#%%
from sklearn.decomposition import PCA
# PCA pour visualisation en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)



# Applique PCA appris sur les données normales
X_test_pca = pca.transform(X_lbp_test)

# Affichage train (normal) + test (potentiellement anormal)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, label='Normal (train)', color='blue')
#plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.3, label='Test', color='red')
plt.title("Projection PCA - Train vs Test")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()
# %%
predictions, scores = detector.predict(sample_test["image"])
print(sample_test["image_path"])
print(f"Prédictions des patchs : {predictions}")
print(f"Scores : {scores}")
print(np.unique(predictions, return_counts=True))
anomaly_map = detector.compute_anomaly_map(sample_test["image"], predictions)
plt.imshow(anomaly_map, cmap='jet')
plt.colorbar()
plt.title("Carte d'anomalies")
plt.show()"""










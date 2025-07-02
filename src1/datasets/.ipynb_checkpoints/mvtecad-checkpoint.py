import numpy as np
import cv2 as cv
from PIL import Image
import os
from enum import Enum


# Exemple simple d'enum pour les splits
class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


# Exemple : toutes les classes du dataset MVTec
_CLASSNAMES = [
    "bottle", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper"
]


class MVTecDataset:
    """
    Dataset MVTec pour apprentissage classique 
    
    Paramètres :
    - source : chemin vers le dossier racine du dataset
    - classname : nom de la classe (ex : 'bottle')
    - resize : taille des images (côté carré)
    - split : train, test ou val
    - train_val_split : ratio pour séparer train/val
    - shuffle : mélanger les données ?
    - transform : fonction callable pour transformer l'image (ex : redimensionner, normaliser, etc.)
    """

    def __init__(self, 
                 source, 
                 classname,
                 resize=256,
                 split=DatasetSplit.TRAIN, 
                 train_val_split=1.0,
                 shuffle=True,
                 transform=None) -> None: 
        
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname else _CLASSNAMES
        self.train_val_split = train_val_split
        # self.resize = resize
        self.shuffle = shuffle
        self.transform = transform
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image.convert("RGB"))

        if self.split == DatasetSplit.TEST and mask_path:
            mask = Image.open(mask_path)
            mask = self.transform(mask) if self.transform else np.array(mask)
        else:
            # masque vide (zeros) si pas disponible
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.splitext(os.path.basename(image_path))[0],
            "image_path": image_path,
        }


    def get_image_data(self):
        """
        Lit les chemins des images et masques pour chaque classe et anomalie.
        Gère aussi le split train/val si train_val_split < 1.
        """
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            # Utiliser train pour val
            if self.split == DatasetSplit.VAL:
                classpath = os.path.join(self.source, classname, DatasetSplit.TRAIN.value)
            else:
                classpath = os.path.join(self.source, classname, self.split.value)

            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                img_list = [os.path.join(anomaly_path, x) for x in anomaly_files]

                # Split train/val si demandé
                if self.train_val_split < 1.0:
                    split_idx = int(len(img_list) * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        img_list = img_list[:split_idx]
                    elif self.split == DatasetSplit.VAL:
                        img_list = img_list[split_idx:]

                imgpaths_per_class[classname][anomaly] = img_list

                # Préparer les masques (seulement pour TEST et anomalie ≠ good)
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    mask_anomaly_path = os.path.join(maskpath, anomaly)
                    mask_files = sorted(os.listdir(mask_anomaly_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(mask_anomaly_path, x) for x in mask_files
                    ]
                else:
                    maskpaths_per_class[classname][anomaly] = [None] * len(img_list)

                # Créer la liste finale des exemples
                for i, image_path in enumerate(img_list):
                    mask_path = None
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        mask_path = maskpaths_per_class[classname][anomaly][i]
                    data_to_iterate.append([classname, anomaly, image_path, mask_path])

        # Mélanger les données si demandé
        if self.shuffle:
            import random
            random.shuffle(data_to_iterate)

        return imgpaths_per_class, data_to_iterate


class Transform:
    """
    Transformation d'image ou de masque.
    
    - size : taille de sortie (w, h)
    - to_gray : convertit les images RGB en NDG si True (ignoré si l'entrée est déjà en NDG)
    - normalize : divise par 255.0 (pour image seulement ; les masques ne sont jamais normalisés)
    """

    def __init__(self, size=(256, 256), to_gray=False, normalize=True):
        self.size = size
        self.to_gray = to_gray
        self.normalize = normalize

    def __call__(self, input_img: Image.Image or np.ndarray) -> np.ndarray:
        """
        Applique la transformation sur une image PIL ou un masque (PIL ou np.ndarray).
        """
        # Convertir PIL → np.ndarray
        if isinstance(input_img, Image.Image):
            input_img = np.array(input_img)

        # S'il s'agit d'une image couleur
        if len(input_img.shape) == 3:
            if self.to_gray:
                input_img = cv.cvtColor(input_img, cv.COLOR_RGB2GRAY)

            input_img = cv.resize(input_img, self.size, interpolation=cv.INTER_AREA)

            if self.normalize:
                input_img = input_img.astype(np.float32) / 255.0
            else:
                input_img = input_img.astype(np.uint8)

        # Sinon, masque ou image déjà en NDG
        else:
            # gardes uniquement les valeurs originales → masque toujours propre
            input_img = input_img // 255
            input_img = cv.resize(input_img, self.size, interpolation=cv.INTER_NEAREST)
            input_img = input_img.astype(np.uint8)

        return input_img

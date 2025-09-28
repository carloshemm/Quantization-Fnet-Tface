import cv2
import numpy as np
import torch
import os
import json
import tqdm
import pandas as pd

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)

from faceNetEmbedding512 import FaceEmbedding512

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed

# --- lazy model global (will be created inside each worker process) ---
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        # instantiate model inside worker to avoid pickling issues
        _embeddings = FaceEmbedding512()
    return _embeddings

embeddings = _get_embeddings()

# keep original variable name
dataset_path = "/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/images"

embeddings_resultados = []

def infer_image(imagepath):
    """
    imagepath is a path relative to dataset_path (so works with subfolders)
    """
    full_path = f"{dataset_path}/{imagepath}"
    img = cv2.imread(full_path)
    if img is None:
        # return an entry indicating failure so you can inspect later
        return {
            "nome": imagepath,
            "embedding": None,
            "error": "imread_failed"
        }


    # Image to embedding 512
    faceembedding = embeddings.getEmbedding(img)

    # if torch tensor -> np -> list
    try:
        if hasattr(faceembedding, "detach"):
            faceembedding = faceembedding.detach().cpu().numpy()
        if isinstance(faceembedding, np.ndarray):
            faceembedding = faceembedding.tolist()
    except Exception:
        # fallback: try converting to list directly
        try:
            faceembedding = list(faceembedding)
        except Exception:
            print("erro na geracao de embedding")
            faceembedding = None

    dado = {
        "nome": imagepath,
        "embedding": faceembedding,
        "error": None
    }

    return dado

if __name__ == "__main__":
    # build a list of image paths relative to dataset_path (walks subfolders)
    imagesDataset = []
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, dataset_path)  # relative path inside dataset_path
                imagesDataset.append(rel)

    results = []
    print("Started")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(infer_image, image) for image in imagesDataset]
        for future in tqdm.tqdm(futures, total=len(imagesDataset)):
            try:
                results.append(future.result())
            except Exception as e:
                # keep going if a worker crashes
                results.append({"nome": None, "embedding": None, "error": repr(e)})

    df = pd.DataFrame(results).sort_values(by=['nome'])
    df.to_csv("output.csv", index=False)

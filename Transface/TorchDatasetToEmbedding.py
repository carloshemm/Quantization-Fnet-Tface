import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model
import dlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)

class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        resnet = get_model("vit_s_dp005_mask_0", dropout=0, fp16=False).cuda()
        resnet.load_state_dict(weight)
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg):

        img = cv2.resize(rimg, (self.image_size[1], self.image_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, weight, local_patch_entropy = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


batch_size = 1
data_shape = (3, 112, 112)

faceness_scores = []


embedding = Embedding("models/glint360k_model_TransFace_S.pt", data_shape, batch_size)



dataset_path = "/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/images"


def infer(img_name):
    try:
        img = cv2.imread(os.path.join(dataset_path,img_name))

        input_blob = embedding.get(img)
        output = embedding.forward_db(input_blob)
        
        return {
            "nome": img_name,
            "embedding": output[0].tolist()[:512] # Convert numpy array to list
        }

        return None
    except Exception as e:
        print(e)
        return None

def main():
    images_dataset = []
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, dataset_path)  # relative path inside dataset_path
                images_dataset.append(rel)

    
    results = []
    for image in tqdm(images_dataset):
        resultado = infer(image)
        results.append(resultado)
    

    embeddings_resultados = [result for result in results if result is not None]
    df = pd.DataFrame(embeddings_resultados).sort_values(by=['nome'])
    df.to_csv("output.csv", index=False)  # Updated CSV name

if __name__ == "__main__":
    main()

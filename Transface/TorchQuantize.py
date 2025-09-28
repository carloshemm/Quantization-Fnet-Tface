import numpy as np
import time
import os
import cv2
import torch
import torch.quantization as tq
from skimage import transform as trans
from backbones import get_model
from sklearn.metrics import roc_curve, auc
import dlib
from tqdm import tqdm


import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)

# Paths and configuration
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DATASET_PATH = "/media/bunto22/TOSHIBA EXT/Dataset/vgg2/val/"
MODEL_PATH = "models/glint360k_model_TransFace_S.pt"
INT8_MODEL_PATH = "models/transface_int8_model.pt"

# Quantization settings
WARMUP_ITERATIONS = 1
BENCHMARK_ITERATIONS = 5
INPUT_SHAPE = (1, 3, 112, 112)


class Embedding:
    def __init__(self, model_state, quantize=False):
        # Load base model on CPU (required before quantization)
        model = get_model("vit_s_dp005_mask_0", dropout=0, fp16=False)
        model.load_state_dict(torch.load(model_state, map_location="cpu"))
        model.eval()

        if quantize:
            print("-- Preparing dynamic quantization on CPU --")
            # Keep model on CPU for quantization
            model = tq.quantize_dynamic(
                model, {torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8
            )
            # Save quantized CPU model
            torch.save(model.state_dict(), INT8_MODEL_PATH)
            print(f"Quantized model saved to {INT8_MODEL_PATH}")
            self.device = torch.device('cpu')
            # No DataParallel for CPU quantized model
            self.model = model
        
        else:
            # Use GPU if available for FP32
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(self.device)
            # Wrap for multi-GPU
            self.model = torch.nn.DataParallel(model)

        self.model.eval()

        # Move model to correct device
        self.model.to(self.device)

        # Pre-compute alignment template
        self.src = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.2041]
        ], dtype=np.float32)
        self.src[:, 0] += 8.0


    def preprocess(self, img):
        img = cv2.resize(img, (112, 112))
        flip = np.fliplr(img)
        blob = np.stack([img, flip], axis=0).astype(np.uint8)
        return blob.transpose(0,3,1,2)

    @torch.no_grad()
    def forward(self, batch):
        x = torch.from_numpy(batch).float().to(self.device)
        x.div_(255).sub_(0.5).div_(0.5)
        feat, _, _ = self.model(x)
        return feat.cpu().numpy()


def benchmark(embedder):
    dummy = np.random.rand(*INPUT_SHAPE).astype(np.float32)
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        embedder.forward(dummy)
    # Timing
    times = []
    for _ in range(BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        embedder.forward(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"Avg {np.mean(times):.2f}ms | Std {np.std(times):.2f}ms | TPS {1000/np.mean(times):.2f}")


def evaluate(embedder, root):
    feats, labels = [], []
    for uid in tqdm(os.listdir(root)):
        pdir = os.path.join(root, uid)
        if not os.path.isdir(pdir):
            continue
        for imgf in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, imgf))

            blob = embedder.preprocess(img)
            feats.append(embedder.forward(blob).flatten())
            labels.append(uid)
    feats = np.vstack(feats)
    labels = np.array(labels)
    # Build pairs
    pos, neg = [], []
    rng = np.random.RandomState(42)
    ids = np.unique(labels)
    for u in ids:
        idx = np.where(labels == u)[0]
        if len(idx) > 1:
            a, b = rng.choice(idx, 2, False)
            pos.append((a, b))
    for _ in range(len(pos)):
        a, b = rng.choice(len(labels), 2, False)
        while labels[a] == labels[b]:
            a, b = rng.choice(len(labels), 2, False)
        neg.append((a, b))
    y_true = [1]*len(pos) + [0]*len(neg)
    y_score = [1 - np.dot(feats[i], feats[j]) / (np.linalg.norm(feats[i]) * np.linalg.norm(feats[j]))
               for i, j in pos + neg]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    print(f"AUC: {auc(fpr, tpr):.4f}")


if __name__ == '__main__':
    # INT8 Dynamic CPU
    print("### INT8 Dynamic CPU ###")
    e_int8 = Embedding(MODEL_PATH, quantize=True)
    benchmark(e_int8)
    evaluate(e_int8, DATASET_PATH)

    # FP32 GPU
    print("### FP32 GPU ###")
    e_fp = Embedding(MODEL_PATH, quantize=False)
    benchmark(e_fp)
    evaluate(e_fp, DATASET_PATH)


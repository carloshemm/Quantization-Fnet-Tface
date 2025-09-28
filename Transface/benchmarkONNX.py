import os
import numpy as np
import torch
from backbones import get_model
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil  # For memory monitoring
import cv2

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)


DatasetPath = "/media/bunto22/TOSHIBA EXT/Dataset/vgg2/val/"
Datasetfolder = os.listdir(DatasetPath)
FoldersPath = [os.path.join(DatasetPath, f) for f in Datasetfolder]
DatasetImages = []

for folder in Datasetfolder:
    images = os.listdir(os.path.join(DatasetPath, folder))
    DatasetImages.extend([os.path.join(DatasetPath, folder, im) for im in images])

rng = np.random.default_rng()
imagesIndexes = rng.choice(len(DatasetImages), 100, replace=False)
calibration_files = [DatasetImages[x] for x in imagesIndexes]


# Example: Load and preprocess calibration images (adjust paths)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

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
        feat = feat.reshape([self.batch_size, 1 * feat.shape[1]])
        return feat.cpu().numpy()

# Initialize model
batch_size = 1
data_shape = (3, 112, 112)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding = Embedding("models/glint360k_model_TransFace_S.pt", data_shape, batch_size)

# Prepare model for export
model_pytorch = embedding.model# Make sure model is on CPU for export
model_pytorch.eval()




sess_fp32 = ort.InferenceSession("TransfaceSmall-512.onnx", providers=["CUDAExecutionProvider"])
sess_int8 = ort.InferenceSession("TransfaceSmall-512-quantized.onnx", providers=["CUDAExecutionProvider"])

def benchmark(model_type, model, input_data):
    latencies = []
    embeddings = []
    
    # Warmup
    for _ in range(10):
        if model_type == "pytorch":
            emb = model.forward_db(input_data)
        else:
            _ = model.run(None, {"input": input_data.cpu().numpy()})
    
    # Benchmark
    for _ in range(100):
        start = time.time()
        if model_type == "pytorch":
            emb = model.forward_db(input_data)
        else:
            emb = model.run(None, {"input": input_data.cpu().numpy()})[0]
        latencies.append((time.time() - start) * 1000)  # ms
        embeddings.append(emb)
    
    avg_latency = np.mean(latencies)
    return avg_latency, embeddings[-1]  # Return last embedding for accuracy test


results = []
for img_path in calibration_files[:10]:  # Test on 10 images for speed
    input_tensor = preprocess_image(img_path).to(device)
    
    # PyTorch (FP32)
    latency_pytorch, emb_pytorch = benchmark("pytorch", embedding, input_tensor)
    emb_pytorch = emb_pytorch
    
    # ONNX FP32
    latency_onnx_fp32, emb_onnx_fp32 = benchmark("onnx", sess_fp32, input_tensor)
    
    # ONNX INT8
    latency_onnx_int8, emb_onnx_int8 = benchmark("onnx", sess_int8, input_tensor)
    
    # Cosine Similarity
    cos_sim_fp32 = cosine_similarity(emb_pytorch, emb_onnx_fp32)[0][0]
    cos_sim_int8 = cosine_similarity(emb_pytorch, emb_onnx_int8)[0][0]
    
    results.append({
        "image": img_path,
        "latency_pytorch": latency_pytorch,
        "latency_onnx_fp32": latency_onnx_fp32,
        "latency_onnx_int8": latency_onnx_int8,
        "cos_sim_fp32": cos_sim_fp32,
        "cos_sim_int8": cos_sim_int8,
    })


print("\n--- Benchmark Results (Averages) ---")
print(f"PyTorch (FP32) Latency: {np.mean([r['latency_pytorch'] for r in results]):.2f} ms")
print(f"ONNX (FP32) Latency:    {np.mean([r['latency_onnx_fp32'] for r in results]):.2f} ms")
print(f"ONNX (INT8) Latency:    {np.mean([r['latency_onnx_int8'] for r in results]):.2f} ms")
print(f"\n--- Accuracy (Cosine Similarity) ---")
print(f"ONNX FP32 vs PyTorch:   {np.mean([r['cos_sim_fp32'] for r in results]):.6f}")
print(f"ONNX INT8 vs PyTorch:   {np.mean([r['cos_sim_int8'] for r in results]):.6f}")


if torch.cuda.is_available():
    print("\n--- GPU Memory Usage ---")
    print(f"PyTorch: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"ONNX FP32: ~{psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB (RAM)")
    # ONNX Runtime GPU memory usage is harder to track; use `nvidia-smi` manually.
import os
import numpy as np
import torch
from facenet.inception_resnet_v1 import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil  # For memory monitoring


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


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Shape: [1, 3, 160, 160]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_pytorch = InceptionResnetV1(pretrained='vggface2').eval().to(device)


sess_fp32 = ort.InferenceSession("facenet-512.onnx", providers=["CUDAExecutionProvider"])
sess_int8 = ort.InferenceSession("facenet-512-quantized.onnx", providers=["CUDAExecutionProvider"])

def benchmark(model_type, model, input_data):
    latencies = []
    embeddings = []
    
    # Warmup
    for _ in range(10):
        if model_type == "pytorch":
            _ = model(input_data)
        else:
            _ = model.run(None, {"input": input_data.cpu().numpy()})
    
    # Benchmark
    for _ in range(100):
        start = time.time()
        if model_type == "pytorch":
            emb = model(input_data)
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
    latency_pytorch, emb_pytorch = benchmark("pytorch", model_pytorch, input_tensor)
    emb_pytorch = emb_pytorch.cpu().detach().numpy()
    
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
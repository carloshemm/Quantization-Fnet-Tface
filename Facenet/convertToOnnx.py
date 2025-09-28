import torch
from facenet.inception_resnet_v1 import InceptionResnetV1
import os
import numpy as np


import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)



model = InceptionResnetV1(pretrained='vggface2').eval()
dummy_input = torch.randn(1, 3, 160, 160)  # Input shape for FaceNet

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "facenet-512.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=13,  # Required for quantization
)



###################################################

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Example: Load and preprocess calibration images (adjust paths)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# Example calibration dataset (use ~100 images)
#load images
DatasetPath = "/media/bunto22/TOSHIBA EXT/Dataset/vgg2/val/"
Datasetfolder = os.listdir(DatasetPath)
FoldersPath = [os.path.join(DatasetPath,f) for f in Datasetfolder]
DatasetImages = []

for folder in Datasetfolder:
    images = os.listdir(os.path.join(DatasetPath,folder))
    DatasetImages.extend([os.path.join(DatasetPath,folder,im) for im in images])
rng = np.random.default_rng()
imagesIndexes = rng.choice(len(DatasetImages), 100, replace=False)

calibration_files = [DatasetImages[x] for x in imagesIndexes]
calibration_data = [preprocess_image(f) for f in calibration_files]



###################################################


from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# Create a DataReader for calibration
class FaceNetDataReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_data = {"input": self.data[self.index].numpy()}
            self.index += 1
            return input_data
        else:
            return None

# Quantize the model
quantize_static(
    "facenet-512.onnx",
    "facenet-512-quantized.onnx",
    calibration_data_reader=FaceNetDataReader(calibration_data),
    quant_format=QuantType.QInt8,  # Use QInt8 for weights/activations
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)

#END OF QUANTIZATION
###################################################


import onnxruntime as ort

# Initialize ONNX Runtime session (CUDA)
providers = ["CUDAExecutionProvider"]  # Use GPU
sess = ort.InferenceSession("facenet-512-quantized.onnx", providers=providers)

# Prepare input (example)
input_tensor = calibration_data[0].numpy()  # Replace with your input

# Run inference
output = sess.run(["embedding"], {"input": input_tensor})
print(output[0].shape)  # Should output (1, 512) embeddings



######################################################


import torch
from facenet.inception_resnet_v1 import InceptionResnetV1
import time

model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
input_tensor = torch.randn(1, 3, 160, 160).cuda()

# Warmup
for _ in range(10):
    _ = model(input_tensor)

# Benchmark
start = time.time()
for _ in range(100):
    embedding_pytorch = model(input_tensor)
latency_pytorch = (time.time() - start) / 100 * 1000  # ms per inference
print(f"PyTorch Latency: {latency_pytorch:.2f} ms")


###########################################################


import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("facenet-512.onnx", providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name

# Warmup
for _ in range(10):
    _ = sess.run(["embedding"], {input_name: input_tensor.cpu().numpy()})

# Benchmark
start = time.time()
for _ in range(100):
    embedding_onnx = sess.run(["embedding"], {input_name: input_tensor.cpu().numpy()})[0]
latency_onnx = (time.time() - start) / 100 * 1000  # ms per inference
print(f"ONNX FP32 Latency: {latency_onnx:.2f} ms")


#################################################################333



import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("facenet-512.onnx", providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name

# Warmup
for _ in range(10):
    _ = sess.run(["embedding"], {input_name: input_tensor.cpu().numpy()})

# Benchmark
start = time.time()
for _ in range(100):
    embedding_onnx = sess.run(["embedding"], {input_name: input_tensor.cpu().numpy()})[0]
latency_onnx = (time.time() - start) / 100 * 1000  # ms per inference
print(f"ONNX FP32 Latency: {latency_onnx:.2f} ms")


######################################################################


sess_quant = ort.InferenceSession("facenet-512-quantized.onnx", providers=["CUDAExecutionProvider"])

# Warmup
for _ in range(10):
    _ = sess_quant.run(["embedding"], {input_name: input_tensor.cpu().numpy()})

# Benchmark
start = time.time()
for _ in range(100):
    embedding_quant = sess_quant.run(["embedding"], {input_name: input_tensor.cpu().numpy()})[0]
latency_quant = (time.time() - start) / 100 * 1000  # ms per inference
print(f"ONNX INT8 Latency: {latency_quant:.2f} ms")


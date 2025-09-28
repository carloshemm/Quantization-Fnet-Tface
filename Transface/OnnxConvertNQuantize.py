import torch
import numpy as np
from backbones import get_model
import os

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)

class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        resnet = get_model("vit_s_dp005_mask_0", dropout=0, fp16=False)
        resnet.load_state_dict(weight)
        self.model = resnet  # Don't use DataParallel for export
        self.model.eval()
        
        # Alignment stuff (not used in export but kept for completeness)
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

# Initialize model
batch_size = 1
data_shape = (3, 112, 112)
embedding = Embedding("models/glint360k_model_TransFace_S.pt", data_shape, batch_size)

# Prepare model for export
model = embedding.model.cpu()  # Make sure model is on CPU for export
model.eval()

# Create dummy input on the same device as model
dummy_input = torch.randn(1, 3, 112, 112)  # Keep on CPU

# Export to ONNX
torch.onnx.export(
    model,  # Use the raw model, not DataParallel
    dummy_input,
    "TransfaceSmall-512.onnx",
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={
        "input": {0: "batch"}, 
        "embedding": {0: "batch"}
    },
    opset_version=13,
    verbose=True  # Add this to see more export details
)

print("ONNX export successful!")


###################################################

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Example: Load and preprocess calibration images (adjust paths)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
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
    "TransfaceSmall-512.onnx",
    "TransfaceSmall-512-quantized.onnx",
    calibration_data_reader=FaceNetDataReader(calibration_data),
    quant_format=QuantType.QInt8,  # Use QInt8 for weights/activations
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)

#END OF QUANTIZATION
###################################################





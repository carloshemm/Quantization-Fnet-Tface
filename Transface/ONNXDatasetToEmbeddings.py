import cv2
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import onnxruntime as ort  # Replaces PyTorch

import os
import pathlib

CD = pathlib.Path(__file__).parent.resolve()
os.chdir(CD)


# --- Initialize ONNX Model (Replaces PyTorch) ---
class ONNXFaceEmbedding512:
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # Use GPU if available
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def getEmbedding(self, aligned_face):
        # Preprocess aligned face for ONNX model
        # 1. Convert BGR to RGB
        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        # 2. Resize to 160x160 (if needed)
        resized_face = cv2.resize(rgb_face, (112, 112))
        # 3. Normalize: (x / 255 - 0.5) / 0.5 (same as PyTorch's [0.5, 0.5, 0.5] mean/std)
        normalized_face = (resized_face.astype(np.float32) / 255.0 - 0.5) / 0.5
        # 4. Convert to CHW format and add batch dimension
        input_tensor = np.transpose(normalized_face, (2, 0, 1))[np.newaxis, ...]  # Shape: [1, 3, 160, 160]

        # Run ONNX inference
        embedding = self.session.run(
            [self.output_name], 
            {self.input_name: input_tensor.astype(np.float32)}
        )[0]
        return embedding.flatten()  # Convert to 1D array (512 dimensions)

# Load quantized ONNX model
embeddings = ONNXFaceEmbedding512("TransfaceSmall-512.onnx")  # Your quantized ONNX file

# --- Dataset Path (unchanged) ---
dataset_path = "/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/images"


# --- Inference Function (modified for ONNX) ---
def infer_image(imagepath):
    try:
        img = cv2.imread(os.path.join(dataset_path, imagepath))
                
        

        # Get embedding using ONNX model
        face_embedding = embeddings.getEmbedding(img)
        
        return {
            "nome": imagepath,
            "embedding": face_embedding.tolist()  # Convert numpy array to list
            }
        
    except Exception as e:
        print(f"Error processing {imagepath}: {e}")
        return None

# --- Main Function (unchanged except CSV name) ---
def main():
    images_dataset = []
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, dataset_path)  # relative path inside dataset_path
                images_dataset.append(rel)
    
    
    #images_dataset = os.listdir(dataset_path)
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(infer_image, image) for image in images_dataset]
        results = []
        for future in tqdm(futures, total=len(images_dataset)):
            results.append(future.result())
    
    # Save to CSV with "onnx-" prefix
    embeddings_resultados = [result for result in results if result is not None]
    df = pd.DataFrame(embeddings_resultados).sort_values(by=['nome'])
    df.to_csv("output.csv", index=False)  # Updated CSV name

if __name__ == "__main__":
    main()
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


class ONNXFaceEmbedding512:
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # Use GPU if available
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def getEmbedding(self, aligned_face):

        rgb_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        resized_face = cv2.resize(rgb_face, (160, 160))
        
        normalized_face = (resized_face.astype(np.float32) / 255.0 - 0.5) / 0.5

        input_tensor = np.transpose(normalized_face, (2, 0, 1))[np.newaxis, ...] 

        # Run ONNX inference
        embedding = self.session.run(
            [self.output_name], 
            {self.input_name: input_tensor.astype(np.float32)}
        )[0]
        return embedding.flatten()  # Convert to 1D array (512 dimensions)


embeddings = ONNXFaceEmbedding512("facenet-512-quantized.onnx")  


# --- Dataset Path (unchanged) ---
dataset_path = "/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/images"

# --- Inference Function (modified for ONNX) ---
def infer_image(imagepath):
    try:
        img = cv2.imread(os.path.join(dataset_path, imagepath))

        
        face_embedding = embeddings.getEmbedding(img)
        
        return {
            "nome": imagepath,
            "embedding": face_embedding.tolist()  # Convert numpy array to list
        }

    except Exception as e:
        print(f"Error processing {imagepath}: {e}")
        return None


def main():
    images_dataset = []


    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, dataset_path)  # relative path inside dataset_path
                images_dataset.append(rel)
    

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(infer_image, image) for image in images_dataset]
        results = []
        for future in tqdm(futures, total=len(images_dataset)):
            results.append(future.result())
    

    embeddings_resultados = [result for result in results if result is not None]
    df = pd.DataFrame(embeddings_resultados).sort_values(by=['nome'])
    df.to_csv("output.csv", index=False)  # Updated CSV name

if __name__ == "__main__":
    main()
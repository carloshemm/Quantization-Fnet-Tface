import numpy as np
import torch
from PIL import Image
import cv2
import time

from facenet.inception_resnet_v1 import InceptionResnetV1

class FaceEmbedding512:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        
    def getEmbedding(self, img):
        try:
            
            image = cv2.resize(img, (160,160))
            image = torch.tensor(image).permute(2,0,1).to(self.device)
            image = image/255.0
            start = time.perf_counter()
            embedding = self.model(image.unsqueeze(0))
            end  = time.perf_counter()
            timeSpent = end-start
            embedding = embedding.cpu()
                
            
            embedding = embedding.detach().numpy()[0].tolist()
            
            
            
            return embedding
        except:
            print("Error en getEmbedding")
            return None
        
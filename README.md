# Experimental Evaluation of Quantization Methods in Facial Recognition


📌 Official Implementation of our proposed method of quantization.

## Abstract
Quantization techniques are recognized for their effective in optimizing large language models. However, their application in smaller language models, such as those used for face recognition, is uncommon. In the field of facial recognition, these techniques can be utilized across a diverse range of settings, from API services in data centers to mobile application deployment. Smaller quantized models can reduce costs while maintaining a strong performance. However, their combination with facial recognition is uncommon. This study tested the impact of 8-bit quantization on embedding generation models for facial recognition. We compared two models: FaceNet (Inception-ResNet) and TransFace (Vision Transformer). We analyzed different precision formats (FP32 and INT8) and inference backends (Torch and ONNX) using datasets such as LFW, VGGFace2, and CelebA to evaluate the top-1 accuracy and cosine distance for similarity. The results show that FaceNet performs well under quantization, maintaining accuracy while reducing precision. In contrast, TransFace exhibited a significant decrease in performance. Quantizing embedding vectors can also reduce storage needs by up to 80% without much loss in performance. These findings support quantization as an effective strategy for optimizing models in resource-limited environments and enhancing facial recognition technology.


## Model Weights
You can download the pretrained weights and configs from [Google Drive](https://drive.google.com/drive/folders/1Cn_-LrrGdaIa4MIAlCQGmBorWmFz0yLX?usp=sharing).
 
## Citation

```bibtex 
@inproceedings{monteiro2025quant,
  author    = {Carlos Henrique Monteiro and Evandro Raphaloski and Edson Takashi Matsubara},
  title     = {Empirical Evaluation of Quantization Methods for Facial Recognition Models},
  booktitle = {Proceedings of the 37th IEEE/SBC International Symposium on Computer Architecture and High Performance Computing},
  year      = {2025},
  pages     = {1--9},
  publisher = {SBAC-PAD},
  doi       = {10.1109/[to-be-assigned]},
  note      = {LeanDL-HPC 2025 Workshop on Lightweight and Efficient Deep Learning in HPC Environments}
}
```

## Acknowledgment
Our code was build base on [ONNX](https://github.com/onnx/onnx.git), [PyTorch](https://github.com/pytorch/pytorch.git), [FaceNet](https://github.com/timesler/facenet-pytorch.git), [Transface](https://github.com/DanJun6737/TransFace.git) and all datasets availables used on this study. Thanks for their public repository and excellent contributions!


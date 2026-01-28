# RepSFNet

Official Implementation of **RepSFNet: A Single Fusion Network with Structural Reparameterization for Crowd Counting**, published at **IEEE AVSS 2025**.

RepSFNet is a lightweight and efficient crowd counting framework designed for accurate and real-time estimation, particularly suitable for low-power edge computing scenarios.

---

## 🔍 Overview

Crowd counting in real-world scenarios remains challenging due to extreme density variations, occlusions, and high computational demands.  
RepSFNet addresses these challenges by introducing a **single fusion architecture with structural reparameterization**, avoiding heavy attention mechanisms and multi-branch designs.

Key highlights:
- Lightweight architecture with low latency
- Large receptive field via reparameterized large kernels
- Single fusion design for efficiency and simplicity
- Suitable for real-time and edge deployment

---

## 🏗 Architecture

RepSFNet consists of three main components:

1. **RepLK-ViT Backbone**  
   - Uses large reparameterized convolutional kernels
   - Efficient multi-scale feature extraction
   - Transformer-like global perception without self-attention

2. **Feature Fusion Module**  
   - Integrates **Atrous Spatial Pyramid Pooling (ASPP)** and  
     **Context-Aware Network (CAN)**
   - Provides robust, density-adaptive context modeling

3. **Concatenate Fusion Module**  
   - Preserves spatial resolution
   - Produces high-quality density maps through multi-level feature concatenation

The network is trained using a combined loss:
- Mean Squared Error (MSE)
- Optimal Transport (OT) loss for spatial distribution alignment

---

## 📊 Experimental Results

RepSFNet is evaluated on multiple public benchmarks:

- **ShanghaiTech (Part A & B)**
- **NWPU-Crowd**
- **UCF-QNRF**

### Performance Highlights
- Competitive accuracy compared to state-of-the-art methods
- Up to **34% lower inference latency** than models such as:
  - P2PNet
  - M-SFANet
  - M-SegNet
  - STEERER
  - Gramformer
- Excellent trade-off between accuracy, speed, and computational cost

These results demonstrate that RepSFNet is well-suited for real-time crowd counting in resource-constrained environments.

---

## 📄 Paper

**RepSFNet: A Single Fusion Network with Structural Reparameterization for Crowd Counting**  
Published in *Proceedings of the IEEE International Conference on Advanced Visual and Signal-Based Systems (AVSS), 2025*.

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{achmadiah2025repsfnet,
  title={RepSFNet: A Single Fusion Network with Structural Reparameterization for Crowd Counting},
  author={Achmadiah, Mas Nurul and Sun, Chi-Chia and Kuo, Wen-Kai and Hsieh, Jun-Wei},
  booktitle={2025 IEEE International Conference on Advanced Visual and Signal-Based Systems (AVSS)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}


# EfficientPS-Med-ConvNeXt-Based-Panoptic-Segmentation-for-Healthcare-Imaging

A lightweight panoptic segmentation framework for medical imaging using a ConvNeXt backbone.

---

## 🧠 Project Overview

**EfficientPS-Med** extends the [EfficientPS](https://arxiv.org/abs/2004.00526) panoptic segmentation architecture to medical images. The original EfficientPS uses a shared backbone with two heads (semantic + instance) and a panoptic fusion module. In **EfficientPS-Med**, we **replace the EfficientNet backbone with ConvNeXt**, a modern convolutional network inspired by vision transformers.

ConvNeXt achieves transformer-level performance using pure convolutions, yielding better speed and accuracy on medical images. This architectural upgrade significantly improves performance on tasks such as **organ** and **lesion segmentation**, and accelerates inference in real-time or clinical environments.

---

## ⚙️ Key Features

- **🔁 ConvNeXt Backbone**  
  Replaces EfficientNet with ConvNeXt-Tiny or Small. ConvNeXt uses inverted bottleneck blocks, patchify stems, and large kernels — delivering high accuracy (87.8% ImageNet Top-1) and fast inference.

- **🔗 Dual Heads (Semantic + Instance)**  
  - **Semantic Head:** Outputs pixel-wise class predictions (e.g., organs, tissues).  
  - **Instance Head (Mask R-CNN):** Outputs segmented objects (e.g., lesions, tumors).  
  Both operate concurrently.

- **🧩 Panoptic Fusion Module**  
  Combines semantic and instance predictions to yield a unified **panoptic segmentation** output — each pixel gets a class and instance ID (if applicable).

- **🔁 Two-Way Feature Pyramid Network (FPN)**  
  Bidirectional FPN extracts fine and coarse details across multiple scales. Improves segmentation for both large organs and small lesions.

- **🧪 Multi-Dataset Support**  
  Easily configurable for different medical datasets including CT, MRI, endoscopy (e.g., **LiTS**, **Kvasir**, etc.).

---

## 🏥 Use Cases & Healthcare Impact

- **Liver CT Segmentation (LiTS)**  
  Auto-delineation of liver and tumor regions supports tumor burden analysis and treatment planning.

- **Endoscopy & Gastrointestinal (Kvasir)**  
  Real-time polyp and ulcer segmentation improves early cancer detection in GI imaging.

- **Preoperative Surgical Mapping**  
  Helps surgeons visualize and identify distinct anatomical structures from 3D panoptic maps.

- **Clinical Workflow Automation**  
  Reduces manual labeling effort, boosts diagnostic consistency, and increases throughput in radiology.

- **Education & Annotation**  
  Helps train medical students and enables faster dataset labeling via model-assisted annotation.

---

## 🧪 Results & Visualizations

<p align="center">
  <img src="outputs/examples/liver_panoptic.png" width="400" alt="Liver Segmentation"/>
  <img src="outputs/examples/kvasir_polyp.png" width="400" alt="Polyp Segmentation"/>
</p>

**Example Metrics:**
- Liver CT (LiTS):  
  - **Semantic mIoU:** 82.5%  
  - **Panoptic Quality (PQ):** 78.1  
  - **Instance AP (Tumors):** 74.2  

- Endoscopy (Kvasir):  
  - **Semantic mIoU:** 79.3%  
  - **PQ:** 75.0  
  - **AP:** 71.5  

> 📈 Training plots and metrics logs (PQ, mIoU, AP over epochs) are available in `/outputs/metrics.csv` and TensorBoard logs.

---

## 🔧 Installation & Setup

```bash
git clone https://github.com/YourOrg/EfficientPS-Med.git
cd EfficientPS-Med
conda create -n ep-med python=3.8
conda activate ep-med
pip install -r requirements.txt
````

---

## 🧬 Datasets

* Supported formats: **COCO**, **Pascal VOC**, **NIfTI** (for 3D).
* Examples:

  * [LiTS](https://competitions.codalab.org/competitions/17094) – Liver CT
  * [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) – Polyp segmentation

Organize data under `data/` or configure via YAML:

```yaml
data_root: data/lits
num_classes: 3
input_size: [512, 512]
```

---

## 🏋️ Training & Evaluation

**Train:**

```bash
python train.py --config configs/liver_segmentation.yaml
```

**Evaluate:**

```bash
python evaluate.py --config configs/liver_segmentation.yaml --checkpoint outputs/checkpoint_final.pth
```

**Outputs:**

* Predicted masks (semantic, instance)
* Panoptic overlays (PNG)
* Metrics (`outputs/metrics.csv`)
* Visual logs (optional TensorBoard)

---

## 🧠 Project Structure

```
configs/        # YAML configs for datasets & experiments
data/           # Dataset loaders and sample data links
models/         # ConvNeXt backbone, dual heads, FPN, fusion
train.py        # Training script
evaluate.py     # Evaluation/inference
outputs/        # Logs, masks, metrics, visualizations
requirements.txt
README.md
LICENSE
```

---

## 🚀 Future Work

* 🧠 **3D Support:** Volumetric CNNs for brain/cardiac CT/MRI.
* 🧠 **ConvNeXt + Transformer Hybrid:** Add axial attention or Swin-style context.
* 📦 **Model Compression:** Quantization/pruning for edge deployment.
* ⚙️ **Deployment Toolkit:** Export to ONNX, REST API, Docker container.
* 🧬 **More Modalities:** Support for ultrasound, histopathology.
* 🖥️ **Interactive GUI:** Plugin for 3D Slicer or web annotation.

---

## 📄 License

Released under the **MIT License**. See `LICENSE` file.

---

## 👨‍💻 Contact

For contributions or questions, feel free to reach out:

* 📧 [cchsaikrishnachowdary@gmail.com](mailto:cchsaikrishnachowdary@gmail.com)
* 💼 [LinkedIn](https://www.linkedin.com/in/sai-krishna-chowdary-chundru)
* 💻 [GitHub](https://github.com/sAI-2025)

---

## 📚 References

* EfficientPS: "EfficientPS: Efficient Panoptic Segmentation" – [Mehta et al., CVPR 2020](https://arxiv.org/abs/2004.00526)
* ConvNeXt: "A ConvNet for the 2020s" – [Liu et al., CVPR 2022](https://arxiv.org/abs/2201.03545)
* Panoptic Segmentation Surveys – various architectural reviews and analysis papers on segmentation strategies.

---

```


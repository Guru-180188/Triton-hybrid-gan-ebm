# Hybrid GAN-EBM Inference Pipeline

### Production-Grade Image Generation & Energy-Based Evaluation with NVIDIA Triton

This project implements a **Hybrid Generative Architecture** where a **Generative Adversarial Network (GAN)** handles image synthesis and an **Energy-Based Model (EBM)** acts as an expert evaluator. The entire pipeline is deployed using **NVIDIA Triton Inference Server** for high-performance, containerized serving.

## 🚀 The Hybrid Approach

In this setup, the models play complementary roles:

* **The Generator (GAN):** Transforms a 128-dimensional latent noise vector into a  RGB image.
* **The Evaluator (EBM):** Analyzes the generated image and assigns an **Energy Score**. Lower energy indicates the image more closely aligns with the learned data distribution (CIFAR-10).
* **The Processor:** A custom client-side pipeline that takes the raw output and applies high-fidelity **Lanczos upscaling** to reach **4K resolution**.

---

## 🏗 System Architecture

The system is built to be modular and scalable, separating the inference logic from the model serving.

### Key Technologies:

* **NVIDIA Triton:** Orchestrates model execution and handles batching.
* **TorchScript (.pt):** Models are serialized from PyTorch to TorchScript to remove Python-dependency during runtime.
* **Docker:** Provides a consistent environment for the NVIDIA Container Toolkit.

---

## 📂 Model Repository Structure

Triton requires a strict folder hierarchy to serve models. **Note:** `.venv` and other scripts are kept outside the `triton_models` folder to avoid loading errors.

```text
.
├── triton_models/             # Root model repository for Triton
│   ├── generator/             # GAN Model Folder
│   │   ├── 1/                 # Version 1
│   │   │   └── model.pt       # TorchScript Model
│   │   └── config.pbtxt       # Model configuration
│   └── ebm/                   # Energy-Based Model Folder
│       ├── 1/
│       │   └── model.pt
│       └── config.pbtxt
├── test_inference.py          # 4K Inference Client
└── requirements.txt           # Python dependencies

```

---

## 🛠 Setup & Deployment

### 1. Configure NVIDIA Container Toolkit

To enable GPU acceleration within the Docker container:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```

### 2. Launch the Triton Server

Run the container and mount your model repository:

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

```

### 3. Run the Inference Client

Once the server shows a `READY` status for both models, execute the client script to generate and upscale an image:

```bash
python3 test_inference.py

```

---

## ⚙️ Configuration Details (`config.pbtxt`)

The models use a 4-dimensional input tensor to satisfy **Convolutional** layer requirements (`ConvTranspose2d`).

**Generator Input Configuration:**

```protobuf
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 128, 1, 1 ]  # [Channels, Height, Width]
  }
]

```

---

## 🖼 Result Output

The pipeline produces a high-resolution image (`generated_4k.png`) by upscaling the  GAN output using **Lanczos interpolation** and a sharpening filter to maintain visual clarity at  resolution.


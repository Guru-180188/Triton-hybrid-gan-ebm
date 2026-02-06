
# Hybrid GAN-EBM Inference Service (NVIDIA Triton)

This repository contains a production-grade deployment of a **Generative Adversarial Network (GAN)** paired with an **Energy-Based Model (EBM)**. The system is orchestrated using the **NVIDIA Triton Inference Server**, enabling high-throughput, low-latency image generation and quality scoring.

## 🚀 System Architecture

Unlike standard inference scripts, this project utilizes **Server-Side Orchestration**:

1. **Generator (GAN):** Transforms 128-dimensional latent noise into  (CIFAR-10 style) images.
2. **Evaluator (EBM):** Analyzes the generated images and assigns an "Energy Score" (Lower = More Realistic).
3. **Ensemble Pipeline:** A Triton Ensemble links both models internally, eliminating the need to pass heavy image data back and forth to the client.

## 🛠 Features

* **Dynamic Batching:** Automatically groups individual requests (up to batch size 32) to maximize GPU/CPU utilization.
* **Production Monitoring:** Real-time metrics for Queue Time, Compute Latency, and Throughput available via the Prometheus endpoint.
* **4K Post-Processing:** Built-in `utils` to upscale outputs to  using high-fidelity Lanczos interpolation.
* **Hardware Agnostic:** Pre-configured for **KIND_CPU**; easily switchable to **KIND_GPU** for production clusters.

---

## 📁 Repository Structure

| File/Folder | Purpose |
| --- | --- |
| `triton_models/` | The Triton Model Repository containing model weights and `config.pbtxt` files. |
| `client/test_inference.py` | Production-grade client script with latency reporting and batching support. |
| `client/utils.py` | Data preparation and 4K image transformation toolkit. |
| `docker-compose.yml` | Standardized environment setup for one-command deployment. |
| `requirements.txt` | Python dependencies for the client application. |

---

## 🚦 Quick Start

### 1. Launch the Inference Server

Ensure you have Docker and Docker Compose installed. From the root directory, run:

```bash
docker-compose up -d

```

### 2. Install Client Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run Inference

The client generates a batch of 4 images, scores them via the EBM, and saves upscaled versions to the `generated_images/` folder.

```bash
python3 client/test_inference.py

```

---

## 📊 Monitoring & Logs

This project prioritizes observability. You can monitor the health of your models directly through the terminal.

**View Latency & Throughput Metrics:**

```bash
curl -s localhost:8002/metrics | grep "nv_inference"

```

**Key Metric Descriptions:**

* `nv_inference_queue_duration_us`: Time requests wait in the scheduler.
* `nv_inference_compute_infer_duration_us`: Actual time spent on model math.
* `nv_inference_request_success`: Total successful generation cycles.

---

## ⚙️ Configuration Tuning

To adjust the performance of the system, edit the `config.pbtxt` files within the `triton_models/` directory:

* **Instance Count:** Increase `count` in `instance_group` to handle more parallel requests.
* **Max Batch Size:** Adjust `max_batch_size` based on your available RAM/VRAM.
* **Hardware Kind:** Swap `KIND_CPU` to `KIND_GPU` to leverage NVIDIA hardware acceleration.

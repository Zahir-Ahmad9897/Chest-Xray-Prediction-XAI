<div align="center">

# 🫁 Chest X-Ray Pneumonia XAI Dashboard

**An interactive web dashboard for Chest X-Ray Pneumonia classification with 5 Explainable AI (XAI) methods.**

Built with **FastAPI** · **PyTorch** · **Tailwind CSS** · **Alpine.js**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#-features)
- [XAI Methods](#-xai-methods)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

This project provides a **web-based dashboard** for classifying chest X-ray images as **Normal** or **Pneumonia** using a fine-tuned **DenseNet-121** model. Beyond prediction, it offers **five explainability methods** that generate visual heatmaps to help clinicians and researchers understand *why* the model made a particular decision.

> **⚠️ Disclaimer:** This tool is intended for **educational and research purposes only**. It is not a certified medical device and should not be used for clinical diagnosis.

---

## ✨ Features

- 🔍 **Binary Classification** — Normal vs. Pneumonia detection from chest X-ray images
- 🧠 **5 XAI Methods** — Grad-CAM, Grad-CAM++, LIME, SHAP, and Integrated Gradients
- 🖥️ **Interactive Dashboard** — Upload images, view predictions, and compare explanations side-by-side
- ⚡ **REST API** — Programmatic access to prediction and explanation endpoints
- 🎨 **Modern UI** — Responsive design with Tailwind CSS and Alpine.js
- ♿ **WCAG 2.1 AA** — Skip links, ARIA labels, focus rings, and reduced-motion support

---

## 🧠 XAI Methods

| Method | Type | Description |
|---|---|---|
| **Grad-CAM** | Gradient-based | Gradient-weighted Class Activation Mapping on `features.norm5` |
| **Grad-CAM++** | Gradient-based | Improved Grad-CAM with alpha weighting for better localization |
| **LIME** | Perturbation-based | Local Interpretable Model-agnostic Explanations via superpixel perturbation |
| **SHAP** | Attribution-based | GradientExplainer for pixel-level Shapley value attribution |
| **Integrated Gradients** | Attribution-based | Captum-based integral attribution from baseline to input |

---

## 🏗️ Architecture

| Component | Implementation Details |
|---|---|
| **Model** | DenseNet-121 (fine-tuned), binary classification head |
| **Grad-CAM** | Uses `torch.autograd.grad()` — no backward hooks (avoids PyTorch 2.x bugs) |
| **DenseNet Patch** | `F.relu(inplace=True)` → `F.relu(inplace=False)` (defense-in-depth) |
| **SHAP** | 10 background samples, 30 nsamples (RAM-optimized) |
| **Backend** | FastAPI with async endpoints, Uvicorn ASGI server |
| **Frontend** | Tailwind CSS + Alpine.js, served as static files |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **pip** package manager
- ~**1 GB** disk space (for PyTorch CPU + model weights)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Zahir-Ahmad9897/Chest-Xray-Prediction-XAI.git
cd Chest-Xray-Prediction-XAI
```

**2. Create and activate a virtual environment**

```bash
# Create
python -m venv venv

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> **💡 Tip (Windows):** If you encounter a `c10.dll` / `WinError 1114` error with PyTorch, install the CPU-only version:
> ```bash
> pip uninstall torch torchvision torchaudio -y
> pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

**4. Download the pre-trained model**

```bash
python -c "from services.model_service import download_model_from_drive; download_model_from_drive()"
```

**5. Run the application**

```bash
python main.py
```

---

## 💻 Usage

1. Open your browser and navigate to **http://localhost:8000**
2. Upload a chest X-ray image (JPEG, PNG)
3. View the **prediction** (Normal / Pneumonia) with confidence score
4. Select one or more **XAI methods** to generate explanation heatmaps
5. Compare explanations side-by-side to understand model reasoning

---

## 📡 API Reference

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "device": "cpu",
  "model_loaded": true
}
```

### Predict

```http
POST /api/predict
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|---|---|---|
| `file` | `UploadFile` | Chest X-ray image (JPEG/PNG) |

### Explain

```http
POST /api/explain?methods=gradcam,gradcam_pp,lime,shap,ig
Content-Type: multipart/form-data
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | `UploadFile` | — | Chest X-ray image (JPEG/PNG) |
| `methods` | `query string` | `gradcam,gradcam_pp,lime,shap,ig` | Comma-separated XAI methods |

### Download Model

```http
POST /api/setup/download-model
```

Downloads the pre-trained model weights from Google Drive and reloads the model.

---

## ⚙️ Configuration

All settings are centralized in [`config.py`](config.py) and can be overridden via environment variables:

| Setting | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MODEL_URL` | *(Google Drive link)* | URL to download model weights |
| `IMG_SIZE` | `224` | Input image size (px) |
| `GRADCAM_TARGET_LAYER` | `features.norm5` | DenseNet layer for Grad-CAM |
| `SHAP_BACKGROUND_SIZE` | `10` | SHAP background samples |
| `SHAP_NSAMPLES` | `30` | SHAP evaluation samples |
| `LIME_NUM_SAMPLES` | `500` | LIME perturbation samples |
| `IG_STEPS` | `25` | Integrated Gradients steps |

---

## 📁 Project Structure

```
chest-xray-pneumonia-xai/
├── main.py                  # FastAPI application entry point
├── config.py                # Centralized configuration
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation (this file)
│
├── services/                # Backend services
│   ├── __init__.py
│   ├── model_service.py     # Model loading, preprocessing, prediction
│   └── xai_service.py       # XAI method implementations
│
├── templates/               # HTML templates
│   └── index.html           # Dashboard UI
│
├── static/                  # Static assets
│   ├── css/                 # Stylesheets
│   └── js/                  # JavaScript files
│
└── model/                   # Model weights (auto-downloaded, git-ignored)
    └── best.pt
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `WinError 1114` / `c10.dll` failed | Install PyTorch CPU version (see [Installation](#installation) tip) |
| `Scripts cannot be loaded` (PowerShell) | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `Model not loaded` (503 error) | Run the model download step or hit `POST /api/setup/download-model` |
| Out of memory during SHAP | Reduce `SHAP_BACKGROUND_SIZE` and `SHAP_NSAMPLES` in `config.py` |
| Slow XAI generation | Use fewer methods in the `methods` query parameter |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Explainable AI in Healthcare**

</div>
# 🚢 Hugging Face Spaces Deployment Guide

This guide details the steps required to deploy the **Transistor Defect Detector** website on Hugging Face Spaces using the **Docker SDK**.

---

## Why Docker?

Hugging Face Spaces supports multiple SDKs (Gradio, Streamlit, Docker). We use the **Docker SDK** because it allows us to deploy a fully customized website built with Flask, HTML5, CSS3, and JavaScript, giving us complete control over styling, layouts, animations, and the user experience.

---

## Deployment Steps

### 1. Create a Hugging Face Account
If you haven't already, sign up at [huggingface.co](https://huggingface.co/).

### 2. Create a New Space
1. Go to the Hugging Face Spaces homepage and click **"Create new Space"** or visit [huggingface.co/new-space](https://huggingface.co/new-space).
2. Enter a name (e.g., `transistor-defect-detector`).
3. Select **Docker** as the SDK (instead of Gradio/Streamlit).
4. Choose the **Blank** template.
5. Set Space visibility to **Public** (or Private if desired).
6. Click **"Create Space"**.

---

### 3. Clone your Space Repository
Once created, copy the clone URL and run it locally:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

---

### 4. Copy Project Files to the Space Folder
Copy the following files and folders from this project directory into the cloned Space directory:
- `configs/`
- `data/download_dataset.py` (optional, but keep folder structure)
- `src/`
- `web/`
- `Dockerfile`
- `requirements.txt`
- `README.md` (Ensure this has the YAML header block at the very top)

Your Space directory structure should match:
```
YOUR_SPACE_NAME/
├── README.md              # Contains Hugging Face Space YAML metadata
├── Dockerfile             # Port and CMD configuration
├── requirements.txt
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── gradcam.py
│   └── utils.py
└── web/
    ├── app.py
    ├── templates/
    └── static/
```

> [!IMPORTANT]
> Make sure `README.md` begins with the required Hugging Face YAML metadata:
> ```yaml
> ---
> title: Transistor Defect Detector
> emoji: 🔬
> colorFrom: blue
> colorTo: purple
> sdk: docker
> app_port: 7860
> ---
> ```

---

### 5. Add Trained Model Weights (Recommended)
By default, the web app will fall back to using pre-trained ImageNet weights if no custom fine-tuned weights are present. To show your trained model's performance on the Space:
1. Run training locally to generate `outputs/checkpoints/best_efficientnet_b0.pth`.
2. Create a folder `outputs/checkpoints/` inside your Space folder.
3. Copy your local `best_efficientnet_b0.pth` weight file into that folder.

*Alternatively, you can upload the weights to the Hugging Face Hub as a model and download them dynamically in `web/app.py` upon startup.*

---

### 6. Commit and Push to Deploy
Run these commands inside your Space directory:
```bash
# Add all files to Git tracking
git add .

# Commit changes
git commit -m "Deploy custom Flask web application with Docker SDK"

# Push to Hugging Face
git push
```

Hugging Face will automatically detect the push, trigger a Docker build using the included `Dockerfile`, and deploy the web application. You can view the build logs and live interface under your Space's URL.

---

## Local Docker Testing

Before pushing to Hugging Face, you can build and run the Docker container locally to verify it starts without errors:
```bash
# Build the image
docker build -t defect-detector .

# Run the container (Map Flask port 7860)
docker run -p 7860:7860 defect-detector
```
Open **[http://localhost:7860](http://localhost:7860)** to test the containerized deployment.

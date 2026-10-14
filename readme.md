# Diffusion Image Denoiser: From Proof of Concept to Production

**Course:** Data, Technology and Design (Term 7, 2026)

This repository implements an end-to-end MLOps workflow for a diffusion-based image denoising application on a **64×64 RGB image**. Starting from the forward diffusion equation:

$$\mathbf{z}_t = \sqrt{\alpha_t}\mathbf{x} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_t$$

we compare a pixel-wise **Linear Regression Baseline** against a **Small Convolutional Neural Network (CNN)** that exploits local spatial structure, and package the resulting model into a tested, containerized, and deployed **Streamlit** web application.

## Project Structure

```text
diffusion-six/
|-- train.py                   # Trains Baseline & CNN models and saves model.pt
|-- model.pt                   # Trained PyTorch model parameters
|-- app.py                     # Streamlit web interface for interactive denoising
|-- part1_poc.py               # Standalone Part 1 PoC comparison script
|-- mountain_64x64.png         # Default 64x64 RGB clean target image
|-- requirements.txt           # Python dependencies
|-- Dockerfile                 # Docker containerization instructions
|-- README.md                  # Documentation and reproduction instructions
|-- .gitignore                 # Ignored cache and virtual environment files
|-- tests/
|   `-- test_diffusion.py      # Automated pytest test suite
`-- .github/
    `-- workflows/
        `-- test.yml           # GitHub Actions CI workflow
```

## 1. How to Install and Run the Application Locally

Clone the repository, install dependencies, train the model (if `model.pt` is not yet generated), and launch the Streamlit interface:

```bash
git clone <repository-url>
cd diffusion-six
python3 -m pip install -r requirements.txt
python3 train.py
python3 -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 2. How to Run the Automated Tests

We use `pytest` to verify image dimensions (`64x64`), forward noise addition, CNN output tensor shapes, and numerical validity of reconstructed images:

```bash
pytest
```

## 3. How to Build and Run the Docker Container

Make sure Docker Desktop is running, then build the image and run the container from inside the `diffusion-six/` directory:

```bash
docker build -t diffusion-six .
docker run -p 8501:8501 diffusion-six
```

Then open `http://localhost:8501` in your browser to use the packaged application.

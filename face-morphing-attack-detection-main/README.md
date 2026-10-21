# Face Morphing Attack Detection Using MixNet

A deep-learning research project that classifies a face image as either a **Morph Attack** or **Bona Fide** image. The project uses a MixNet-S model, evaluates a baseline checkpoint against an augmentation-based fine-tuning experiment, and provides a Streamlit web application for interactive predictions.

> **Disclaimer:** This is a research and educational prototype. Its predictions and softmax scores are not guaranteed to be correct and must not be used as the sole basis for identity verification or other high-stakes decisions.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Research Paper](#research-paper)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Confusion Matrices](#confusion-matrices)
- [Web Application](#web-application)
- [Repository Structure](#repository-structure)
- [Run Locally](#run-locally)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements and References](#acknowledgements-and-references)

## Project Overview

Face morphing combines facial characteristics from two or more people into a single image. Such images can pose a risk to face-based identity-document verification because a morphed image may resemble more than one person. **Morphing Attack Detection (MAD)** aims to distinguish morphed images from genuine, non-morphed face images.

This project implements a binary image classifier with the following classes:

- **Morph Attack** — class label `0`
- **Bona Fide** — class label `1`

A Streamlit interface allows a user to upload a JPG, JPEG, or PNG image and view the model's predicted class and softmax score.

## Objectives

- Study research on face morphing attack detection and synthetic training data.
- Use a MixNet-S model to classify face images.
- Evaluate a baseline model on a separate evaluation dataset.
- Conduct an augmentation-based fine-tuning experiment and compare its results with the baseline.
- Report classification metrics, Equal Error Rate (EER), and confusion matrices.
- Deploy a simple web interface for demonstration.

## Research Paper

The project refers to:

**Marco Huber et al., “SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data” (2022).**

The paper discusses synthetic training data for morphing attack detection and the SMDD dataset. The described data-generation pipeline includes StyleGAN2-ADA, CR-FIQA-based sample selection, and multiple morph-generation approaches, including landmark-based and GAN-based methods.

The paper motivates careful evaluation across different morph-generation methods, since some morphs can be more difficult to detect than others.

**Important distinction:** The augmentation experiment in this repository—random horizontal flipping and mild colour jitter—is an experiment conducted for this project. It should not be interpreted as an augmentation method prescribed by the research paper. The metrics reported below are this project's own results on its evaluation dataset; they are not the paper's reported competition results.

## Dataset

The experiments used separate training/validation and evaluation datasets.

### Training and validation data

- **Total images:** 20,000
- **Bona Fide:** 12,500
- **Morph Attack:** 7,500
- **Training split:** 16,000 images (80%)
- **Validation split:** 4,000 images (20%)
- **Split method:** stratified split with `random_state=42`

### Independent evaluation data

- **Total images:** 40,000
- **Bona Fide:** 25,000
- **Morph Attack:** 15,000

The class mapping used during evaluation is `0 = Morph Attack` and `1 = Bona Fide`.

The datasets were accessed through Kaggle datasets during development. The dataset names used in the project were:

- Training images: `sakshi0414/smdd-20k-morphed-image-detection`
- Evaluation images: `sakshi0414/smdd-evaluation-dataset`
- Code/model implementation source: `sakshi0414/smdd-mixfacenet-code`
- Training checkpoints: `sakshi0414/smdd-training-checkpoints`

Dataset access and redistribution terms should be checked on the respective dataset pages. This repository does not imply ownership of the datasets or permission to redistribute their images.

## Model Architecture

The classifier uses the **MixNet-S** architecture with the following recorded configuration:

| Setting | Value |
|---|---|
| Architecture | MixNet-S |
| Embedding size | `128` |
| Width scale | `1.0` |
| GDW size | `1024` |
| Shuffle | `False` |
| Input size | `224 × 224` RGB |
| Number of parameters | `2,679,384` |
| Output classes | 2 |

Model construction:

```python
from backbones import mixnet_s

model = mixnet_s(
    embedding_size=128,
    width_scale=1.0,
    gdw_size=1024,
    shuffle=False
)
```

### Preprocessing

Both training/evaluation inference and the web app use the following basic preprocessing:

1. Convert the image to RGB.
2. Resize it to `224 × 224` pixels.
3. Convert it to a tensor.
4. Normalize each channel with mean `0.5` and standard deviation `0.5`.

## Methodology

### 1. Baseline model

The baseline was trained in Kaggle using a Tesla T4 GPU. Recorded training settings:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.0001` |
| Batch size | `16` |
| Maximum epochs | `20` |
| Early stopping patience | `5` |
| Checkpoint selection | Lowest validation EER |

The baseline checkpoint was then evaluated on the separate 40,000-image evaluation dataset.

### 2. Augmentation-based fine-tuning experiment

The experimental model was **fine-tuned from the baseline checkpoint**, not trained from scratch. It used the same model architecture and the same training/validation split.

Training augmentations:

- Random horizontal flip with probability `0.5`.
- Mild colour jitter: brightness `0.15`, contrast `0.15`, saturation `0.10`, applied with probability `0.5`.

Fine-tuning settings:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.00001` |
| Batch size | `16` |
| Epochs | `5` |
| Loss | Cross-entropy |
| Checkpoint selection | Lowest validation loss |

The experiment was evaluated on the same 40,000-image evaluation dataset. It did **not** improve generalisation, so the baseline checkpoint was retained for the deployed app.

## Experimental Results

All results below are from this project's evaluation on the separate 40,000-image dataset.

| Metric | Baseline MixNet | Augmentation fine-tuned MixNet |
|---|---:|---:|
| Accuracy | **98.23%** | 94.24% |
| Precision (Bona Fide class) | **97.41%** | 91.64% |
| Recall (Bona Fide class) | 99.82% | **99.90%** |
| F1-score (Bona Fide class) | **98.60%** | 95.59% |
| Equal Error Rate (EER) | **0.83%** | 2.04% |

**Interpretation:** The fine-tuned model's accuracy decreased by 3.99 percentage points, and EER increased by 1.21 percentage points. The baseline is therefore the selected model for the web app.

Precision, recall, and F1-score in this table refer specifically to the **Bona Fide class**. They are not attack-class metrics.

### Baseline confusion matrix

Rows represent the true class and columns represent the predicted class. Class order is `[Morph Attack, Bona Fide]`.

| True class | Predicted Morph Attack | Predicted Bona Fide |
|---|---:|---:|
| Morph Attack | 14,337 | 663 |
| Bona Fide | 45 | 24,955 |

The baseline misclassified 663 morph attacks as bona fide and 45 bona fide images as morph attacks.

### Augmentation fine-tuned confusion matrix

| True class | Predicted Morph Attack | Predicted Bona Fide |
|---|---:|---:|
| Morph Attack | 12,722 | 2,278 |
| Bona Fide | 25 | 24,975 |

The experimental model misclassified 2,278 morph attacks as bona fide and 25 bona fide images as morph attacks. Although its bona fide recall was slightly higher, its attack false-acceptance count and overall EER were worse.

### Fine-tuning history

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.0088 | 99.70% | 0.0042 | 99.92% |
| 2 | 0.0049 | 99.84% | 0.0026 | 99.90% |
| 3 | 0.0076 | 99.76% | 0.0030 | 99.90% |
| 4 | 0.0068 | 99.86% | 0.0026 | 99.90% |
| 5 | 0.0043 | 99.88% | 0.0033 | 99.88% |

High validation accuracy did not translate into better performance on the separate evaluation dataset. This is why evaluation-set performance, not validation accuracy alone, guided model selection.

## Web Application

The project includes a Streamlit application with these features:

- Upload JPG, JPEG, or PNG images.
- Preview the uploaded image.
- Run MixNet inference when **Analyze Image** is clicked.
- Display `Morph Attack` or `Bona Fide`.
- Display the softmax score for the predicted class.
- Show a research-use disclaimer.

**Live demo:** [Face Morphing Attack Detection](https://sakshi-chougale-face-morphing-attack-detection-app-hpfehr.streamlit.app/)

Two manual examples were checked in the deployed interface: one expected morph-attack image was classified as `Morph Attack`, and one expected bona-fide image was classified as `Bona Fide`. Both displayed 100.00% softmax scores. These two examples verify the basic interface workflow only; they do not establish real-world accuracy.

> The displayed softmax score is not a calibrated probability. A score of 100% does not guarantee that a prediction is correct.

## Repository Structure

The main files should be organised as follows:

```text
face-morphing-attack-detection/
├── app.py                 # Streamlit application
├── best_model.pth         # Baseline MixNet checkpoint
├── requirements.txt       # Python dependencies
└── backbones/             # MixNet architecture implementation
```

The exact contents inside `backbones/` depend on the model implementation used in the project.

## Run Locally

### Requirements

- Python 3.10 or a compatible Python version supported by the installed PyTorch and torchvision releases.
- PyTorch
- torchvision
- Streamlit
- Pillow

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Sakshi-Chougale/face-morphing-attack-detection.git
   cd face-morphing-attack-detection
   ```

2. Create and activate a virtual environment (recommended).

   Windows:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit in your browser.

**Dependency note:** The repository's `requirements.txt` currently lists `streamlit`, `torch`, `torchvision`, and `Pillow` without version pins. If deployment or local installation produces compatibility issues, pin a tested compatible set of versions.

## Limitations

- Results are tied to the particular datasets, split, checkpoint, preprocessing, and evaluation protocol used.
- Performance may vary across capture devices, lighting, image quality, demographic groups, and unseen morph-generation methods.
- A high overall accuracy does not remove the risk of false acceptance of morph attacks.
- The augmentation fine-tuning experiment did not improve performance on the independent evaluation dataset.
- Softmax scores are not calibrated confidence estimates.
- The application performs image classification; it is not a complete identity verification, document validation, or liveness detection system.
- Only a small number of examples were manually tested through the deployed UI; broad operational validation has not been established.

## Future Work

- Test on additional independent datasets and different morph-generation methods.
- Report attack-class precision, recall, F1-score, false-acceptance rate, and false-rejection rate alongside accuracy and EER.
- Investigate domain shift, dataset bias, and potential data leakage.
- Evaluate additional augmentation policies and fine-tuning settings with a held-out test set.
- Calibrate scores and choose a decision threshold according to the intended use and risk requirements.
- Add image-quality checks and clearer model/version information to the interface.

## Acknowledgements and References

1. M. Huber et al., **“SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data,”** 2022. Verify the complete author list, publication venue, pages, and DOI from the original paper before formal academic submission.
2. MixNet implementation used in this project. Add the exact upstream paper/repository citation for the implementation included in `backbones/`.
3. Streamlit documentation: https://docs.streamlit.io/

Dataset references:

- Training dataset: https://www.kaggle.com/datasets/sakshi0414/smdd-20k-morphed-image-detection
- Evaluation dataset: https://www.kaggle.com/datasets/sakshi0414/smdd-evaluation-dataset
- Code/model implementation dataset: https://www.kaggle.com/datasets/sakshi0414/smdd-mixfacenet-code
- Training checkpoints dataset: https://www.kaggle.com/datasets/sakshi0414/smdd-training-checkpoints

Check each dataset's licence and terms before redistributing its data or derived assets.

---

## Author

**Sakshi Chougale**

Project: Face Morphing Attack Detection Using MixNet  
Live demo: https://sakshi-chougale-face-morphing-attack-detection-app-hpfehr.streamlit.app/

If you use or extend this work, please cite the research paper and the upstream MixNet implementation, and clearly distinguish your own evaluation results from published benchmark results.

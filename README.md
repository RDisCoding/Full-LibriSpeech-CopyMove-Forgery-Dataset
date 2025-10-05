# Full LibriSpeech Copy–Move Forgery Dataset

## 📘 Overview

The **Full LibriSpeech Copy–Move Forgery Dataset** is a benchmark created to support research in **audio forgery detection**, particularly **copy–move manipulation** detection and localization. In copy–move forgery, segments from a single audio file are duplicated and repositioned, making detection difficult for traditional verification systems.

This dataset provides cleanly partitioned, speaker-disjoint splits (train/validation/test), detailed temporal annotations, and multiple forgery strength levels to facilitate reproducible and fair evaluation of deep learning models.

---

## 📦 Dataset Access

The full dataset (~25 GB) is publicly available on Hugging Face:

👉 [https://huggingface.co/datasets/TheAnalyzer/Full-LibriSpeech-CopyMove-Forgery-Dataset](https://huggingface.co/datasets/TheAnalyzer/Full-LibriSpeech-CopyMove-Forgery-Dataset)

You can clone it using Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/TheAnalyzer/Full-LibriSpeech-CopyMove-Forgery-Dataset
```

---

## ⚙️ Usage

Preprocessing scripts, baseline CNN implementations, and evaluation notebooks are available in this repository to help reproduce the results presented in the paper.

```bash
git clone https://github.com/RDisCoding/Full-LibriSpeech-CopyMove-Forgery-Dataset
cd LibriSpeech-CMF
```

---

## 📬 Contact

For questions or collaborations, please contact: **[rdiscoding@gmail.com](mailto:rdiscoding@gmail.com)**

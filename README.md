# HES: Early Multi-Modal Fake News Detection

HES (Hierarchical Early-fusion System) is an advanced deep learning framework designed for early detection of fake news. The system integrates multiple modalities (text and images) to identify misinformation at an early stage, helping combat the spread of fake news across social media platforms.

## 🚀 Features

- **Early Detection**: Detects fake news at an early stage before widespread propagation
- **Comprehensive Metrics**: Evaluates performance using multiple metrics (Accuracy, Precision, Recall, F1, AUC, etc.)

## 📋 Prerequisites

### System Requirements
- Python >= 3.8
- CUDA-compatible GPU (recommended for training)
- Sufficient RAM for processing large multimodal datasets

### Dependencies
Install the required dependencies:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111
pip install tensorflow==2.7.0
pip install openai-clip==1.0.1
pip install transformers==4.12.5
pip install sentencepiece==0.1.96
pip install scikit-learn==1.0
pip install numpy==1.24.4
pip install pandas==2.0.1
pip install matplotlib==3.5.2
pip install librosa==0.10.1
```

## 📦 Data Preparation

Prepare your dataset in the following format:
- Text features
- Image features
- Labels (real/fake)
- Scenario information
- Event identifiers

## 🧰 Pre-trained Models

You'll need to download pre-trained CLIP models for optimal performance. The system supports[ViT-B/32](https://github.com/openai/CLIP).

## 🚀 Usage

### Training
To train the model, run:

```bash
bash train.sh
```

### Testing
To evaluate the trained model, run:

```bash
bash test.sh
```


## 🏗️ Architecture

The system is organized as follows:

```
src/
├── data/                  # Data loading and preprocessing
├── module/                # Model implementations
│   ├── CompareModel/      # Comparison models (CLIP, DICE, etc.)
│   ├── model/             # Main model implementations
│   ├── loss/              # Loss functions
│   ├── wrapper/           # Model wrappers
│   └── ablation/          # Ablation study models
├── tools/                 # Utilities (metrics, preprocessing, etc.)
├── main.py               # Main entry point
└── Inference.py          # Inference script
```

## 📊 Evaluation Metrics

The framework includes comprehensive evaluation metrics:
- Accuracy
- Precision, Recall, F1-score
- AUC (Area Under Curve)

## 📚 Citation

If you use this code in your research, please cite:

```
@article{ma2025graphing,
}


```

Other comparison models:
```
@article{liu2023robust,
  title={Robust domain misinformation detection via multi-modal feature alignment},
  author={Liu, Hui and Wang, Wenya and Sun, Hao and Rocha, Anderson and Li, Haoliang},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={19},
  pages={793--806},
  year={2023},
  publisher={IEEE}
}
```

## 🔐 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✅ Acknowledgments
We thank the researchers whose work made this repository possible, particularly the authors of the compared models (CLIP, DICE, EAAN, etc.) and the original papers referenced in this work.
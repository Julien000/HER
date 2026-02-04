# HES: Hierarchical Event-Scenario Semantic Modeling for Early Fake News Detection

Detecting fake news in real-world settings is fundamentally challenged by unseen events, where shifts in topic, style, and modality undermine the generalization of existing models. Most approaches treat each event as isolated, ignoring latent semantic structures shared across them. In this work, we challenge this assumption and propose a framework that explicitly models both invariant cues of misinformation and context-dependent event semantics. By disentangling and dynamically integrating these two types of information, our method captures both transferable deception patterns and useful contextual signals from semantically related events. This design enables effective generalization across diverse scenarios without sacrificing specificity. Experiments on multiple benchmarks demonstrate consistent improvements in both unimodal and multimodal settings. Beyond performance, our approach offers a new perspective on event-level generalization, revealing that context, when structured appropriately, can be a strength rather than a source of noise..

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
│   ├── model/             # Main model implementations
│   ├── loss/              # Loss functions
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
@article{
}

```


## 🔐 License

This project is licensed under the MIT License - see the LICENSE file for details.

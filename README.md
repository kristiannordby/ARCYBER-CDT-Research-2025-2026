# Cybersecurity Article Classification for Common Crawl Data

**Army Cyber Command Capstone Research Project**  
United States Military Academy, Class of 2026

---

## Overview

This repository contains the cybersecurity classification component of our Army Cyber Command capstone research project. We evaluate three distinct machine learning approaches for classifying web articles from the [Common Crawl](https://commoncrawl.org/) corpus as related or unrelated to cybersecurity. This work is part of a larger research effort that also includes classification tasks for advertisement detection and national security relevance (not included in this repository, but following identical methodologies).

Our research addresses the challenge of identifying security-relevant content from massive web archives, enabling automated threat intelligence gathering and security research at scale.

---

## Project Objectives

The primary objectives of this research are to:

1. **Develop scalable binary classifiers** capable of identifying cybersecurity-related content from the Common Crawl dataset
2. **Evaluate multiple embedding and classification approaches** to determine optimal architectures for this task
3. **Address severe class imbalance** through weighted random sampling techniques
4. **Assess multilingual performance** across English and non-English content
5. **Deploy production-ready models** for real-world threat intelligence applications

---

## Dataset

### Source
All training and evaluation data is sourced from [Common Crawl](https://commoncrawl.org/), a repository of web crawl data composed of petabytes of data collected since 2008.

### Composition
- **General Sample**: 200,000 randomly selected articles representing the natural distribution of web content
- **Cyber-Biased Sample**: 70,000 articles pre-filtered for potential cybersecurity relevance
- **Total Dataset**: 262,398 unique articles (after deduplication)
- **Language Coverage**: Both English and non-English content, with separate evaluation metrics

### Class Distribution
The dataset exhibits severe class imbalance reflecting real-world web content:
- **Cybersecurity-related**: ~5% of articles
- **Non-cybersecurity**: ~95% of articles

This imbalance necessitates specialized sampling strategies during training.

---

## Methodology

### Class Imbalance Strategy

To address the extreme class imbalance without data duplication, we implement **Weighted Random Sampling** during training:

- Each training epoch samples a 50/50 balanced distribution of positive and negative examples
- Sampling weights are computed as the inverse of class frequencies
- This approach achieves balanced training without artificially duplicating minority class samples
- Validation and test sets maintain the natural class distribution for realistic performance evaluation

### Architecture Comparison

We evaluate three fundamentally different approaches to the classification task:

#### 1. **EmbeddingGemma + Neural Network** (`cyberGemma.ipynb`)
- **Embedding Model**: Google's [EmbeddingGemma](https://huggingface.co/google/gemma-2b-embedding)
- **Architecture**: Feed-forward neural network classifier trained on pre-computed embeddings
- **Input Dimension**: 2048-dimensional embeddings
- **Advantages**: Computationally efficient inference, separates embedding generation from classification
- **Implementation**: Pre-tokenization of all samples, memory-efficient training pipeline

#### 2. **mmBERT End-to-End Fine-Tuning** (`mmCyberTrainer.py`)
- **Base Model**: [mmBERT (Multilingual and Monolingual BERT)](https://huggingface.co/jhu-clsp/mmBERT-base)
- **Architecture**: Full transformer model fine-tuned for binary sequence classification
- **Training Strategy**: End-to-end optimization with mixed precision training
- **Optimization**: 
  - Pre-tokenization of entire dataset
  - Gradient accumulation for effective larger batch sizes
  - WeightedRandomSampler for balanced training
  - Early stopping based on validation AUC
- **Advantages**: Learns task-specific representations, superior multilingual performance

#### 3. **LaBSE + Neural Network** (`CyberLABSE.ipynb`)
- **Embedding Model**: [Language-Agnostic BERT Sentence Embedding (LaBSE)](https://huggingface.co/sentence-transformers/LaBSE)
- **Architecture**: Feed-forward neural network classifier trained on LaBSE embeddings
- **Input Dimension**: 768-dimensional embeddings
- **Advantages**: Specifically designed for multilingual semantic similarity, strong cross-lingual performance
- **Implementation**: Embedding-based approach with efficient training pipeline

---

## Technical Implementation

### Training Pipeline Features

All three approaches implement:

- **Pre-tokenization**: Complete dataset tokenization performed once before training
- **Weighted Random Sampling**: Probabilistic 50/50 class balance each epoch without data duplication
- **Memory Efficiency**: No artificial data duplication, streaming data loading where applicable
- **Mixed Precision Training**: FP16 automatic mixed precision for faster training (BERT approach)
- **Gradient Accumulation**: Effective large batch sizes without memory overflow (BERT approach)
- **Early Stopping**: Validation AUC monitoring with configurable patience
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, Specificity, NPV
- **Language-Stratified Evaluation**: Separate performance metrics for English vs. non-English content

### Evaluation Metrics

All models are evaluated on:

- **Primary Metrics**: AUC-ROC, F1 Score, Precision, Recall
- **Secondary Metrics**: Accuracy, Specificity, Negative Predictive Value
- **Confusion Matrix**: Complete breakdown of classification outcomes
- **Language-Specific Performance**: Separate evaluation for English and non-English subsets

---

## Results

### Overall Performance Summary

| Model | Test AUC | F1 Score | Precision | Recall | Accuracy |
|-------|----------|----------|-----------|--------|----------|
| **mmBERT Fine-tuned** | **0.9400** | **0.5246** | 0.4330 | **0.6655** | 0.9547 |
| **EmbeddingGemma + NN** | 0.9291 | 0.5766 | **0.4744** | 0.7350 | 0.9239 |
| **LaBSE + NN** | 0.9194 | 0.5283 | 0.7373 | 0.4117 | 0.9482 |

### Language-Specific Performance

#### English Articles

| Model | AUC | F1 | Precision | Recall |
|-------|-----|----|-----------| -------|
| **EmbeddingGemma** | 0.9291 | 0.5766 | 0.4744 | 0.7350 |
| **LaBSE** | 0.9194 | 0.5283 | 0.7373 | 0.4117 |

#### Non-English Articles

| Model | AUC | F1 | Precision | Recall |
|-------|-----|----|-----------| -------|
| **mmBERT** | **0.9400** | 0.5246 | 0.4330 | 0.6655 |
| **EmbeddingGemma** | 0.9400 | 0.5246 | 0.4330 | 0.6655 |
| **LaBSE** | 0.9392 | 0.5186 | 0.6231 | 0.4441 |

### Key Findings

1. **All models achieve >91% AUC**, demonstrating strong discriminative ability despite severe class imbalance
2. **mmBERT shows superior overall performance**, particularly in AUC and recall metrics
3. **Precision-recall trade-offs vary by approach**: LaBSE favors precision, while Gemma favors recall
4. **Multilingual performance is strong** across all models, with minimal degradation on non-English content
5. **Weighted sampling successfully balances training** while maintaining realistic test performance

---

## Repository Structure

```
.
├── cyberGemma.ipynb              # EmbeddingGemma approach
├── mmCyberTrainer.py             # mmBERT fine-tuning script
├── CyberLABSE.ipynb              # LaBSE approach
├── data/
│   ├── general_sample_200K_*     # General web sample embeddings
│   └── cyber_biased_sample_70K_* # Cyber-biased sample embeddings
├── datasets/
│   └── cyber_*_embeddings.npz    # Preprocessed embedding arrays
├── models/                        # Trained model checkpoints
└── README.md                      # This file
```

---

## Trained Models

All trained models are publicly available on Hugging Face:

**🤗 Hugging Face Repository**: [kristiangnordby](https://huggingface.co/kristiangnordby)

Available models:
- `kristiangnordby/gemma_cyber` - EmbeddingGemma-based classifier
- `kristiangnordby/mmBERT-cyber-classifier-optimized` - mmBERT fine-tuned model
- `kristiangnordby/cyberLabse` - LaBSE-based classifier

Each model repository includes:
- Complete model weights and configuration
- Tokenizer (for BERT-based models)
- Detailed performance metrics
- Model card with training details
- Inference examples

---

## Requirements

### Core Dependencies
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
numpy>=1.24.0
huggingface-hub>=0.16.0
```

### Additional Dependencies
```
tqdm
jupyter
matplotlib
seaborn
```

### Hardware Requirements
- **Minimum**: 16GB RAM, CPU-only training possible
- **Recommended**: 32GB RAM, NVIDIA GPU with 16GB+ VRAM for BERT fine-tuning
- **Storage**: ~50GB for datasets and model checkpoints

---

## Usage

### Training a Model

#### EmbeddingGemma Approach
```python
# Open and run cyberGemma.ipynb in Jupyter
jupyter notebook cyberGemma.ipynb
```

#### mmBERT Fine-Tuning
```bash
python mmCyberTrainer.py
```

#### LaBSE Approach
```python
# Open and run CyberLABSE.ipynb in Jupyter
jupyter notebook CyberLABSE.ipynb
```

### Loading a Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# For mmBERT model
tokenizer = AutoTokenizer.from_pretrained("kristiangnordby/mmBERT-cyber-classifier-optimized")
model = AutoModelForSequenceClassification.from_pretrained("kristiangnordby/mmBERT-cyber-classifier-optimized")

# Example inference
text = "Critical vulnerability discovered in Apache Log4j library"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()  # 0: non-cyber, 1: cyber
```

---

## Experimental Design

### Training Configuration

| Parameter | EmbeddingGemma | mmBERT | LaBSE |
|-----------|---------------|---------|--------|
| **Batch Size** | 256 | 128 | 256 |
| **Learning Rate** | 1e-3 | 2e-5 | 1e-3 |
| **Optimizer** | Adam | Adam | Adam |
| **Max Epochs** | 100 | 50 | 100 |
| **Early Stopping** | 15 epochs | 10 epochs | 15 epochs |
| **Train/Val/Test Split** | 68%/12%/20% | 68%/12%/20% | 68%/12%/20% |
| **Precision** | FP32 | Mixed FP16 | FP32 |

### Stratification Strategy

All dataset splits maintain stratification by:
1. **Class label** (cyber vs. non-cyber)
2. **Language** (English vs. non-English)

This ensures representative performance evaluation across both dimensions.

---

## Future Work

Potential extensions of this research include:

1. **Multi-class classification**: Expanding beyond binary classification to categorize specific types of cybersecurity threats
2. **Active learning**: Iteratively improving model performance through strategic sample selection
3. **Domain adaptation**: Fine-tuning models for specific cybersecurity subdomains (malware, phishing, vulnerabilities)
4. **Temporal analysis**: Tracking evolution of cybersecurity discourse over time in Common Crawl data
5. **Ensemble methods**: Combining predictions from multiple models for improved robustness
6. **Explainability**: Implementing attention visualization and feature importance analysis
7. **Real-time deployment**: Developing production inference pipelines for continuous threat monitoring

---

## Related Work

This repository focuses on cybersecurity classification. Related components of our capstone project include:

- **Advertisement Classification**: Binary classification of web content as advertising material
- **National Security Classification**: Identifying articles relevant to national security topics

These tasks utilize identical methodologies and training pipelines but are trained on separate labeled datasets.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nordby2026cyber,
  title={Cybersecurity Article Classification for Common Crawl Data},
  author={Nordby, Kristian and collaborators},
  year={2026},
  institution={United States Military Academy},
  note={Army Cyber Command Capstone Research Project}
}
```

---

## Acknowledgments

This research was conducted as part of the Army Cyber Command capstone program at the United States Military Academy. We acknowledge:

- Army Cyber Command for project sponsorship and domain expertise
- United States Military Academy faculty for academic guidance
- The Common Crawl organization for providing open access to web archive data
- The Hugging Face community for transformer model infrastructure
- Google, Johns Hopkins University, and the NLP research community for pre-trained embedding models

---

## License

This project is released under the MIT License. See `LICENSE` file for details.

Models are provided for research and educational purposes. Please review individual model licenses on Hugging Face.

---

## Contact

**Project Lead**: Kristian Nordby  
**Institution**: United States Military Academy, Class of 2026  
**Hugging Face**: [kristiangnordby](https://huggingface.co/kristiangnordby)

For questions about this research, please open an issue in this repository.

---

## Project Status

**Current Phase**: Model deployment and evaluation complete  
**Last Updated**: January 2026  
**Status**: ✅ Active Research

This README provides comprehensive documentation for the cybersecurity classification component of our Army Cyber Command capstone research. For questions or collaboration opportunities, please reach out through the contact information provided above.

# CAFA 5: Multimodal Protein Function Prediction

## Overview

This repository contains the research code and methodology for the CAFA 5 (Critical Assessment of Function Annotation) challenge. The project focuses on multilabel classification of protein function by predicting Gene Ontology (GO) terms (Cellular Component, Molecular Function, Biological Process).

The core innovation lies in a multimodal deep learning framework that fuses:

1. Sequence Data (Raw Amino Acids & k-mers)
1. Structural Domains (InterPro annotations)
1. Physicochemical Properties
1. High-Level Embeddings (ESM2 & T5 Transformer embeddings)

This work demonstrates the application of first-principles modeling in computational biology, bridging physical chemistry, deep learning, and statistical inference for biomedical prediction tasks.

## Skills Demonstrated

The codebase reflects a systematic approach to AI research, highlighting the following competencies:

- Model Architecture & Design: Development of custom Deep Neural Networks (CNNs, RNNs, MLPs) and Multimodal Fusion architectures using TensorFlow/Keras.
- Optimization & Loss Engineering: Implementation of custom loss functions (Focal Loss, Dice Loss, Weighted Combo Loss) to address severe class imbalance and hierarchy constraints in GO terms.
- Data Engineering: Robust pipelines for data ingestion, preprocessing, and embedding extraction (InterPro, ESM2, T5). Efficient use of TF Data pipelines for memory optimization.
- Scientific Rigor: Adherence to reproducibility (seeds, splits), metric validation (Precision/Recall/F1 with class weighting), and error analysis.
- Systems Thinking: CLI wrappers (e.g., iprscan5.py) for external tool integration, environment management (.env), and structured experiment tracking (TensorBoard).

## Methodology Pipeline

The project follows a reproducible research pipeline:

1. Data Ingestion: Parsing FASTA (sequences) and OBO (GO hierarchy) files.
2. Feature Extraction:
   - Sequence: k-mer frequency vectors (window-based).
   - Structure: InterPro domain binarization.
   - Physics: 53 physicochemical properties (flexibility, instability, etc.).
   - Embeddings: Fixed context vectors from ESM2 and T5.
3. Model Fusion: Concatenation of feature vectors passed through shared and specific Dense layers with BatchNorm & Dropout.
4. Training:
   - Loss: Adaptive loss strategies (Focal, Dice, Weighted CE).
   - Optimizer: AdamW with Cosine Decay schedules.
   - Validation: Early stopping based on macro-F1 score.
5. Inference: Multi-head prediction aggregating probabilities across different ontologies (CCO, MFO, BPO).

## Key Achievements & Results

- Performance: Achieved **Top 25%** ranking with competitive F1 scores by balancing model complexity and overfitting via regularization.
- Handling Imbalance: Successfully implemented class-weighted metrics to optimize for rare GO terms (e.g., Focal Loss for tail classes).
- Hierarchy Awareness: Incorporated ontology graph analysis (ancestor/descendant logic) during loss calculation where applicable.
- System Integration: Successfully integrated external bioinformatics tools (InterProScan, Diamond) within the Python workflow via Python clients.

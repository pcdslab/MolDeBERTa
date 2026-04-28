# MolDeBERTa

MolDeBERTa: Foundational Model for Physicochemical and Substructure-Informed Molecular Representation Learning

\[[Dataset on HuggingFace](https://huggingface.co/datasets/SaeedLab/MolDeBERTa)\] | \[[Model Collection](https://huggingface.co/collections/SaeedLab/moldeberta)\] | \[[Cite](#citation)\]

The paper is under review.

## Abstract
Foundational models that learn the "language" of molecules are essential for accelerating material and drug discovery. These self-learning models can be trained on large collections of unlabelled molecules, enabling applications such as property prediction, molecule design, and screening for specific functions. However, existing molecular language models rely on masked language modeling, a generic token-level objective that is agnostic to physicochemical and substructure molecular properties. Here we introduce MolDeBERTa, a chemistry-informed self-supervised molecular encoder built upon the DeBERTaV2 architecture with byte-level Byte-Pair Encoding (BPE) tokenization. MolDeBERTa is pretrained on up to 123 million SMILES from PubChem using three novel pretraining objectives designed to inject strong inductive biases for molecular properties and substructure similarity directly into the latent space. The model is systematically investigated across three architectural scales, two dataset sizes, and five distinct pretraining objectives, of which three are novel and two are adapted from prior work. When evaluated on 9 MoleculeNet benchmarks, MolDeBERTa achieves the best overall performance on 4 out of 9 tasks and outperforms SMILES-based encoders on 7 out of 9 tasks, with up to a 16% reduction in regression error, and improvements of up to 2.2 ROC-AUC points on classification tasks.

## System Requirements
- A computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- CUDA-enabled GPU with at least 6 GB of memory.

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)


### Fork the Repository
- Fork this repository to your own account.
- Clone your fork to your machine.

### Create a Conda Environment
```bash
cd <repository_directory>
conda env create --file environment.yml
```

### Activate the Environment
```bash
conda activate moldeberta
```

## Running the Experiments
### Pretraining
Follow the steps below to pretrain the MolDeBERTa models.

1. Train the tokenizer
```bash
cd src
python train_tokenizer.py
```
---

2. Generate molecule descriptors and features Prepare the data for MTR, MLC, and contrastive objectives:
```bash
cd src
python generate_data.py --dataset 10M
```
Arguments:
**--dataset**: Dataset size (10M or 123M)

---

3. Run the script corresponding to your desired objective:
```bash
cd src
python mlc.py --model_size tiny --dataset 10M
```
Available pretraining scripts:
* `mlm.py`: Masked Language Modeling (MLM)
* `mlc.py`: Multi-Label Classification (MLC)
* `mtr.py`: Multi-Task Regression (MTR)
* `contrastive_mlc.py`: Contrastive-based MLC
* `contrastive_mtr.py`: Contrastive-based MTR

Arguments:
* **--model_size**: Architecture size (tiny, small, or base)
* **--dataset**: Dataset size (10M or 123M)

---

### Finetuning
To finetune a pretrained model on downstream tasks:
```bash
cd src
python finetuning.py --model_path ../pretrained/moldeberta-tiny-10M-mlc --task bace-regression
```
Arguments:
* **--model_path**: Path to the model (can be a HuggingFace model or a local directory)
* **--task**: The downstream task to evaluate. Options: `bace-classification`, `bace-regression`, `bbbp`, `clearance`, `clintox`, `delaney`, `hiv`, `lipo`, `tox21`

### Explainer
To interpret model predictions and generate visualizations of atom importance:
```bash
cd src
python explainer.py --model_path "../finetuned/pretrained_moldeberta-tiny-10M-mlc-bace-regression-finetuned" --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" --target_label 0
```
Arguments:
* **--model_path**: Path to a model finetuned on a specific task (can be a HuggingFace model or a local directory)
* **--smiles**: The SMILES string of the molecule to analyze
* **--target_label**: The index of the output neuron to explain. Use `0` for regression tasks (single output). Use `0` to `n-1` for classification tasks (where `n` is the number of classes)

---

## Citation
The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).


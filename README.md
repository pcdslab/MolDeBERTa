# MolDeBERTa

MolDeBERTa: Grounding Molecular Encoders in Chemical Reality via Chemistry-Informed Pretraining

The paper is under review.

## Abstract
Encoder-based molecular transformer foundation models for SMILES strings have become the dominant paradigm for learning molecular representations, achieving substantial progress across a wide range of downstream chemical tasks. Despite these advances, most existing models rely on first-generation transformer architectures and are predominantly pretrained using masked language modeling, which is a generic objective that fails to explicitly encode physicochemical or structural information. In this work, we introduce MolDeBERTa, an encoder-based molecular framework built upon the DeBERTaV2 architecture and pretrained on large-scale SMILES data. We systematically investigate the interplay between model scale, pretraining dataset size, and pretraining objective by training 30 MolDeBERTa variants across three architectural scales, two dataset sizes, and five distinct pretraining objectives. Crucially, we introduce three novel pretraining objectives designed to inject strong inductive biases regarding molecular properties and structural similarity directly into the model's latent space. Across nine downstream benchmarks from MoleculeNet, MolDeBERTa achieves state-of-the-art performance on 7 out of 9 tasks under a rigorous fine-tuning protocol. Our results demonstrate that chemically grounded pretraining objectives consistently outperform standard masked language modeling. Finally, based on atom-level interpretability analyses, we provide qualitative evidence that MolDeBERTa learns task-specific molecular representations, highlighting chemically relevant substructures in a manner consistent with known physicochemical principles. These results establish MolDeBERTa as a robust encoder-based foundation model for chemistry-informed representation learning.

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
Below is an example command for pretraining MolDeBERTa models:

1. Train the tokenizer
```bash
cd src
python train_tokenizer.py
```
---

2. Generate molecule descriptors and structure features for MTR, MLC, and contrastive-based objectives:
```bash
cd src
python generate_data.py --dataset 10M
```
You can use the following parameters for **dataset**:
* **10M**: dataset with 10M SMILES
* **123M**: dataset with 123M SMILES

---

3. Pretrain MolDeBERTa:
```bash
cd src
python mlc.py --model_size tiny --dataset 10M
```
You can change the filename depending on pretraining objective:
* `mlm.py`: pretraining based on MLM objective
* `mlc.py`: pretraining based on MLC objective
* `mtr.py`: pretraining based on MTR objective
* `contrastive_mlc.py`: pretraining based on contrastive-based MLC objective
* `contrastive_mtr.py`: pretraining based on contrastive-based MTR objective

You can use the following parameters for **model_size**:
* **tiny**: tiny architecture
* **small**: small architecture
* **base**: base architecture

You can use the following parameters for **dataset**:
* **10M**: dataset with 10M SMILES
* **123M**: dataset with 123M SMILES
---

### Finetuning
Below is an example command for finetuning:

```bash
cd src
python finetuning.py --model_path ../pretrained/moldeberta-tiny-10M-mlc --task bace-regression
```
You can use the any model path (HuggingFace path ou local path) for **model_path**

You can use the following parameters for **task**:
* **bace-classification**: bace classification task
* **bace-regression**: bace regression task
* **bbbp**: bbbp task
* **clearance**: clearance task
* **clintox**: clintox task
* **delaney**: delaney task
* **hiv**: hiv task
* **lipo**: lipo task
* **tox21**: tox21 task

### Explainer
Below is an example command for explain the prediction, generating an image showing the atom importance for the model prediction:

```bash
cd src
python explainer.py --model_path "../finetuned/pretrained_moldeberta-tiny-10M-mlm-bace-regression-finetuned" --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" --target_label 0
```
You can use any model pretrained model on a specific task (HuggingFace path ou local path) for **model_path**

You can use any smiles string in the parameter **smiles**

You can select which is the target label in **target_label**. If is has only one target label (such as regression tasks in the finetuning evaluated tasks), you should use **0**. Otherwise, if it has multiple output neurons (such as binary or multi-classification tasks), you can use from **0** up to **n-1** (the number of output neurons minus 1)

---

## Citation
The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).


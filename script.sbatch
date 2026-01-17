#!/bin/bash 
#SBATCH --job-name=github_test
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus=1
#SBATCH --nodelist=dragon-compute-04
#SBATCH --mem-per-cpu=2gb 
#SBATCH --output=%x-%j.out 
#SBATCH --error=%x-%j.err 

echo "Job started at: $(date)" 
echo "Running on node: $(hostname)" 
echo "Current working directory: $(pwd)" 
echo "Job ID: ${SLURM_JOB_ID}" 
echo "Job Name: ${SLURM_JOB_NAME}" 

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate moldeberta
cd src
# python train_tokenizer.py
# python generate_data.py --dataset 10M
# python mlc.py --model_size tiny --dataset 10M
# python finetuning.py --model_path ../pretrained/moldeberta-tiny-10M-mlc --task bace-regression
# python explainer.py --model_path "../finetuned/pretrained_moldeberta-tiny-10M-mlc-bace-regression-finetuned" --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" --target_label 0
conda deactivate

echo "Job Finished at: $(date)"
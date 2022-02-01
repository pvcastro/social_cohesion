#!/bin/bash

#SBATCH --job-name=ONNX_conversion
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:0
#SBATCH --output=slurm_ONNX_conversion_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

MODEL_NAME=$1

module purge
#module load anaconda3/2020.02
#
#source /scratch/mt4493/twitter_labor/code/envs/env_to_tar/inference_2021_env/bin/activate

#echo "pyenv activated"

singularity exec --overlay /scratch/mt4493/twitter_labor/code/envs/singularity/inference/env.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /scratch/mt4493/training_prosociality/code/twitter_social_preferences/code/training_impact/classification/deployment/onnx_model_conversion.py \
      --model_name ${MODEL_NAME}"


#!/bin/bash
#SBATCH --job-name=training_berts_nigeria
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_training_berts_nigeria_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=mt4493@nyu.edu

MODEL_TYPE=$1
##WITH_NON_INEQ_US=$3

TIMESTAMP=$(date +%s)
JOB_ID=${SLURM_JOB_ID}

if [[ ${MODEL_TYPE} == "bert" ]]; then
  MODEL_NAME="DeepPavlov/bert-base-cased-conversational"
elif [[ ${MODEL_TYPE} == "bigbird" ]]; then
  MODEL_NAME="google/bigbird-roberta-base"
fi

#echo ${MODEL_TYPE}
#echo ${MODEL_NAME}

SLASH=/
if [[ ${MODEL_NAME} == *"/"* ]]; then
  MODEL_NAME_WITHOUT_SLASH=${MODEL_NAME//[${SLASH}]/-}
else
  MODEL_NAME_WITHOUT_SLASH=${MODEL_NAME}
fi


# Load packages and activate virtual environment
module purge

#create output folder
mkdir /scratch/mt4493/nigeria/trained_models/"${MODEL_NAME_WITHOUT_SLASH}"_"v1"_"${TIMESTAMP}"
OUTPUT_DIR=/scratch/mt4493/ineq/trained_models/${MODEL_NAME_WITHOUT_SLASH}_v1_${TIMESTAMP}
SINGULARITY_PATH=/scratch/mt4493/twitter_labor/code/envs/singularity/finetuning_bert/env.ext3

echo "Output folder created: ${OUTPUT_DIR}"

cd /scratch/mt4493/nigeria/code/social_cohesion/hate_speech/model_training

echo '***********************STARTING TRAINING***************************************************'
singularity exec --nv \
      --bind /home/mt4493/resolv.conf:/etc/resolv.conf:ro \
      --overlay ${SINGULARITY_PATH}:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 train_bert.py \
  --timestamp ${TIMESTAMP} \
  --output_dir ${OUTPUT_DIR}"

rm -rf ${OUTPUT_DIR}/models/!(best_model)
echo '***********************DONE TRAINING *******************************************************'
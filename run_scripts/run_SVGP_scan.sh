#!/bin/bash
#SBATCH --time=6:00:00 # walltime format is h:mm:ss.
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=PIS0029
#SBATCH --mail-user=felice.27@osu.edu
#SBATCH --mail-type=all
#SBATCH --output=SVGP_parameter_testing.out
#SBATCH --cluster=ASCEND
module reset
source /users/PAS2038/felice27/miniconda3/bin/activate
conda activate ldiaml
#conda activate rapids-22.10
module load cuda/11.8.0
num=$CUDA_VISIBLE_DEVICES
echo "USING NODE: ${num}"
set -x
cd $SLURM_SUBMIT_DIR
date +"%T"
python -u Hyperparameter_SVGP.py
date +"%T"
seff $SLURM_JOB_ID
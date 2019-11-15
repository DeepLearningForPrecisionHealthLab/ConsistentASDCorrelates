#!/bin/bash

#SBATCH --job-name=AugmentedDenseNet
#SBATCH --partition=GPUv100s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5-12:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_Dense_Augmented.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

module load parallel

# SRUN arguments
CORES_PER_TASK=72

# Load and setup task
module load python/3.6.4-anaconda
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAutismTF_GPU_v5

python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/DataAugmenter.py 3

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
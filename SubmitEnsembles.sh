#!/bin/bash

#SBATCH --job-name=EnsemblingModels
#SBATCH --partition=256GBv1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_Ensembles.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

module load parallel

# SRUN arguments
CORES_PER_TASK=56

# Setup to run
module load python/3.6.4-anaconda
cd /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAuttfGPUv4

# Run the file
python EnsemblingModels.py

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
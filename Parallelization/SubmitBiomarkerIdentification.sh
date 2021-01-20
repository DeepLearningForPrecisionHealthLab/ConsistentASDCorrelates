#!/bin/bash

#SBATCH --job-name=BiomarkerID_64x
#SBATCH --partition=GPU4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=16-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_Biomarker_Identification.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

module load parallel

# Setup to run
module load python/3.6.4-anaconda
cd /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAutismTF_GPU_v5

# Run the file
python BioMarkerIdentification.py

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
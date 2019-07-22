#!/bin/bash

#SBATCH --job-name=ShallowML_LSGC
#SBATCH --partition=256GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_ShallowML.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

# load the right modules
module load python/3.6.4-anaconda
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAuttfGPUv3

#run the script
python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/IMPAC_Autism_Run_multiple_test.py

#echo the time
now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
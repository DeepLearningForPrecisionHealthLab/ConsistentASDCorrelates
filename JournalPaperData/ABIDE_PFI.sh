#!/bin/bash
#SBATCH --job-name=ABIDE_PFI
#SBATCH --partition=GPUA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_ABIDE_PFI.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

#run the script
/project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAutismTF_GPU_v5/bin//python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/BioMarkerIdentification.py

#echo the time
now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
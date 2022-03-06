#!/bin/bash
#SBATCH --job-name=ABIDE_PFI_array
##SBATCH --partition=384GB,256GBv1,256GB,128GB
#SBATCH --partition=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_ABIDE_PFI_array.txt"
#SBATCH --array=[0-29]%30

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

#run the script
/project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAutismTF_GPU_v5/bin//python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/ABIDE_PFI.py $SLURM_ARRAY_TASK_ID

#echo the time
now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
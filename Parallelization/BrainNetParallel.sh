#!/bin/bash

#SBATCH --job-name=BrainNet_LSGC
#SBATCH --partition=GPUp100
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=20-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_BrainNet_MultiConn.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

module load parallel

# SRUN arguments
CORES_PER_TASK=56

INPUTS_COMMAND="ls -vr ./IniFiles/B* | shuf"

TASK_SCRIPT='python script_wrapper_brainnet.py'

SRUN_CMD="srun --exclusive -N1 -n1 -c $CORES_PER_TASK"
PARALLEL_CMD="parallel --delay .2 -j $SLURM_NTASKS --joblog Logs/$SLURM_JOB_ID.task.log"

# set up the lockfiles, it's important to do it here so it only gets done once!
module load caffe/1.0.0-segnet
export PYTHONPATH=/cm/shared/apps/caffe/1.1.0.0_segnet/caffe/python:$PYTHONPATH
module load python/2.7.14-anaconda
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperBrainNetProcessing
LOCK_DIR="Locks/"$SLURM_JOB_ID'_lock_files'
eval "python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/filelock_utilities.py " $SLURM_JOB_NODELIST $LOCK_DIR
# wait for the above to finish
wait
echo "Lock files completed, running the networks"
# Run the jobs
eval $INPUTS_COMMAND | $PARALLEL_CMD $SRUN_CMD $TASK_SCRIPT $LOCK_DIR {}

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job End: $now"
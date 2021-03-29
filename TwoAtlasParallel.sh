#!/bin/bash

#SBATCH --job-name=pairwise_IMPAC_features_02_07_2019
#SBATCH --partition=GPUp100
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=4-00:00:00
#SBATCH --workdir="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization"
#SBATCH --output="/project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/Logs/log_%j_Dense.txt"

now=$(date +"%m-%d-%Y,%H:%M:%S")
echo "Job start: $now"

keys='basc064_basc122, basc064_basc197, basc064_craddock_scorr_mean, basc064_harvard_oxford_cort_prob_2mm, basc064_msdl, basc064_power_2011, anatomical_basc064_basc122, basc122_basc197, basc122_craddock_scorr_mean, basc122_harvard_oxford_cort_prob_2mm, basc122_msdl, basc122_power_2011, anatomical_basc064_basc197, anatomical_basc122_basc197, basc197_craddock_scorr_mean, basc197_harvard_oxford_cort_prob_2mm, basc197_msdl, basc197_power_2011, anatomical_basc064_craddock_scorr_mean, anatomical_basc122_craddock_scorr_mean, anatomical_basc197_craddock_scorr_mean, craddock_scorr_mean_harvard_oxford_cort_prob_2mm, craddock_scorr_mean_msdl, craddock_scorr_mean_power_2011, anatomical_basc064_harvard_oxford_cort_prob_2mm, anatomical_basc122_harvard_oxford_cort_prob_2mm, anatomical_basc197_harvard_oxford_cort_prob_2mm, anatomical_craddock_scorr_mean_harvard_oxford_cort_prob_2mm, harvard_oxford_cort_prob_2mm_msdl, harvard_oxford_cort_prob_2mm_power_2011, anatomical_basc064_msdl, anatomical_basc122_msdl, anatomical_basc197_msdl, anatomical_craddock_scorr_mean_msdl, anatomical_harvard_oxford_cort_prob_2mm_msdl, msdl_power_2011, anatomical_basc064_power_2011, anatomical_basc122_power_2011, anatomical_basc197_power_2011, anatomical_craddock_scorr_mean_power_2011, anatomical_harvard_oxford_cort_prob_2mm_power_2011, anatomical_msdl_power_2011'

modelnums='01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49'

module load parallel

for modelnum in $modelnums
    do
    for key in $keys
    do
        # SRUN arguments
        CORES_PER_TASK=56

        echo "running $key$modelnum"

        cp ./IniFiles/Dense_$modelnum.ini ./IniFiles/Temp/$key$modelnum.ini

        INPUTS_COMMAND="ls -v ./IniFiles/Temp/$key$modelnum.ini"

        TASK_SCRIPT='python script_wrapper_dense.py'

        SRUN_CMD="srun --exclusive -N1 -n1 -c $CORES_PER_TASK"
        PARALLEL_CMD="parallel --delay .2 -j $SLURM_NTASKS --joblog Logs/$SLURM_JOB_ID.task.log"

        # set up the lockfiles, it's important to do it here so it only gets done once!
        module load python/3.6.4-anaconda
        source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/CooperAuttfGPU
        LOCK_DIR="Locks/"$SLURM_JOB_ID'_lock_files'
        eval "python /project/bioinformatics/DLLab/Cooper/Code/AutismProject/Parallelization/filelock_utilities.py " $SLURM_JOB_NODELIST $LOCK_DIR
        # wait for the above to finish
        wait
        echo "Lock files completed, running the networks"
        # Run the jobs
        eval $INPUTS_COMMAND | $PARALLEL_CMD $SRUN_CMD $TASK_SCRIPT $LOCK_DIR {}

        now=$(date +"%m-%d-%Y,%H:%M:%S")
        done
    done
echo "Job End: $now"

import os
import sys
import subprocess
sys.path.append('/project/bioinformatics/DLLab/shared/Lab_Utilities') #Add to the path where the filelock_utilities are
import filelock_utilities
import time

# sys.argv[1] = relative path to the folder containing the lock files. DO NOT CHANGE THIS!
# sys.argv[2] = the file to run
# the working directory will be the set workdir from your slurm script

print('The conda environment for the wrapper: %s'%os.environ['CONDA_DEFAULT_ENV'])
lock_folder = sys.argv[1]
# sets and locks an available GPU
gpu_to_use = filelock_utilities.use_and_lock_gpu(lock_path=lock_folder)
print('The GPU device that is being set is %s'%gpu_to_use)

# run the code with all the system arguments after the first.
subprocess.call('python IMPAC_DenseNetwork.py %s'%(' '.join(sys.argv[2:])), shell=True)

#release the GPU
filelock_utilities.unlock_gpu(gpu_to_use, lock_path=lock_folder)
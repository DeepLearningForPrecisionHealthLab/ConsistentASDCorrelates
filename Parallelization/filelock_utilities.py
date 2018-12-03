"""
Author: Alex Treacher
Date: 08/02/18

These functions are designed to allow multiple jobs get sent to a node with multiple GPU cards.
These functions allow jobs to run in parallel and use dedicated a GPU card.
This stops the jobs from attempting to run on the same card and fighting for resources (which can cause errors)
Example usage:

gpu_to_use = use_and_lock_gpu()
try:
    run your code here
except:
    pass
unlock_gpu(gpu_to_use)
"""


import os
import sys
from filelock import FileLock
import platform

# The number of cards per node, this needs to be changes if the grid changes!
# This should not contain any non-gpu nodes as they are used for testing
cards_per_node = {
                  'Nucleus042': 1,
                  'Nucleus043': 1,
                  'Nucleus044': 1,
                  'Nucleus045': 1,
                  'Nucleus046': 1,
                  'Nucleus047': 1,
                  'Nucleus048': 1,
                  'NucleusC006': 1,
                  'NucleusC018': 1,
                  'Nucleus162': 2,
                  'Nucleus163': 2,
                  'Nucleus164': 2,
                  'Nucleus165': 2,
                  'Nucleus166': 2,
                  'Nucleus167': 2,
                  'Nucleus168': 2,
                  'Nucleus169': 2,
                  'Nucleus170': 2,
                  'Nucleus171': 2,
                  'Nucleus172': 2,
                  'Nucleus173': 2,
                  }

cards_per_node = {
                 'Nucleus042':1,
                 'Nucleus043':1,
                 'Nucleus044':1,
                 'Nucleus045':1,
                 'Nucleus046':1,
                 'Nucleus047':1,
                 'Nucleus048':1,
                 'Nucleus049':1,
                 'NucleusC019':1,
                 'NucleusC021':1,
                 'NucleusC026':1,
                 'NucleusC002':1,
                 'NucleusC003':1,
                 'NucleusC004':1,
                 'NucleusC005':1,
                 'NucleusC006':1,
                 'NucleusC007':1,
                 'NucleusC008':1,
                 'NucleusC009':1,
                 'NucleusC010':1,
                 'NucleusC011':1,
                 'NucleusC012':1,
                 'NucleusC013':1,
                 'NucleusC014':1,
                 'NucleusC015':1,
                 'NucleusC016':1,
                 'NucleusC017':1,
                 'NucleusC018':1,
                 'NucleusC019':1,
                 'NucleusC020':1,
                 'NucleusC021':1,
                 'NucleusC022':1,
                 'NucleusC023':1,
                 'NucleusC024':1,
                 'NucleusC025':1,
                 'NucleusC026':1,
                 'NucleusC027':1,
                 'NucleusC028':1,
                 'NucleusC029':1,
                 'NucleusC030':1,
                 'NucleusC031':1,
                 'NucleusC032':1,
                 'NucleusC033':1,
                 'Nucleus162':2,
                 'Nucleus163':2,
                 'Nucleus164':2,
                 'Nucleus165':2,
                 'Nucleus166':2,
                 'Nucleus167':2,
                 'Nucleus168':2,
                 'Nucleus169':2,
                 'Nucleus170':2,
                 'Nucleus171':2,
                 'Nucleus172':2,
                 'Nucleus173':2,


                 }


lock_folder='/project/bioinformatics/DLLab/shared/LockFiles'

def fPadZeros(sNuc):
    """
    Pads a Nucleus name to length 3, eg. makes
    NucleusC26 -> NucleusC026
    NucleusC1 -> NucleusC001
    :param sNum: numerical string
    :return: numerical string padded with preceding zeros to a len of 3
    """
    if "C" in sNuc:
        iTargetLen=11
    else:
        iTargetLen=10

    while len(sNuc)<iTargetLen:
        sNuc=sNuc[0:8]+'0'+sNuc[8:]
    return sNuc

def make_gpu_filelock(node_name=platform.node(), num_gpu=None, lock_path=lock_folder, overwrite=False):
    """
    This sets up the lock files for the node.
    Make sure you know the number of GPU cards; otherwise it will lead it resource sharing or inefficient running
    :param node_name: (str) The name of the node. EG Nucleus167
    :param num_gpu: (int) The number of GPU cards, if None the number of cards will be looked up from the hard coded list
    :param lock_path: (str) the path to the folder containing the lock files)
    :param overwrite: (bool) If the files exist, overwrite.
    :return: None
    """
    node_name=fPadZeros(node_name)

    lock_file = os.path.join(lock_path,'%s.txt.lock'%node_name)
    print(lock_file, node_name)
    txt_file = os.path.join(lock_path, '%s.txt'%node_name)
    if (not overwrite) and (os.path.exists(lock_file) and os.path.exists(txt_file)):
        print('Already made')
    else:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        if os.path.exists(txt_file):
            os.remove(txt_file)
        if num_gpu==None:
            try:
                num_gpu=cards_per_node[node_name]
            except KeyError:
                raise ValueError('%s is not included in the harded GPU list, please update this in filelock_utilities.py or input num_gpus'%node_name)
        fileinput='0'*num_gpu
        print('making lockfile %s'%lock_file)
        with FileLock(lock_file):
            with open(txt_file, 'w') as f:
                f.write(fileinput)

def use_and_lock_gpu(node_name=platform.node(), lock_path=lock_folder):
    """
    This uses the lock file to reserve and set a GPU card that is not in use.
    WARNING: be sure to deactivate it using unlock_gpu once you're code has run!
    :param node_name: (str) The node to reserve. By default it is set to the node the script is running on. This can be found with platform.node
    :param lock_path: (str) The path to the lock folder
    :return: (int) The gpu index to use, this allows for deactivation later
    """
    node_name = fPadZeros(node_name)

    print('The conda environment for the filelock_utilties: %s'%os.environ['CONDA_DEFAULT_ENV'])
    # lock_file = os.path.join(lock_path, '%s.txt.lock' % node_name)
    # txt_file = os.path.join(lock_path, '%s.txt' % node_name)
    lock_file = os.path.join(lock_path,'%s.txt.lock'%node_name)
    txt_file = os.path.join(lock_path, '%s.txt'%node_name)
    with FileLock(lock_file):
        with open(os.path.join(os.getcwd(), txt_file), 'r+') as f:
            line = f.readline()
            for i, value in enumerate(line):
                if int(value) == 0:
                    gpu_index = i
                    line = line[:gpu_index]+'1'+line[gpu_index+1:]
                    f.seek(0)
                    f.truncate()
                    f.write(line)
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                    return gpu_index
    #if it didnt finish in the loop no cards are available
    raise IndexError('All gpus currently taken, too may jobs have been submitted to this node\n'
                     'Change in the .sh submit file:\n#SRUN arguments: '
                     'CORES_PER_TASK=number of cpu nodes divided by n tasks')

def unlock_gpu(gpu_index, node_name=platform.node(), lock_path=lock_folder):
    """
    This unlocks the GPU for use of the next job
    :param gpu_index:(int) the index of the GPU to unlock, usually 0 or 1
    :param node_name:(str) Name of the node to unlock the GPU on
    :param lock_path:(str) Location of the folder containing the lock file
    :return:
    """
    lock_file = os.path.join(lock_path, '%s.txt.lock' % node_name)
    txt_file = os.path.join(lock_path, '%s.txt' % node_name)
    with FileLock(lock_file):
        with open(txt_file, 'r+') as f:
            line = f.readline()
            line = line[:gpu_index]+'0'+line[gpu_index+1:]
            f.seek(0)
            f.truncate()
            f.write(line)

if __name__ == "__main__":
    """
    This will set up the lock files for the list of nodes, commonly used when using srun or parallel
    #sys.argv[1] = the node list
    #sys.argv[2] = the file to make the lock files
    """
    import re
    #get the system arguments
    node_str = sys.argv[1]
    lock_path = sys.argv[2]
    print(node_str)
    #make sure the lock_path exists
    if not os.path.exists(lock_path):
        os.makedirs(lock_path)

    #parse the node list from #SLURM_JOB_NODELIST
    node_prefix_search = re.findall('([A-Za-z0-9]+)\[', node_str)
    multinode_search = re.findall(r'(?:\[?)([0-9]+[\-]*[0-9]*)[\,|\]]+', node_str)
    #if there is only one node
    if node_prefix_search.__len__() == 0:
        node_list = [node_str]
    #for multiple nodes
    else:
        node_list = []
        node_prefix = node_prefix_search[0]
        for node_range_str in multinode_search:
            node_numbers = [int(x) for x in node_range_str.split('-')]
            # for a range of nodes in the list
            if node_numbers.__len__() == 2:
                number_list = list(range(node_numbers[0], node_numbers[1]+1))
                for i in number_list:
                    node_list.append("%s%03i"%(node_prefix, i))
            # for single nodes in the list
            elif node_numbers.__len__() == 1:
                node_list.append("%s%s"%(node_prefix, node_numbers[0]))
            #this shouldn't happen!
            else:
                raise ValueError('Issue with the node range string %s'%node_numbers)

    #make the filelocks
    for node in node_list:
        testing = True
        if not node in cards_per_node.keys():
            #used for testing on the non-gpu nodes
            make_gpu_filelock(node_name=node, lock_path=lock_path, num_gpu=2)
        else:
            num_gpu = cards_per_node[node]
            make_gpu_filelock(node_name=node, lock_path=lock_path, num_gpu=num_gpu)

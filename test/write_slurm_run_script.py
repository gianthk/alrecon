#!/usr/bin/env python
import os

import solara

job_name = 'BEATS_recon_pippo'
h5file = 'pippo.h5'
recon_dir = 'pippo/recon/'
work_dir = 'pippo/'
COR = 55

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

class slurmjob:
    def __init__(self, job_name=None, job_dir=None, ntasks=48, cpus_per_task=2, max_time_min=30, partition='cpu', mem_per_cpu=2, max_threads=96, recon_script='BEATS_recon.py'):
        self.job_name = job_name
        self.job_dir = job_dir
        self.job_file = os.path.join(self.job_dir, "%s.job" % self.job_name)
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.max_time_min = max_time_min
        self.partition = partition
        self.mem_per_cpu = mem_per_cpu
        self.max_threads = max_threads
        self.recon_script = recon_script

    def write_header(self):
        with open(self.job_file, "x") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name={0}%j\n".format(self.job_name))
            fh.writelines("#SBATCH --output=.out{0}%j.out\n".format(self.job_name))
            fh.writelines("#SBATCH --error=.out/{0}%j.err\n".format(self.job_name))
            fh.writelines("#SBATCH --ntasks={0}\n".format(self.ntasks))
            fh.writelines("#SBATCH --cpus-per-task={0}\n".format(self.cpus_per_task))
            fh.writelines("#SBATCH --time=00:{0}:00\n".format(self.max_time_min))
            fh.writelines("#SBATCH --partition={0}\n".format(self.partition))
            fh.writelines("#SBATCH --mem-per-cpu={0}\n\n".format(self.mem_per_cpu))
            # fh.writelines("#SBATCH --qos=normal\n")
            # fh.writelines("#SBATCH --mail-type=ALL\n")
            # fh.writelines("#SBATCH --mail-user=$USER@sesame.org.jo\n")

            fh.writelines("# Modules section:\n")
            fh.writelines("ml load anaconda/tomopy\n\n")
            fh.writelines("# Variables section:\n")
            fh.writelines("export NUMEXPR_MAX_THREADS={0}\n\n".format(self.max_threads))

    def write_python_command(self, alrecon_state):
        '''Assemble reconstruction python command string'''
        py_command = ('python {0} {1} '
                      '--recon_dir {2} '
                      '--work_dir {3} '
                      '--cor {4}\n'.format(self.recon_script, alrecon_state.h5file.value, alrecon_state.recon_dir.value, os.path.dirname(alrecon_state.recon_dir.value), alrecon_state.COR.value))
        
        if alrecon_state.cluster_sino.value:



# job_directory = "%s/.job" % os.getcwd()
job_directory = "%s/pippo" % os.getcwd()
# scratch = os.environ['SCRATCH']
# data_dir = os.path.join(scratch, '/project/LizardLips')

# Make top level directories
mkdir_p(job_directory)
# mkdir_p(data_dir)

job_file = os.path.join(job_directory, "%s.job" % job_name)
# job_data = os.path.join(data_dir, job_name)

# Create job directories
# mkdir_p(job_data)

# assemble python command string
py_command = ('python BEATS_recon.py {0} '
              '--recon_dir {1} '
              '--work_dir {2} '
              '--cor {3}\n'.format(h5file, recon_dir, work_dir, COR))

# write job file
with open(job_file, "x") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH --job-name={0}%j\n".format(job_name))
    fh.writelines("#SBATCH --output=.out{0}%j.out\n".format(job_name))
    fh.writelines("#SBATCH --error=.out/{0}%j.err\n".format(job_name))
    fh.writelines("#SBATCH --ntasks=48\n")
    fh.writelines("#SBATCH --cpus-per-task=2\n")
    fh.writelines("#SBATCH --time=00:30:00\n")
    fh.writelines("#SBATCH --partition=cpu\n")
    fh.writelines("#SBATCH --mem-per-cpu=2\n\n")
    # fh.writelines("#SBATCH --qos=normal\n")
    # fh.writelines("#SBATCH --mail-type=ALL\n")
    # fh.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")

    fh.writelines("# Modules section:\n")
    fh.writelines("ml load anaconda/tomopy\n\n")
    fh.writelines("# Variables section:\n")
    fh.writelines("export NUMEXPR_MAX_THREADS=96\n\n")

    fh.writelines(py_command)

# launch slurm job
# os.system("sbatch %s" % job_file)
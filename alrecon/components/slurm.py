"""
Class for handling of slurm job file creation within alrecon.
For more information, visit the project homepage:
	https://github.com/gianthk/alrecon
"""

import os
import logging

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

class slurmjob:
    def __init__(self, job_name='', job_dir=''):
        self.job_name = job_name
        self.job_dir = job_dir
        self.job_file = os.path.join(self.job_dir, "%s.job" % self.job_name)
        self.recon_command = ''

    def write_header(self, alrecon_state):
        '''Add here write_header calls for other compute nodes.'''
        if alrecon_state.node.value == 'rum':
            self.write_header_rum(alrecon_state)
        else:
            logging.error('Compute node {0} not known.'.format(alrecon_state.node))

    def write_header_rum(self, alrecon_state):
        '''
        Writes header to slurm job file for the SESAME BEATS beamline.
        SBATCH settings and module load call are specific to the HPC cluster rum.sesame.org.jo.
        If required for integration with a different cluster, for example if you different module load statements, you should start by creating a copy of this method.
        '''
        with open(self.job_file, "w") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name={0}_%j\n".format(self.job_name))
            fh.writelines("#SBATCH --output={0}_%j.out\n".format(self.job_name))
            fh.writelines("#SBATCH --error={0}_%j.err\n".format(self.job_name))
            fh.writelines("#SBATCH --ntasks={0}\n".format(alrecon_state.ntasks))
            fh.writelines("#SBATCH --cpus-per-task={0}\n".format(alrecon_state.cpus_per_task))
            fh.writelines("#SBATCH --time={0}\n".format(alrecon_state.max_time_min))
            fh.writelines("#SBATCH --partition={0}\n".format(alrecon_state.partition.value))
            fh.writelines("#SBATCH --nodelist={0}\n".format(alrecon_state.nodelist.value))
            fh.writelines("#SBATCH --gres={0}\n".format(alrecon_state.gres.value))
            fh.writelines("#SBATCH --mem-per-cpu={0}\n\n".format(alrecon_state.mem_per_cpu.value))
            # fh.writelines("#SBATCH --qos=normal\n")
            # fh.writelines("#SBATCH --mail-type=ALL\n")
            # fh.writelines("#SBATCH --mail-user=$USER@sesame.org.jo\n")

            fh.writelines("# Modules section:\n")
            fh.writelines("ml load anaconda/tomopy\n\n")
            fh.writelines("# Variables section:\n")
            fh.writelines("export NUMEXPR_MAX_THREADS={0}\n\n".format(alrecon_state.max_threads))

    def write_recon_command(self, alrecon_state):
        with open(self.job_file, "a") as fh:
            fh.writelines("{0}\n\n".format(self.recon_command))

    def set_recon_command(self, alrecon_state):
        '''Set reconstruction python command.
            The default command considers a reconstruction call on rum.sesame.org.jo using the python script BEATS_recon.py.
            You can add set_recon_command calls tailored to different scripts.
        '''

        if 'BEATS_recon.py' in alrecon_state.recon_script.value:
            self.set_recon_command_beats(alrecon_state)
        else:
            logging.error('Script {0} not known.'.format(alrecon_state.recon_script))

    def set_recon_command_beats(self, alrecon_state):
        '''Assemble reconstruction python command for reconstruction call on rum.sesame.org.jo using the python script BEATS_recon.py.
            Parameters
            ----------
            alrecon_state : alrecon
                Alrecon application state. Contains the reconstruction settings as solara reactive state variables.
            '''

        py_command = ('python {0} {1} --recon_dir {2} --work_dir {3} --cor {4} --ncore {5} --algorithm {6}'.format(alrecon_state.recon_script.value, alrecon_state.h5file.value, alrecon_state.recon_dir.value, os.path.dirname(alrecon_state.recon_dir.value), alrecon_state.COR.value, alrecon_state.ncore.value, alrecon_state.algorithm.value))

        # Add projections range argument
        if alrecon_state.recon_proj_range.value:
            py_command = (py_command + (' --proj {0} {1} {2}'.format(alrecon_state.proj_range.value[0], alrecon_state.proj_range.value[1], 1)))

        # Add sinogram range argument
        if alrecon_state.recon_sino_range.value:
            py_command = (py_command + (' --sino {0} {1} {2}'.format(alrecon_state.sino_range.value[0], alrecon_state.sino_range.value[1], 1)))

        # Add phase retrieval arguments
        if alrecon_state.phase_object.value:
            py_command = (py_command + (' --phase --alpha {0} --pixelsize {1} --energy {2} --sdd {3}'.format(alrecon_state.alpha.value, 1e-3*alrecon_state.pixelsize.value, alrecon_state.energy.value, alrecon_state.sdd.value)))

            if alrecon_state.pad.value:
                py_command = py_command + ' --phase_pad'
            else:
                py_command = py_command + ' --no-phase_pad'

        # Add stripe removal arguments
        if alrecon_state.stripe_remove.value:
            py_command = (py_command + (
                ' --stripe_method {0} --snr {1} --size {2} --drop_ratio {3} --dim {4} --la_size {5} --sm_size {6}'.format(alrecon_state.stripe_removal_method.value, alrecon_state.snr.value, alrecon_state.size.value, alrecon_state.drop_ratio.value, alrecon_state.dim.value, alrecon_state.la_size.value, alrecon_state.sm_size.value)))

            if alrecon_state.norm.value:
                py_command = py_command + ' --norm'
            else:
                py_command = py_command + ' --no-norm'

        # Add uint data conversion arguments
        if alrecon_state.uintconvert.value:
            py_command = py_command + (' --dtype {0} --data_range {1} {2}'.format(alrecon_state.bitdepth.value, alrecon_state.Data_min.value, alrecon_state.Data_max.value))

        # Add circular mask arguments
        if alrecon_state.circmask.value:
            py_command = py_command + (' --circ_mask --circ_mask_ratio {0}'.format(alrecon_state.circmask_ratio.value))

        # Add midplanes arguments
        if alrecon_state.write_midplanes.value:
            py_command = py_command + ' --midplanes'

        # set the reconstruction command
        self.recon_command = py_command
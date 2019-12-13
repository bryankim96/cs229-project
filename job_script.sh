#!/bin/bash
# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=gputest1
# Get email notification when job finishes or fails
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=ematsu@stanford.edu
# Define how long you job will run d-hh:mm:ss
#SBATCH --time 05:00:00
# GPU jobs require you to specify partition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
# Number of tasks
#SBATCH --cpus=4 
##SBATCH --cpus-per-task=8

# ntasks=20

# srun python main.py 

# define and create a unique scratch directory
# SCRATCH_DIRECTORY=/global/work/${USER}/job-array-example/${SLURM_JOBID}
# mkdir -p ${SCRATCH_DIRECTORY}
# cd ${SCRATCH_DIRECTORY}

# cp ${SLURM_SUBMIT_DIR}/main.py ${SCRATCH_DIRECTORY}

# each job will see a different ${SLURM_ARRAY_TASK_ID}

source ../.venv/cs229/bin/activate

echo "now processing task.. "
python test_gpu.py &> joboutput.txt

# after the job is done we copy our output back to $SLURM_SUBMIT_DIR
# cp output.txt ${SLURM_SUBMIT_DIR}

# we step out of the scratch directory and remove it
# cd ${SLURM_SUBMIT_DIR}
# rm -rf ${SCRATCH_DIRECTORY}

# happy end
exit 0

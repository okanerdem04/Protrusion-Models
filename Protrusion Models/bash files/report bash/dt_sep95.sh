#!/bin/bash

#SBATCH --job-name=pm_dt_sep95
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=16G
#SBATCH --account=chem036964

## Direct output to the following files.
## (The %j is replaced by the job id.)
#SBATCH -o '%x'.txt

# Ensure that the MPI module is loaded
echo "This jobs runs on the following machine:"
echo "${SLURM_JOB_NODELIST}"
printf "\n"

python ./main.py 175 2.0 1.0 50 24 100 "./reportdatadensity/dt_sep95_1.out" -0.5 380
python ./main.py 175 2.0 1.0 50 24 100 "./reportdatadensity/dt_sep95_2.out" -0.5 380
python ./main.py 175 2.0 1.0 50 24 100 "./reportdatadensity/dt_sep95_3.out" -0.5 380
python ./main.py 175 2.0 1.0 50 24 100 "./reportdatadensity/dt_sep95_4.out" -0.5 380
python ./main.py 175 2.0 1.0 50 24 100 "./reportdatadensity/dt_sep95_5.out" -0.5 380





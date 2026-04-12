#!/bin/bash

#SBATCH --job-name=pm_st_sigma4
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

python ./main.py 175 2.0 1.0 50 16 100 "./reportdatasigma/sigma4_1.out" -0.5 220 
python ./main.py 175 2.0 1.0 50 16 100 "./reportdatasigma/sigma4_2.out" -0.5 220
python ./main.py 175 2.0 1.0 50 16 100 "./reportdatasigma/sigma4_3.out" -0.5 220
python ./main.py 175 2.0 1.0 50 16 100 "./reportdatasigma/sigma4_4.out" -0.5 220
python ./main.py 175 2.0 1.0 50 16 100 "./reportdatasigma/sigma4_5.out" -0.5 220





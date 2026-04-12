#!/bin/bash

#SBATCH --job-name=pm_sc_a0.5_l1_j0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
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

python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_0.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_2.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_3.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_4.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_5.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_6.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_7.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_8.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_9.out" 0
python ./main.py 175 0.5 1.0 50 24 100 "./reportdatasinglecell/sc_a0.5_l1.0_j0_10.out" 0



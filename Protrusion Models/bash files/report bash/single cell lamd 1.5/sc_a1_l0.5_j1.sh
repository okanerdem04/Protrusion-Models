#!/bin/bash

#SBATCH --job-name=pm_sc_a1.0_l1.5_j1
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

python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_0.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_2.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_3.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_4.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_5.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_6.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_7.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_8.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_9.out" -1
python ./main.py 175 1.0 1.5 50 24 100 "./reportdatasinglecell/sc_a1.0_l1.5_j1_10.out" -1



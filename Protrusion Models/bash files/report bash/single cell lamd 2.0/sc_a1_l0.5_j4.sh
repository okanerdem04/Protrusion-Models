#!/bin/bash

#SBATCH --job-name=pm_sc_a1.0_l2_j4
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

python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_0.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_2.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_3.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_4.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_5.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_6.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_7.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_8.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_9.out" -4
python ./main.py 175 1.0 2 50 24 100 "./reportdatasinglecell/sc_a1.0_l2_j4_10.out" -4



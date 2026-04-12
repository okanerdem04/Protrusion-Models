#!/bin/bash

#SBATCH --job-name=pm_sc_a1.5_l1.5_j2
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

python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_0.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_2.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_3.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_4.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_5.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_6.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_7.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_8.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_9.out" -1.33
python ./main.py 175 1.5 1.5 50 24 100 "./reportdatasinglecell/sc_a1.5_l1.5_j2_10.out" -1.33



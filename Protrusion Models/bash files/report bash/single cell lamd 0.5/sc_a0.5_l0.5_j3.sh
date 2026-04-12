#!/bin/bash

#SBATCH --job-name=pm_sc_a0.5_l0.5_j3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
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

python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_0.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_2.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_3.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_4.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_5.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_6.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_7.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_8.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_9.out" -6
python ./main.py 175 0.5 0.5 50 24 100 "./reportdatasinglecell/sc_a0.5_l0.5_j3_10.out" -6



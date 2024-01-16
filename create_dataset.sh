#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=staging
#SBATCH --time=10:00:00

module load 2023
module load CUDA/12.1.1

cp -r $HOME/data_set_creation/simulated_raw_374 $TMPDIR

srun python $HOME/data_set_creation/splitter.py

cp -r $TMPDIR/data_prepped /scratch-shared/$USER/dataset
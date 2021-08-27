#!/bin/bash

#SBATCH --account=def-your-account
#SBATCH --job-name=feps_model
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=your-email-address
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output "out/feps_model_%A.out"

cp ../data.tar $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/data.tar -C $SLURM_TMPDIR/ 
cd $SLURM_TMPDIR

python3 /path/main.py --path_0 "/path/data_FEPS_0.json" --path_1 "/path/data_FEPS_1.json" --seed 42 --model "whole-brain" --reg "lasso"

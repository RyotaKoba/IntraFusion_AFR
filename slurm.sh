#!/bin/bash
#SBATCH --job-name=OTFusionPruning
#SBATCH --partition=a6000_ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=result2.txt
#SBATCH --error=error2.txt
#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv rkoba_pruning.sif sh start.sh

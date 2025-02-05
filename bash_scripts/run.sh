#!/bin/bash

#SBATCH --job-name=mpi_python_job
#SBATCH --account=def-someuser
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=mpi_python_job_%j.out

# Load required modules
module load python/3.9
module load openmpi
module load mpi4py

# Run your Python script
mpirun -n 6 python your_script.py

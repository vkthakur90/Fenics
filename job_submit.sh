#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --job-name=dolfinx_job
#SBATCH --output=dolfinx_%j.out
#SBATCH --error=dolfinx_%j.err
        
module load OpenMPI/4.1.1-GCC-10.3.0       

# Run the job
srun --mpi=pmix singularity exec ~/fenics/dolfinx.sif python3 diffusion.py


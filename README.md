Scripts which interface with TRIQS to perform DMFT calculations.

To run TRIQS on the cluster:
 - module load singularity
 - singularity pull docker://flatironinstitute/triqs
   - This step only needs to be done once. It creates triqs_latest.sif, which should be put in a convenient location.
 - mpirun -np <number of processes> singularity exec triqs_latest.sif python -c <commands>
   - or something similar. Maybe a bash script containing the wanted commands instead of python.
   - singularity shell triqs_latest.sif opens up a shell environment where TRIQS is loaded.
   - Note that singularity creates a self-contained execution environment. You may need to reload modules or reinitialise environment variables.

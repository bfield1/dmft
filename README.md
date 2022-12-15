```
dmft - Python package using TRIQS to perform DMFT calculations
Copyright (C) 2022 Bernard Field

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

Dynamical mean-field theory (DMFT) is a useful method for calculating the properties of strongly correlated lattices.
See e.g. <https://aip.scitation.org/doi/abs/10.1063/1.1800733> for an overview or <https://link.aps.org/doi/10.1103/RevModPhys.68.13> for a comprehensive review.

This code handles the full stack of DMFT calculations, from initialisation, input/output, and post-processing.
It has utilities allowing it to read and do limited post-processing even without TRIQS installed.
Core functions are available via command line arguments for easy interfacing with batch scripts.

HDF5 is used to save the data.

# Citation

If you use this code in your research, please cite it.
The author is Bernard Field.
You can cite the code directly:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7439857.svg)](https://doi.org/10.5281/zenodo.7439857)

Other publications which can be cited (including my thesis) are in preparation.
Check back soon to see when the reference list will be updated.

# DMFT

The dmft.dmft module allows for running DMFT calculations.
TRIQS/CTHYB (a quantum Monte Carlo continuous-time hybridisation expansion solver) is used as the impurity solver.

dmft.dmft records the measured quantities for each loop.
It records simulation parameters in 'params' and code metadata in 'code', either in the root or, if it has changed since the first loop, within each loop.

It also allows for continuation jobs using the 'continue' sub-parser.
A continuation job will use the existing system parameters (although if it has a substrate it must be flagged with --substrate), although different computational parameters can be indicated with --newparams.

The produced Green's functions have two blocks, 'up' and 'down', with each block being a 1-by-1 matrix.

By default, paramagnetism is enforced, such that the spin up and down solutions are the same.

## New lattices

To consider new lattices, implement a method which calculates the density of states DOS, along with an energy mesh and the width of each energy point/spacing between them, as in dmft.dos.
You can then provide that DOS as an argument in DMFTHubbard.set_dos.
You may wish to build a subclass of DMFTHubbard or DMFTHubbardSubstrate which handles this automatically, along with writing appropriate metadata for the lattice.

Cluster DMFT is currently not supported.

# MaxEnt

dmft.maxent is a wrapper for the TRIQS/MaxEnt analysis.
It is a method for analytic continuation, obtaining a spectral function from imaginary time data.
This wrapper method reads data output by dmft.dmft and provides numerous useful plotting methods.

# Record post processing

dmft.record_post_proc does some post-processing which requires TRIQS and records it to the HDF5 archive.
Namely, it looks at density, effective spin (calculated from the density matrix), and Pade approximants.

# Compress file

dmft.compress_file cuts out some unnecessary data in order to reduce the file size.
It removes the reconstruction of the Green's function at each alpha from MaxEnt.
And it can remove intermediate DMFT loops, keeping just the last few (which should be converged).

# Plotting

dmft.plot_loops allows inspecting the convergence of data across DMFT loops.

dmft.maxent.MaxEnt has methods for displaying data, either for inspecting convergence or plotting spectra.

dmft.plot_sweeps is useful for generating high-quality figures comparing many different DMFT calculations.

# Logging

dmft.logging contains several utilities for saving and manipulating data in HDF5 archives.
It is primarily focused on recording and displaying text-based files, which might represent logs from a calculation or a copy of the job script.

It uses h5py rather than TRIQS/h5, allowing it to run without TRIQS.
(There may be some discrepancies in how they prefer to handle multi-line text, though.)

# Fake TRIQS

Modified copies of triqs, triqs_maxent, and h5 are included under dmft.faketriqs.
They are intended to allow saved DMFT data to be read without having to install and compile TRIQS.
Python-only functions are provided un-modified, while C-functions were either omitted or, where essential, replacement functionality using Python is provided.
h5py provides an interface with HDF5 archives, although to avoid data inconsistencies dmft.faketriqs.h5 is limited to be read-only.

triqs and triqs_maxent are used under GNU-GPL v3, while h5 is used under the Apache License v2.
The original copyright for these modules is with the Simons Foundation.

Several other dmft modules issue warnings when imported if TRIQS cannot be found, informing the user that some functions may be restricted.
To silence these warnings, include the following line of code prior to the imports:
```python3
import dmft.faketriqs.silenceimportlogger
```

# Cluster use

To run TRIQS on the cluster:
 - module load singularity
 - singularity pull docker://flatironinstitute/triqs
   - This step only needs to be done once. It creates triqs_latest.sif, which should be put in a convenient location.
 - mpirun -np <number of processes> singularity exec triqs_latest.sif python -c <commands>
   - or something similar. Maybe a bash script containing the wanted commands instead of python.
   - singularity shell triqs_latest.sif opens up a shell environment where TRIQS is loaded.
   - Note that singularity creates a self-contained execution environment. You may need to reload modules or reinitialise environment variables.

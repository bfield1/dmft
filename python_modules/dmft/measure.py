"""
Functions for measuring DMFT quantities.
"""

import numpy as np
import h5py

import triqs.gf

from dmft.utils import archive_reader, h5_read_full_path, get_last_loop
import dmft.logging.cat

def quasiparticle_residue(sigma, block='up', index=0):
    """Returns the quasiparticle residue from the (imaginary frequency) self-energy"""
    # Extract beta
    beta = sigma[block].mesh.beta
    # Calculate quasiparticle residue
    return 1/(1 - sigma[block](0)[index,index].imag * beta/np.pi)

@archive_reader
def quasiparticle_residue_from_archive(archive, loop=None, block='up', index=0):
    """
    Loads Sigma_iw from an archive and gives the quasiparticle residue
    Defaults to the last loop.
    """
    # If no loop given, get the last loop
    if loop is None:
        loop = get_last_loop(archive)
    # If loop is an integer, convert to string
    elif isinstance(loop, int):
        loop = 'loop-{:03d}'.format(loop)
    # Implied is the possibility of passing loop as an explicit string
    # Load Sigma_iw, the self-energy
    sigma = archive[loop]['Sigma_iw']
    # Give the quasiparticle residue
    return quasiparticle_residue(sigma, block, index)

@archive_reader
def density_from_archive(archive, loop=None):
    """
    Loads G_iw from an archive and returns the total density.
    Defaults to the last loop
    """
    # If no loop given, get the last loop
    if loop is None:
        loop = get_last_loop(archive)
    # If loop is an integer, convert to string
    elif isinstance(loop, int):
        loop = 'loop-{:03d}'.format(loop)
    # Implied is the possibility of passing loop as an explicit string
    # Load G_iw, the Green's function
    G = archive[loop]['G_iw']
    # Give the total density.
    # Need real because it has a machine-error imaginary component
    return G.total_density().real

def average_sign_from_archive(archive):
    """
    Loads the average sign of the last DMFT loop

    The average sign should be 1. If it is less than 1 (especially if it is
    close to zero) then the Monte Carlo calculation is not converged and more
    statistics are needed.

    Implementation is a bit tricky, as I historically haven't explicitly
    recorded the average sign. It instead lives in the DMFT logs. So I need to
    parse some text.

    To make matters even worse, for some infuriating unknown reason the strings
    written by h5py are not readable by TRIQS' h5.

    But then later I do record the average sign properly.
    """
    if isinstance(archive, str):
        with h5py.File(archive, 'r') as A:
            return average_sign_from_archive(A)
    # First, check if the average sign is recorded
    # Get the last loop:
    last_loop = sorted([k for k in archive if k[0:4] == 'loop'])[-1]
    # See if the desired data is in the last loop
    if 'average_sign' in archive[last_loop]:
        return archive[last_loop]['average_sign'][()]
    # Otherwise, we need to dig into the logs to extract the average sign
    # Look at the logs
    log_group = archive['logs']
    # Get the names of the logs
    log_names = [k for k in log_group if k[0:4] == 'dmft']
    # Get the last log
    log_name = sorted(log_names)[-1]
    log = dmft.logging.cat.read_string(archive, 'logs/'+log_name)
    # If it is a single block of text, split by newlines
    # We want log to be a list, where each item is a line of text
    if isinstance(log, str):
        log = log.split('\n')
    # Go through and find the last "Average sign:" line
    sign_line = None
    for line in log:
        if "Average sign:" in line:
            sign_line = line
    if sign_line is None:
        raise ValueError("Could not find Average sign")
    # Parse the number. It is space delimited
    return float(sign_line.split()[-1])

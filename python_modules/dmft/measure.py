"""
Functions for measuring DMFT quantities.
"""

import numpy as np

import triqs.gf

from dmft.utils import archive_reader, h5_read_full_path, get_last_loop

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
        loop = 'loop-{:d03}'.format(loop)
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
        loop = 'loop-{:d03}'.format(loop)
    # Implied is the possibility of passing loop as an explicit string
    # Load G_iw, the Green's function
    G = archive[loop]['G_iw']
    # Give the total density.
    # Need real because it has a machine-error imaginary component
    return G.total_density().real

"""
Saves post-processing measurements to the archive

Especially useful for when TRIQS might not be available later
"""

import numpy as np

# Need full version because we're writing
from h5 import HDFArchive
import triqs.gf as gf
import triqs.operators as op
import triqs.atom_diag

import dmft.measure
from dmft.utils import archive_writer, get_last_loop, format_loop

def save_density_to_loop(SG):
    """
    With a HDFArchiveGroup for the loop already open, writes total density
    """
    G = SG['G_iw']
    SG['density'] = G.total_density().real

@archive_writer
def save_density_to_archive(archive, loop=None):
    """
    Writes total density to the archive

    Inputs:
        archive - str or HDFArchive
        loop - None (do last loop), int, or str
    Effect:
        saves to loop-###/density
    """
    loop = format_loop(archive, loop)
    # Save the density
    save_density_to_loop(archive[loop])

def save_effective_spin_to_loop(SG):
    """
    With a HDFArchiveGroup for the loop already open, writes effective spin
    """
    # Read data from the archive
    rho = SG['density_matrix']
    h_loc_diag = SG['h_loc_diagonalization']
    # Define the operator
    Sz = 0.5 * (op.n('up',0) - op.n('down',0))
    # Take the expectation value
    SS = triqs.atom_diag.trace_rho_op(rho, 3*Sz*Sz, h_loc_diag)
    # Convert to effective spin
    spin = -0.5 + 0.5 * np.sqrt(1 + 4*SS)
    # Write
    SG['effective_spin'] = spin

@archive_writer
def save_effective_spin_to_archive(archive, loop=None):
    """
    Writes effective spin to the archive, if the density matrix was recorded

    Inputs:
        archive - str or HDFArchive
        loop - None (do last loop), int, or str
    Effect:
        saves to loop-###/effective_spin
    """
    loop = format_loop(archive, loop)
    # Save the density
    try:
        save_effective_spin_to_loop(archive[loop])
    except KeyError:
        raise KeyError(f"density_matrix was not recorded in {loop}.")

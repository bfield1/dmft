"""
Saves post-processing measurements to the archive

Especially useful for when TRIQS might not be available later

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""

import argparse

import numpy as np

# Need full version because we're writing
from h5 import HDFArchive
import triqs.gf as gf
import triqs.operators as op
import triqs.atom_diag

import dmft.measure
from dmft.utils import archive_writer, get_last_loop, format_loop, count_loops

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

def save_pade_to_loop(SG, window=(-50,50)):
    """
    With a HDFArchiveGroup for the loop already open, writes pade approximation
    """
    G = SG['G_iw']
    SG['pade'] = dmft.measure.pade_from_Giw(G, window)

@archive_writer
def save_pade_to_archive(archive, loop=None, window=(-50,50)):
    """
    Writes Pade approximation to G(w) to the archive

    Inputs:
        archive - str or HDFArchive
        loop - None (do last loop), int, or str
    Effect:
        saves to loop-###/pade
    """
    loop = format_loop(archive, loop)
    # Save the density
    save_pade_to_loop(archive[loop], window)

def save_all_to_loop(SG):
    """
    With HDFArchiveGroup of a loop, writes density and effective spin (if applicable)
    """
    save_density_to_loop(SG)
    if 'density_matrix' in SG:
        save_effective_spin_to_loop(SG)

@archive_writer
def save_all_to_archive(archive, loop=None):
    """
    Writes density and effective spin (if applicable) to archive

    Inputs:
        archive - str or HDFArchive
        loop - None (do last loop), int, or str
    """
    loop = format_loop(archive, loop)
    save_all_to_loop(archive[loop])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Records post-processing data in archives (their last loop only)")
    parser.add_argument('archives', nargs='+',
            help="HDF5 archives made by dmft.dmft to modify.")
    parser.add_argument('-a','--all', action='store_true',
            help="Do all loops instead of just the last loop.")
    parser.add_argument('-p','--pade', type=float, nargs='*',
            help="Also do Pade approximation. May optionally give two numbers to define an energy window (default -50 50).")
    args = parser.parse_args()

    # Process/validate arguments
    if args.pade is not None:
        if len(args.pade) == 0:
            args.pade = [-50,50]
        elif len(args.pade) != 2:
            parser.error(f"--pade must have 0 or 2 arguments, but {len(args.pade)} were given.")

    for A in args.archives:
        print(A)
        if args.all:
            # Post-process all the loops
            # Get the number of loops
            nloops = count_loops(A)
            # Iterate over all loops
            for i in range(nloops):
                try:
                    save_all_to_archive(A, i)
                except KeyError:
                    # This given loop is missing. No matter
                    pass
                else:
                    # The loop does exist, so don't need to try-except
                    if args.pade is not None:
                        save_pade_to_archive(A, i, args.pade)
        else:
            # Post-process only the last loop
            save_all_to_archive(A)
            if args.pade is not None:
                save_pade_to_archive(A, window=args.pade)

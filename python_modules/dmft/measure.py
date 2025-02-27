"""
Functions for measuring DMFT quantities.

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""

import warnings
import logging

import numpy as np
import matplotlib.pyplot as plt
import h5py

from dmft.faketriqs.importlogger import logger as triqslogger
try:
    import triqs.gf as gf
    import triqs.operators as op
    import triqs.atom_diag
    import triqs.version
    _triqs_available = True
except ImportError:
    triqslogger.warning("triqs not found. Loading fake version")
    import dmft.faketriqs.triqs.gf as gf
    _triqs_available = False

from dmft.utils import archive_reader, h5_read_full_path, get_last_loop, format_loop
import dmft.logging.cat
from dmft.maxent import MaxEnt

logger = logging.getLogger(__name__)

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
    loop = format_loop(archive, loop)
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
    loop = format_loop(archive, loop)
    if _triqs_available:
        # Load G_iw, the Green's function
        G = archive[loop]['G_iw']
        # Give the total density.
        # Need real because it has a machine-error imaginary component
        return G.total_density().real
    else:
        # Check if it's recorded
        if 'density' in archive[loop]:
            return archive[loop]['density']
        else:
            raise ImportError(f"Cannot measure density because TRIQS not available, and density was not recorded in {loop}.")

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

def band_edges_from_archive(archive, threshold=0.001, choice='Chi2Curvature'):
    """
    Get the edges of the band gap from the spectrum.

    Inputs:
        archive - str pointing to h5 archive
        threshold - number, below which we treat spectrum as 0
        choice - str or int, which MaxEnt alpha/analyzer to use.
    Output:
        bandmin, bandmax - two numbers, bottom and top of the band gap
    """
    result = MaxEnt.load(archive)
    return band_edges(result.omega, result.get_spectrum(choice), threshold)

def band_edges(omega, spectrum, threshold=0.001):
    """
    Get the edges of the band gap from the spectrum.

    Inputs:
        omega - array of energies
        spectrum - array of spectral function values at corresponding omega
        threshold - number, below which we treat spectrum as 0
    Output:
        bandmin, bandmax - two numbers, bottom and top of the band gap
    """
    mask = (omega >= 0) & (spectrum > threshold)
    if len(omega[mask]) == 0:
        top = omega[-1]
    else:
        top = omega[mask][0]
    mask = (omega <= 0) & (spectrum > threshold)
    if len(omega[mask]) == 0:
        bottom = omega[0]
    else:
        bottom = omega[mask][-1]
    return bottom, top

def band_gap(omega, spectrum, threshold=0.001):
    """
    Gets the band gap

    Inputs:
        omega - array of energies
        spectrum - array of spectral function values at corresponding omega
        threshold - number, below which we treat spectrum as 0
    Output:
        gap - number, the band gap
    """
    bottom, top = band_edges(omega, spectrum, threshold)
    return top - bottom

def band_gap_from_archive(archive, threshold=0.001, choice='Chi2Curvature'):
    """
    Get the band gap from an archive.

    Inputs:
        archive - str pointing to h5 archive
        threshold - number, below which we treat spectrum as 0
        choice - str or int, which MaxEnt alpha/analyzer to use.
    Output:
        gap - number, the band gap
    """
    result = MaxEnt.load(archive)
    return band_gap(result.omega, result.get_spectrum(choice), threshold)

def plot_band_edges(omega, spectrum, threshold=0.001):
    """
    Plots band edges of a spectrum with the spectrum.
    Intended as a quick diagnostic.
    Inputs:
        omega - array of energies
        spectrum - array of spectral function values at corresponding omega
        threshold - number, below which we treat spectrum as 0
    """
    bottom, top = band_edges(omega, spectrum, threshold)
    plt.plot(omega, spectrum)
    plt.axvline(bottom, c='k', lw=1)
    plt.axvline(top, c='k', lw=1)
    plt.show()

def plot_band_edges_from_archive(archive, threshold=0.001, choice='Chi2Curvature'):
    """
    Plots band edges of a spectrum with the spectrum.
    Intended as a quick diagnostic.
    Inputs:
        archive - str pointing to h5 archive
        threshold - number, below which we treat spectrum as 0
        choice - str or int, which MaxEnt alpha/analyzer to use.
    """
    result = MaxEnt.load(archive)
    return plot_band_edges(result.omega, result.get_spectrum(choice), threshold)

@archive_reader
def static_observable_from_archive(archive, my_op, loop=None):
    """
    Runs triqs.atom_diag.trace_rho_op on a DMFT run

    Inputs:
        archive - h5 archive or str pointing to one
        my_op - triqs.operators operators
        loop - optional non-negative int or str (default None).
    Output:
        scalar, the expectation value of the given operator
    """
    loop = format_loop(archive, loop)
    # Load rho if it exists
    if 'density_matrix' in archive[loop]:
        rho = archive[loop]['density_matrix']
    else:
        raise KeyError(f"density_matrix was not recorded in {loop}.")
    # If density_matrix exists, then h_loc_diag probably does too
    h_loc_diag = archive[loop]['h_loc_diagonalization']
    # Do the trace
    return triqs.atom_diag.trace_rho_op(rho, my_op, h_loc_diag)

@archive_reader
def effective_spin_from_archive(archive, loop=None):
    """
    S_eff(S_eff+1) = <S.S> = 3<S_z S_z>, solve for S_eff
    
    Inputs: archive - h5 archive or str
        loop - optional non-negative int or str (default None)
    Output: scalar, the effective spin.
    """
    if _triqs_available:
        Sz = 0.5 * (op.n('up',0) - op.n('down',0))
        SS = static_observable_from_archive(archive, 3*Sz*Sz, loop)
        return -0.5 + 0.5 * np.sqrt(1 + 4*SS)
    else:
        # Check if it's recorded
        loop = format_loop(archive, loop)
        if 'effective_spin' in archive[loop]:
            return archive[loop]['effective_spin']
        else:
            raise ImportError(f"Cannot measure effective spin because TRIQS not available, and effective_spin was not recorded in {loop}.")

def pade_from_Giw(Giw, window=(-50,50)):
    """
    From an imaginary-frequency Green's function, get real frequency with Pade

    Accuracy doesn't seem to depend on the size of the window, so make it
    as big as you need.

    Inputs:
        Giw - BlockGf or Gf with MeshImFreq
        window - tuple of two numbers, energy window
    Output:
        BlockGf or Gf with same block/matrix structure as Giw but with a
            MeshReFreq with window.
    """
    if not _triqs_available:
        raise ImportError("Cannot do Pade approximants because TRIQS not available.")
    version = tuple(int(x) for x in triqs.version.version.split('.'))
    # Get a real frequency Green's function matching the Giw structure
    if version < (3,1,0):
        def gRe_from_gIm(g):
            gRe = gf.GfReFreq(window=window, target_shape=g.target_shape,
                    indices=g.indices, name=g.name)
            gRe.set_from_pade(g)
            return gRe
    else:
        # Gf indices were deprecated around version 3.1.0.
        def gRe_from_gIm(g):
            gRe = gf.GfReFreq(window=window, target_shape=g.target_shape,
                    name=g.name)
            gRe.set_from_pade(g)
            return gRe
    if isinstance(Giw, gf.Gf):
        return gRe_from_gIm(g)
    if isinstance(Giw, gf.BlockGf):
        blocks = [gRe_from_gIm(g) for (name, g) in Giw]
        names = [name for (name, g) in Giw]
        return gf.BlockGf(block_list=blocks, name_list=names)
    raise TypeError("Unrecognised type for argument Giw")

@archive_reader
def pade_from_archive(archive, loop=None, window=(-50,50)):
    """
    Gets the Pade-approximated real-frequency G from a dmft archive

    Inputs:
        archive - HDFArchive or str, file to load data from
        loop - int or str or None, loop to get G_iw from. If None, get from
            latest loop.
        window - tuple of two numbers, energy window
    Output:
        BlockGf with real frequency mesh
    """
    if len(window) != 2:
        raise ValueError("window must have two elements")
    loop = format_loop(archive, loop)
    # If we have TRIQS, compute it
    if _triqs_available:
        Giw = archive[loop]['G_iw']
        return pade_from_Giw(Giw, window)
    else:
        # Check if we have it recorded
        if 'pade' in archive[loop]:
            G = archive[loop]['pade']
            # Check the window
            if (min(G.mesh.values()) != window[0]) or (max(G.mesh.values()) != window[1]):
                logger.warning(f"Requested window {window}, but recorded Green's function has window {(min(G.mesh.values()), max(G.mesh.values()))}")
                # A mismatch is not a critical error, but the user might like 
                # to know.
            return G
        else:
            raise ImportError(f"Cannot get Pade approximants because TRIQS not available, and pade was not recorded in {loop}.")

def spectrum_from_G(G, validate=True):
    """
    From Gf with real frequency mesh, get spectrum.

    Inputs:
        G - Gf (not BlockGf)
        validate - Boolean, do tests to check if the resulting spectrum is
            sensible. Logs warnings if not.
    Outputs:
        omega - (N,) array
        data - (N,) array
    """
    # Frequency mesh
    omega = np.array([x for x in G.mesh.values()])
    # Data. -Im(G)/pi to make it spectral function.
    # Squeeze to convert 1*1 matrix to a scalar.
    data = -G.imag.data.squeeze() / np.pi
    if omega.shape != data.shape:
        raise ValueError("Matrix-values Green's function not allowed.")
    # Validation tests
    if validate:
        # Spectrum should be positive
        if data.min() < -1e-3:
            logger.warning(f"Spectrum has negative values, down to {data.min()}.")
        # Spectrum should be normalised
        norm = np.trapz(data, omega)
        if abs(norm - 1) > 0.05:
            logger.warning(f"Spectrum not normalised. Norm = {norm}.")
    return omega, data

def pade_spectrum_from_archive(archive, loop=None, window=(-50,50),
        validate=True, block='up', indexL=0, indexR=0):
    """
    Gets the Pade-approximated spectrum from a dmft archive

    Inputs:
        archive - HDFArchive or str, file to load data from
        loop - int or str or None, loop to get G_iw from. If None, get from
            latest loop.
        window - tuple of two numbers, energy window
        validate - Boolean, do tests to check if the resulting spectrum is
            sensible. Logs warnings if not.
        block - str, block name
        indexL - int, matrix left index
        indexR - int, matrix right index
    Outputs:
        omega - (N,) array
        data - (N,) array
    """
    G = pade_from_archive(archive, loop, window)
    return spectrum_from_G(G[block][indexL,indexR], validate)


def covariance(Glist, real=False, iL=0, iR=0):
    """
    Gives the covariance matrix from a list of Green's functions

    Inputs:
        Glist - list-like of Green's functions. Must each have same length (N).
        real - Boolean. Whether to keep just the real part of Glist.
            Useful for some Green's functions (e.g. G_tau) which are fully real
            but use complex numbers anyway.
        iL, iR - indices
    Output:
        cov - (N,N) numeric np.array, covariance matrix
    """
    # Extract the data. dtype catches the case of inhomogeneous data
    if real:
        m = np.array([G.data.real[:,iL,iR] for G in Glist], dtype=float)
    else:
        m = np.array([G.data[:,iL,iR] for G in Glist], dtype=complex)
    # Get the covariance matrix
    return np.cov(m, rowvar=False)

@archive_reader
def covariance_from_archive(archive, loops, block='up', Gf='G_tau', real=False,
        iL=0, iR=0):
    """
    Gets the covariance of the Green's functions from an archive

    Useful for MaxEnt

    Inputs:
        archive - str pointing to hdf5 file made by dmft, or HDFArchive
        loops - positive int, number of loops to sample G from
        block - str, which Green's function block to read
        Gf - str, which Green's function to read
        real - Boolean. Whether to keep just the real part of Glist.
            Useful for some Green's functions (e.g. G_tau) which are fully real
            but use complex numbers anyway.
        iL, iR - indices
    Output:
        cov - (N,N) numeric np.array, covariance matrix
    """
    # What's the last array?
    last = get_last_loop(archive)
    # Convert to index, trimming off prefix from 'loop-###'
    last = int(last[5:])
    # Collect the Green's functions
    Glist = []
    for i in range(loops):
        if last - i < 0:
            warnings.warn("Requested more loops than available.")
            # Correct the number of loops
            loops = i
            break
        # Load G
        Glist.append(archive[f'loop-{last-i:03d}/{Gf}'][block])
    # Get the covariance matrix
    return covariance(Glist, real, iL, iR)

def standard_deviation(Glist, real=False, iL=0, iR=0):
    """
    Gives the tau-dependent standard deviation from a list of Green's functions

    Inputs:
        Glist - list-like of Green's functions. Must each have same length (N).
        real - Boolean. Whether to keep just the real part of Glist.
            Useful for some Green's functions (e.g. G_tau) which are fully real
            but use complex numbers anyway.
        iL, iR - indices
    Output:
        std - (N,) numeric np.array, standard deviation at each point
    """
    # Extract the data. dtype catches the case of inhomogeneous data
    if real:
        m = np.array([G.data.real[:,iL,iR] for G in Glist], dtype=float)
    else:
        m = np.array([G.data[:,iL,iR] for G in Glist], dtype=complex)
    # Get the standard deviation
    return np.std(m, axis=0)

@archive_reader
def standard_deviation_from_archive(archive, loops, block='up', Gf='G_tau', real=False,
        iL=0, iR=0):
    """
    Gets the standard deviation of the Green's functions from an archive

    Useful for MaxEnt

    Inputs:
        archive - str pointing to hdf5 file made by dmft, or HDFArchive
        loops - positive int, number of loops to sample G from
        block - str, which Green's function block to read
        Gf - str, which Green's function to read
        real - Boolean. Whether to keep just the real part of Glist.
            Useful for some Green's functions (e.g. G_tau) which are fully real
            but use complex numbers anyway.
        iL, iR - indices
    Output:
        std - (N,) numeric np.array, standard deviation at each point
    """
    # What's the last array?
    last = get_last_loop(archive)
    # Convert to index, trimming off prefix from 'loop-###'
    last = int(last[5:])
    # Collect the Green's functions
    Glist = []
    for i in range(loops):
        if last - i < 0:
            warnings.warn("Requested more loops than available.")
            # Correct the number of loops
            loops = i
            break
        # Load G
        Glist.append(archive[f'loop-{last-i:03d}/{Gf}'][block])
    # Get the standard deviation
    return standard_deviation(Glist, real, iL, iR)

@archive_reader
def get_O_tau(archive):
    """Gets the latest O_tau from an archive"""
    # First try the last loop
    loop = get_last_loop(archive)
    i_loop = int(loop[5:]) # Convert 'loop-###' to a number.
    while i_loop >= 0:
        if 'O_tau' in archive[loop]:
            return archive[loop+'/O_tau']
        # No O_tau there, go to the next loop
        i_loop -= 1
        loop = f'loop-{i_loop:03d}'
    # We haven't found any O_tau. Raise an error
    raise KeyError("O_tau not found")

def integrate_tau(G, real=True):
    """Integrate G_tau or O_tau over whole domain"""
    # Extract x variable
    tau = np.array([t for t in G.mesh.values()])
    # Get real component only if requested
    if real:
        data = G.data.real
    else:
        data = G.data
    # Integrate
    return np.trapz(data, tau)

@archive_reader
def integrate_O_tau_from_archive(archive, real=True):
    """Integrate O_tau from archive (to, e.g. compute spin susceptibility)"""
    return integrate_tau(get_O_tau(archive), real)

def chiT(O_tau):
    """Integrate O_tau and multiply by temperature"""
    chi = integrate_tau(O_tau)
    return chi / O_tau.mesh.beta

@archive_reader
def chiT_from_archive(archive):
    """Integrate O_tau and multiply by temperature"""
    return chiT(get_O_tau(archive))

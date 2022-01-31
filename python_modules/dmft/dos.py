"""
Generates the density of states.
"""

from math import pi

import numpy as np

def dos_from_band(dispersion, nk, dims=2, bins=None, de=None, emin=None, emax=None):
    """
    Creates discrete DOS from a continuous band dispersion

    Histograms band energies from a discrete k-grid

    Inputs:
        dispersion - f(kx, ky,...) -> np.array. Band dispersion, takes numpy
            arrays for the momenta in fractional coordinates [0,1).
            The structure of the returned values isn't important, as long as it
            spits out some number of floats for each k-point.
        nk - positive integer or tuple of integers. K-point grid to use.
        dims - positive integer. Number of dimensions.
        bins - bins argument for np.histogram. Number of energy points.
        de - positive float, used if bins is not provided. Width of energy
            bins.
        emin, emax - floats, manually specified energy range.
    Outputs: Several numpy arrays, shape (number_of_bins,).
        rho - density of states
        energy - midpoints of energy bins
        delta - width of energy bins
    DOS is normalised such that np.sum(rho*delta) == 1.
    """
    # Check that we have a bins or de argument
    if bins is None and de is None:
        raise TypeError("Must provide one of bins or de")
    # Tuple-fy nk
    try:
        len(nk)
    except TypeError:
        nk = (nk,)*dims
    else:
        if len(nk) != dims:
            raise ValueError("nk's length does not match dims")
    # Create a k-point grid
    ks = [np.linspace(0, 1, n) for n in nk]
    ks = np.meshgrid(*ks)
    # Produce a dispersion
    energies = dispersion(*ks)
    # Record energy bounds
    if emin is None:
        emin = energies.min()
    if emax is None:
        emax = energies.max()
    # Determine how many bins we need
    if bins is None:
        # Number of bins needed
        bins = int(np.ceil((emax - emin)/de))
        # Now we expand the energy range so the bins are of the right width,
        # keeping the midpoint the same.
        emin = (emax + emin - bins*de)/2
        emax = bins*de + emin
    # Histogram it
    rho, bin_edges = np.histogram(energies, bins, range=(emin,emax), density=True)
    # Energy differences
    delta = np.diff(bin_edges)
    # Bin midpoints
    energy = (bin_edges[0:-1] + bin_edges[1:])/2
    return rho, energy, delta

def kagome_dos(t, offset, nk, bins=None, de=None):
    """
    DOS for a kagome lattice

    The Dirac point is at -t (so offset=t if want Dirac point at 0 energy)

    Inputs:
        t - float, hopping constant
        offset - float, energy offset of bands
        nk - positive integer or tuple of integers. K-point grid to use.
        bins - bins argument for np.histogram. Number of energy points.
        de - positive float, used if bins is not provided. Width of energy
            bins.
    Outputs: Several numpy arrays, shape (number_of_bins,).
        rho - density of states
        energy - midpoints of energy bins
        delta - width of energy bins
    DOS is normalised such that np.sum(rho*delta) == 3 (because 3 bands).
    """
    # Define the three kagome bands
    def band1(kx, ky):
        return -t - t * np.sqrt(4*(np.cos(kx*pi)**2 + np.cos(ky*pi)**2 + np.cos((ky-kx)*pi)**2) - 3) + offset
    def band2(kx, ky):
        return -t + t * np.sqrt(4*(np.cos(kx*pi)**2 + np.cos(ky*pi)**2 + np.cos((ky-kx)*pi)**2) - 3) + offset
    def band3(kx, ky):
        return 2*t + offset
    # Combine the bands into a single function
    def dispersion(kx, ky):
        return np.array([band1(kx,ky), band2(kx,ky), band3(kx,ky)])
    # Compute the DOS
    rho, energy, delta =  dos_from_band(dispersion, nk, dims=2, bins=bins, de=de)
    # Normalise the DOS because we have three bands
    rho *= 3
    return rho, energy, delta

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
        # The full_like function ensures that this scalar value is made to fill
        # whatever array shape is fed into it.
        return np.full_like(kx, 2*t + offset)
    # Combine the bands into a single function
    def dispersion(kx, ky):
        return np.array([band1(kx,ky), band2(kx,ky), band3(kx,ky)])
    # Compute the DOS
    rho, energy, delta =  dos_from_band(dispersion, nk, dims=2, bins=bins, de=de)
    # Normalise the DOS because we have three bands
    rho *= 3
    return rho, energy, delta

def bethe_dos(t, offset=0, bins=None, de=None):
    """
    Return the DOS for a Bethe lattice with infinite coordination.

    This is just a semicircle with radius 2t.
    
    Inputs:
        t - float, hopping constant
        offset - float, energy offset of bands
        bins - integer. Number of energy points.
        de - positive float, used if bins is not provided. Width of energy
            bins.
    Outputs: Several numpy arrays, shape (number_of_bins,).
        rho - density of states
        energy - midpoints of energy bins
        delta - width of energy bins
    DOS is normalised such that np.sum(rho*delta) == 1.
    """
    # The DOS is \sqrt{1-(E/2t)^2}/(\pi t).
    # We want to histogram it. So we integrate it into bins.
    # The indefinite integral is (x\sqrt{1-x^2}+\arcsin(x))/\pi, x=E/2t
    # Determine our energy range and the number of bins required,
    if bins is None and de is None:
        raise TypeError("Must provide one of bins or de")
    emin = -2*t + offset
    emax = 2*t + offset
    if bins is None:
        # Number of bins needed
        bins = int(np.ceil((emax - emin)/de))
        # Now we expand the energy range so the bins are of the right width,
        # keeping the midpoint the same.
        emin = (emax + emin - bins*de)/2
        emax = bins*de + emin
    # Get the bin edges
    bin_edges = np.linspace(emin, emax, bins+1)
    # Get the bin centers and energy differences
    energy = (bin_edges[0:-1] + bin_edges[1:])/2
    delta = np.diff(bin_edges)
    # Trim the bin edges so we remain in the valid integration domain
    # We didn't do this earlier because our 'true' bin edges for the purposes
    # of histograms may be beyond the boundary.
    # If we've set a constant de, only the extreme bin edges will be out of range
    bin_edges[0] = -2*t + offset
    bin_edges[-1] = 2*t + offset
    # Get our DOS by integration in each bin
    def integral(x):
        # x = (E-offset)/2t
        return (x * np.sqrt(1-x**2) + np.arcsin(x))/np.pi
    rho = integral((bin_edges[1:]-offset)/(2*t)) - integral((bin_edges[0:-1]-offset)/(2*t))
    # Normalise such that sum(rho*delta) == 1
    rho = rho/delta
    return rho, energy, delta

def square_dos(t, offset, nk, bins=None, de=None):
    """
    DOS for a square lattice

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
    DOS is normalised such that np.sum(rho*delta) == 1.
    """
    # Define the square band
    def band(kx, ky):
        return -2*t * (np.cos(2*kx*pi) + np.cos(2*ky*pi))
    # Compute the DOS
    rho, energy, delta =  dos_from_band(band, nk, dims=2, bins=bins, de=de)
    return rho, energy, delta

def triangle_dos(t, offset, nk, bins=None, de=None):
    """
    DOS for a triangular lattice

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
    DOS is normalised such that np.sum(rho*delta) == 1.
    """
    # Define the triangular band
    def band(kx, ky):
        return -2*t * (np.cos(2*kx*pi) + np.cos(2*ky*pi) + np.cos(2*pi*(ky-kx)))
    # Compute the DOS
    rho, energy, delta =  dos_from_band(band, nk, dims=2, bins=bins, de=de)
    return rho, energy, delta

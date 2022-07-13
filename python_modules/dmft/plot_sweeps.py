"""
Does plotting of DMFT data between different archives
"""

import numbers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from dmft.maxent import MaxEnt
from dmft.utils import h5_read_full_path, archive_reader
from dmft.measure import quasiparticle_residue_from_archive, density_from_archive, effective_spin_from_archive
from dmft.plot_loops import wrap_plot

@wrap_plot
def plot_spectrum(archive_list, colors, vals=None, choice='Chi2Curvature', ax=None, colorbar=True, legend=False, offset=0, legendlabel='', xmin=None, xmax=None, block=None):
    """
    Plots the spectra from archive_list on the same Axes

    Inputs:
        archive_list - list of str's or HDFArchive's, archives to read
            the MaxEnts and spectra from. Assumes standard layout and that
            the desired spectrum is in the last MaxEnt analysis.
        colors - list of colors (compatible with matplotlib). The color to draw
            each spectrum from archive_list. Lengths must match.
        vals - list of numbers or strings (must be numbers if colorbar=True).
            Values to attach to each spectrum for legend purposes.
            If colorbar=True, these are numbers to order the color entries.
            If legend=True, these are values to put in the legend entries.
            Optional.
        choice - string. MaxEnt Analyzer to use. Default 'Chi2Curvature'
        colorbar - Boolean. Whether to draw a colorbar. Default True.
        legend - Boolean. Whether to draw a legend. Default False.
        offset - Non-negative number. Amount to shift each spectrum vertically
            so they can be distinguished. Default 0.
        legendlabel - string. Optional. Label to attach to colorbar or legend.
        xmin, xmax - numbers, Optional. Limits for x axis.
        block - string, optional. Block to read (e.g. 'up', 'down').
    """
    # Check compatibility of arguments
    if len(archive_list) != len(colors):
        raise ValueError("archive_list and colors must have matching length")
    if vals is not None and len(archive_list) != len(vals):
        raise ValueError("archive_list and vals must have matching length")
    # Catch a trivial case: no data
    if len(archive_list) == 0:
        return
    # Load the data
    maxents = [MaxEnt.load(A, block=block) for A in archive_list]
    # Plot the spectra
    for i in range(len(maxents)):
        # First spectrum we can use normal plotting.
        # We can also use normal plotting if no offset is called for
        if i == 0 or offset == 0:
            maxents[i].plot_spectrum(choice=choice, ax=ax, inplace=False,
                    color=colors[i])
        # Otherwise, need to do some manual work to get an offset.
        else:
            # Get data for the spectrum
            A = maxents[i].get_spectrum(choice)
            omega = maxents[i].omega
            # Offset
            A += offset * i
            # Plot
            ax.plot(omega, A, c=colors[i])
    # Adjust the maximum of the plot, because it defaults to the ylim of
    # the first spectrum plotted
    ax.autoscale()
    ax.set_ylim(bottom=0)
    ax.set_xlim(xmin, xmax)
    # Create the legend
    if legend:
        ax.legend(vals, title=legendlabel)
    # Create the colorbar
    if colorbar:
        fig = ax.figure
        # Convert vals to a numeric array if it exists
        if vals is not None:
            true_vals = np.asarray(vals)
            # This ensures all entries in vals have same type
            vals_is_num = isinstance(true_vals[0], numbers.Number)
        else:
            # If no vals, default to integers/index
            #true_vals = np.arange(len(archive_list))
            vals_is_num = False
        # Generate the cmap and norm for the colorbar
        if vals_is_num:
            # Sort the colors to be in order of val
            # Invoke the Decorate-Sort-Undecorate idiom
            val_col = [(v, c) for v,c in zip(true_vals, colors)]
            val_col.sort() # Tuples support lexicographical sorting
            colors = [c for v,c in val_col]
            true_vals = np.array([v for v,c in val_col])
        if vals_is_num and len(colors) > 1:
            # We need the midpoints between the vals.
            diff = np.diff(true_vals, prepend=true_vals[0])
            boundaries = true_vals - diff/2
            # The boundaries also need the endpoints of true_vals
            # as well as the midpoints.
            boundaries = np.concatenate((boundaries, [true_vals[-1]]))
            # Pad the boundaries so edges get a full block
            boundaries[0] -= diff[1]/2
            boundaries[-1] += diff[-1]/2
            # Map boundaries to range 0 to 1
            scale_bound = (boundaries - boundaries[0])/(boundaries[-1]-boundaries[0])
            # We want to build a linearly segmented colormap, but with
            # discontinuities
            # This lets us have block segments with different sizes
            # Create a value-color pair list
            clist = []
            for i,c in enumerate(colors):
                clist.append((scale_bound[i], c))
                clist.append((scale_bound[i+1], c))
            cmap = mcolors.LinearSegmentedColormap.from_list('from_list',clist)
            # Generate norm
            norm = mcolors.Normalize(vmin=boundaries[0], vmax=boundaries[-1])
        else:
            # Non-numeric vals. Map to indices instead
            # Or we only have 1 color, which is a trivial case.
            cmap = mcolors.ListedColormap(colors)
            # If we have one colour that has a value, centre our plot on it.
            if len(colors) == 1 and vals_is_num:
                val = vals[0]
            else:
                val = 0
            norm = mcolors.Normalize(vmin=val-0.5, vmax=val+len(colors)-0.5)
            # Integers are centred on each color
        # Create the colorbar
        fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                label=legendlabel)

@wrap_plot
def plot_density(archive_list, vals=None, ax=None, color=None, xlabel='', marker='o', ymin=0, ymax=2, halfline=True):
    """
    Plots the density from different archives in a scatter plot

    Inputs:
        archive_list - list of strs pointing to h5 archives written by dmft
        vals - optional, list of numbers of same length as archive_list.
            Used for x-axis in plotting.
        color - matplotlib colour. Optional.
        xlabel - string. Label for x-axis. Default ''
        marker - matplotlib marker specification. Default 'o'
        ymin - number. Default 0
        ymax - number. Default 2
        halfline - Boolean. Draw a horizontal line at density=1? Default True.
    """
    # Verify that lengths match
    if vals is not None and len(archive_list) != len(vals):
        raise ValueError("Length of archive_list and vals must match")
    # Load the density
    densities = [density_from_archive(A) for A in archive_list]
    # If vals is None, assume we want sequential integers
    if vals is None:
        vals = np.arange(0,len(archive_list))
    # Plot
    ax.plot(vals, densities, linestyle='None', color=color, marker=marker)
    ax.set_ylim(ymin, ymax)
    if halfline:
        ax.axhline(1, color='k', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Density $n$')

@wrap_plot
def plot_quasiparticle_residue(archive_list, vals=None, ax=None, color=None, xlabel='', marker='o', ymin=0, ymax=1):
    """
    Plots the quasiparticle residue from different archives in a scatter plot

    Inputs:
        archive_list - list of strs pointing to h5 archives written by dmft
        vals - optional, list of numbers of same length as archive_list.
            Used for x-axis in plotting.
        color - matplotlib colour. Optional.
        xlabel - string. Label for x-axis. Default ''
        marker - matplotlib marker specification. Default 'o'
        ymin - number. Default 0
        ymax - number. Default 1
    """
    # Verify that lengths match
    if vals is not None and len(archive_list) != len(vals):
        raise ValueError("Length of archive_list and vals must match")
    # Load the Z
    Zs = [quasiparticle_residue_from_archive(A) for A in archive_list]
    # If vals is None, assume we want sequential integers
    if vals is None:
        vals = np.arange(0,len(archive_list))
    # Plot
    ax.plot(vals, Zs, linestyle='None', color=color, marker=marker)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Quasiparticle Residue $Z$')

@wrap_plot
def plot_effective_spin(archive_list, vals=None, ax=None, color=None, xlabel='', marker='o', ymin=0, ymax=1):
    """
    Plots the effective spin from different archives in a scatter plot

    Inputs:
        archive_list - list of strs pointing to h5 archives written by dmft
        vals - optional, list of numbers of same length as archive_list.
            Used for x-axis in plotting.
        color - matplotlib colour. Optional.
        xlabel - string. Label for x-axis. Default ''
        marker - matplotlib marker specification. Default 'o'
        ymin - number. Default 0
        ymax - number. Default 1
    """
    # Verify that lengths match
    if vals is not None and len(archive_list) != len(vals):
        raise ValueError("Length of archive_list and vals must match")
    # Load the effective spin
    spins = [effective_spin_from_archive(A) for A in archive_list]
    # If vals is None, assume we want sequential integers
    if vals is None:
        vals = np.arange(0,len(archive_list))
    # Plot
    ax.plot(vals, spins, linestyle='None', color=color, marker=marker)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Effective spin')

"""
Does plotting of DMFT data comparing different DMFT loops
"""

import functools
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import triqs.gf # Needed to import the DMFT data
from h5 import HDFArchive
import triqs_maxent as me # Needed to import the Maxent data
import triqs.utility.mpi as mpi

from dmft.maxent import MaxEnt
from dmft.utils import h5_read_full_path

def archive_reader(func):
    """
    For a function which reads a HDFArchive as the first argument,
    it opens the archive if a string was passed.
    """
    @functools.wraps(func)
    def reader(archive, *args, **kwargs):
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                return func(A, *args, **kwargs)
        else:
            return func(archive, *args, **kwargs)
    return reader

def wrap_plot(func):
    """
    Wrapper for plotting routines, with boilerplate pre and post processing.
    Sets up the Figure and Axes.
    Then runs the plotting, passing ax.
    Then draws title, invokes tight layout, saves figure, shows, and returns.
    """
    @functools.wraps(func)
    def plotter(*args, ax=None, inplace=True, title=None, save=None, tight=True, **kwargs):
        """
        Plotter keyword arguments:
            ax - matplotlib Axes
            inplace - Boolean, whether or not to plt.show(). If True, returned
                objects have undefined behaviour.
            title - string, plot title.
            save - string, file to save to.
            tight - Boolean, if True call fig.tight_layout()
        Outputs:
            Figure, Axes
        """
        # Generate the Axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            plt.sca(ax)
            fig = ax.figure
        # Do the plotting
        func(*args, ax=ax, **kwargs)
        # Write the title
        if title is not None:
            ax.set_title(title)
        # Tighten the layout
        if tight:
            fig.tight_layout()
        # Save the figure
        if save is not None:
            fig.savefig(save)
        # Show the plot
        if inplace:
            plt.show()
        # Return Figure and Axes
        return fig, ax
    # Update the documentation to fit the original function
    # As I want to concatenate docstrings, I need to do it
    # manually rather than using functools.wrap
    plotter.__doc__ = func.__doc__ + plotter.__doc__
    plotter.__name__ = func.__name__
    return plotter

@archive_reader
def count_loops(archive):
    """Returns the number of DMFT loops in archive"""
    return len([k for k in archive if k[0:5] == 'loop-'])

@archive_reader
def get_maxent(archive, block='up'):
    """
    Get the MaxEnt corresponding to each loop

    Inputs: archive - HDFArchive or string
        block - string, 'up' or 'down'
    Output: List of MaxEnt objects, arranged by loop.
        Will have None values where no matching data can be found.
    """
    total_loops = count_loops(archive)
    mearchive = archive['maxent']
    results = []
    # Load the relevant MaxEnt data, where it exists.
    for analysis in mearchive:
        SG = mearchive[analysis]
        entry = dict()
        entry['name'] = analysis
        try:
            entry['maxent'] = MaxEnt.load(SG,'results')
        except KeyError:
            entry['maxent'] = None
        try:
            loop = SG['dmft_loop'] # "loop-xxx"
            entry['loop'] = int(loop[5:])
        except:
            entry['loop'] = None
        try:
            entry['block'] = SG['block']
        except KeyError:
            entry['block'] = None
        results.append(entry)
    # Sort the results
    # I assume all entries are of standard form, analysis_xx, so I can sort
    # them lexicographically. If I have more than 100 entries this assumption
    # fails.
    if len(results) > 100:
        warn("More than 100 results. Naming and sorting of results may be inconsistent.")
    results.sort(key=lambda x: x['name'])
    # Let us filter by block
    blocks = [entry['block'] for entry in results]
    # Case: no blocks are specified at all
    if blocks.count(None) == len(blocks):
        # In this case, just assume it is a match
        # No processing required
        pass
    # Case two: blocks are specified, but none match
    elif block not in blocks:
        warn(f"No matches for block {block}")
        # Return an empty output.
        return [None] * total_loops
    # Case: all blocks match
    elif blocks.count(block) == len(blocks):
        # No processing required
        pass
    # Case: some blocks match
    else:
        # Collect the indices which do not match block
        idx = [i for i in range(len(blocks)) if blocks[i] != block]
        # Delete non-matching elements
        # Reversing the order (from highest to lowest) preserves indices
        # while editing the list.
        for i in reversed(idx):
            results.pop(i)
    # Now we shall filter out cases where maxent data is missing
    maxents = [entry['maxent'] for entry in results]
    if None in maxents:
        if maxents.count(None) == len(maxents):
            warn(f"All MaxEnt results are missing!")
            return [None] * total_loops
        # Collect indices where maxents is None
        idx = [i for i in range(len(maxents)) if maxents[i] is None]
        # Delete these elements
        for i in reversed(idx):
            results.pop(i)
    # Now deal with missing loops
    loops = [entry['loop'] for entry in results]
    if None in loops:
        # Case: No loops given
        if loops.count(None) == len(loops):
            # In this case, we assume the data sequentially maps to loops
            if len(loops) > total_loops:
                raise ValueError("Loops are not labelled and there are too many of them.")
            elif len(loops) == total_loops:
                return [entry['maxent'] for entry in results]
            else:
                mylist = [entry['maxent'] for entry in results]
                # Pad the missing entries with None's.
                return mylist + [None] * (total_loops - len(mylist))
        # Mixture of loops given and not given
        else:
            raise ValueError("Mixture of specified and unspecified loops. Don't know how to proceed.")
    else:
        # We're now going to collect the loops into a list, with preference
        # for the MaxEnts created later.
        mylist = [None] * total_loops
        for entry in results:
            mylist[entry['loop']] = entry['maxent']
        return mylist

@wrap_plot
@archive_reader
def plot_spectrum(archive, block='up', choice='Chi2Curvature', ax=None, colorbar=True):
    """
    Plots the spectral function as a function of DMFT loop

    Inputs: archive - HDFArchive or string pointing to the archive
        block - string, up or down, which Green's function block to use
        choice - string or integer, which alpha value to use for MaxEnt
            analytic continuation for the spectral function.
        ax - Axes to plot to
        colorbar - Boolean, whether to draw a colorbar.
    Outputs: Figure, Axes
    """
    # Load spectra
    spectra = get_maxent(archive, block)
    loops = len(spectra)
    # Plot spectra
    for i in range(loops):
        if spectra[i] is not None:
            spectra[i].plot_spectrum(choice=choice, ax=ax, inplace=False)
            # Colour the line we just plotted a shade of red.
            ax.get_lines()[-1].set_color([i/loops, 0, 0])
    # Adjust the maximum of the plot, because it defaults to the ylim
    # of the first spectrum plotted.
    ax.autoscale()
    ax.set_ylim(bottom=0)
    # Create the colorbar
    if colorbar:
        fig = ax.figure
        # Create a discrete colormap for these shades of red
        cmap = mcolors.ListedColormap([[i/loops,0,0] for i in range(loops)])
        # Get normalisation such that integers are centred on the colors.
        norm = mcolors.Normalize(vmin=-0.5, vmax=loops-0.5)
        # Create the colorbar
        fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    # The plot has been created. We now return to the wrapper function.

def quasiparticle_residue(sigma, block='up', index=0):
    """Returns the quasiparticle residue from the (imaginary frequency) self-energy"""
    # Extract beta
    beta = sigma[block].mesh.beta
    # Calculate quasiparticle residue
    return 1/(1 - sigma[block](0)[0,0].imag * beta/np.pi)

@wrap_plot
@archive_reader
def plot_quasiparticle_residue(archive, ax, block='up', index=0):
    """
    Plots the quasiparticle residue as a function of DMFT loop
    """
    n_loops = count_loops(archive)
    Z = []
    for i in range(n_loops):
        sigma = h5_read_full_path(archive, 'loop-{:03d}/Sigma_iw'.format(i))
        Z.append(quasiparticle_residue(sigma, block, index))
    ax.plot(np.arange(n_loops), Z, '-o')
    ax.set_xlabel('Loop')
    ax.set_ylabel('Z')

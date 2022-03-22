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
        colorbar - Boolean, whether to draw a colorbar.
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
        fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm),
                ax=ax, label='Loop')
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

    Inputs:
        archive - HDFArchive or str
        ax - Matplotlib Axes
        block - str (default 'up'), block of Sigma to read
        index - integer (default 0), index within block to read
    """
    n_loops = count_loops(archive)
    Z = []
    for i in range(n_loops):
        sigma = h5_read_full_path(archive, 'loop-{:03d}/Sigma_iw'.format(i))
        Z.append(quasiparticle_residue(sigma, block, index))
    ax.plot(np.arange(n_loops), Z, '-o')
    ax.set_xlabel('Loop')
    ax.set_ylabel('Z')

@wrap_plot
@archive_reader
def plot_density(archive, ax):
    """
    Plots the total density as a function of DMFT loop.

    Inputs:
        archive - HDFArchive or str
        ax - Matplotlib Axes
    """
    n_loops = count_loops(archive)
    density = []
    for i in range(n_loops):
        g = h5_read_full_path(archive, 'loop-{:03d}/G_iw'.format(i))
        density.append(g.total_density().real)
    ax.plot(np.arange(n_loops), density, '-o')
    ax.set_xlabel('Loop')
    ax.set_ylabel('Density')

@wrap_plot
@archive_reader
def plot_greens_function(archive, gf_name='G_iw', ax=None, block='up', xmax=None, ymin=None, ymax=None, colorbar=True):
    """
    Plots a Green's function as a function of DMFT loop

    Inputs:
        archive - HDFArchive or str
        gf_name - str (default G_iw), name of Green's function to read
        block - str (default 'up'), block of Green's function to read
        xmax, ymin, ymax - numbers (optional), axes limits
        colorbar - Boolean (default True), whether to draw colorbars.
    """
    n_loops = count_loops(archive)
    for i in range(n_loops):
        # Load the Green's function
        g = h5_read_full_path(archive, 'loop-{:03d}/{}'.format(i,gf_name))
        # Invoke the _plot_ protocol to get plotting data
        data = g[block]._plot_({})
        # If this is the first loop, extract some metadata
        if i == 0:
            xlabel = data[0]['xlabel']
            ylabel = data[0]['ylabel']
        # I don't use triqs.plot.mpl_interface oplot because I want to contro
        # the color.
        # Plot real value (red)
        ax.plot(data[0]['xdata'], data[0]['ydata'], color=(i/n_loops,0,0))
        # Plot imaginary value (blue, dashed)
        ax.plot(data[1]['xdata'], data[1]['ydata'], color=(0,0,i/n_loops), ls='--')
    # Green's functions are symmetric or antisymmetric, so values less than 0
    # can be ignored
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    # Draw the axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # We need a double colorbar, because we have a real and imaginary component
    if colorbar:
        fig = ax.figure
        # Create a discrete colormap for these shades of red
        cmap1 = mcolors.ListedColormap([[i/n_loops,0,0] for i in range(n_loops)])
        # And for the shades of blue
        cmap2 = mcolors.ListedColormap([[0,0,i/n_loops] for i in range(n_loops)])
        # Get normalisation such that integers are centred on the colors.
        norm = mcolors.Normalize(vmin=-0.5, vmax=n_loops-0.5)
        # Create the colorbars
        clb = fig.colorbar(cm.ScalarMappable(cmap=cmap1, norm=norm),
                ax=ax, label='Loop', pad=0.005)
        clb.ax.set_title('Re')
        # This second colorbar is on the inside. It doesn't need ticks as
        # it will share the other colorbar's ticks.
        # Note that if fraction is too small to fit the width of the colorbar
        # the colorbar will shrink (because it is specified with a constant
        # aspect ratio). This setting is fine for the default plot.
        clb = fig.colorbar(cm.ScalarMappable(cmap=cmap2, norm=norm),
                ax=ax, ticks=[], fraction=0.05, pad=0.01)
        clb.ax.set_title('Im')

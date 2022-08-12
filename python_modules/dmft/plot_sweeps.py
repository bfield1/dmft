"""
Does plotting of DMFT data between different archives
"""

import numbers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from dmft.maxent import MaxEnt
from dmft.utils import h5_read_full_path, archive_reader
from dmft.measure import quasiparticle_residue_from_archive, density_from_archive, effective_spin_from_archive, integrate_O_tau_from_archive, chiT_from_archive
from dmft.plot_loops import wrap_plot

def sweep_plotter(ymin=None, ymax=None, ylabel='', logx=False):
    """
    Wrapper for plotting values from a sweep across different archives

    func: archive -> float
    ymin, ymax - floats. Overwrite the default values
    ylabel - str. Overwrite the default values
    """
    # Uses the decorator factory idiom. The sweep_plotter() decorator takes
    # some parameters then returns a proper decorator.
    def decorator(func):
        def plotter(archive_list, vals=None, ax=None, color=None, xlabel='',
                marker='o', ymin=ymin, ymax=ymax, ylabel=ylabel, logx=logx,
                logy=False, xmin=None, xmax=None, linestyle='None', **kwargs):
            """
            Inputs:
                archive_list - list of strs pointing to h5 archives written by dmft
                vals - optional, list of numbers of same length as archive_list.
                    Used for x-axis in plotting.
                ax - matplotlib Axes to plot to
                color - matplotlib colour. Optional.
                xlabel - string. Label for x-axis. Default ''
                marker - matplotlib marker specification. Default 'o'
                ymin - number. Lower y-axis limit
                ymax - number. Upper y-axis limit
                ylabel - str. y-axis label
                logx - Boolean, log x scale?
                logy - Boolean, log y scale?
                xmin - number. Lower x-axis limit
                xmax - number. Upper x-axis limit
                linestyle - str. matplotlib linestyle (default 'None')
            """
            # Verify that lengths match
            if vals is not None and len(archive_list) != len(vals):
                raise ValueError("Length of archive_list and vals must match")
            # Load the data
            data = [func(A) for A in archive_list]
            # If vals is None, assume we want sequential integers
            if vals is None:
                vals = np.arange(0,len(archive_list))
            # Plot
            if logx and not logy:
                plotf = ax.semilogx
            elif logy and not logx:
                plotf = ax.semilogy
            elif logx and logy:
                plotf = ax.loglog
            else:
                plotf = ax.plot
            plotf(vals, data, linestyle=linestyle, color=color, marker=marker, **kwargs)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)
            # Annotate
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        # Update the documentation to fit the original function
        # As I want to concatenate docstrings, I need to do it
        # manually rather than using functools.wrap
        plotter.__doc__ = func.__doc__ + plotter.__doc__
        plotter.__name__ = func.__name__
        return plotter
    return decorator

@wrap_plot
def plot_spectrum(archive_list, colors, vals=None, choice='Chi2Curvature',
        ax=None, colorbar=True, legend=False, offset=0, legendlabel='',
        xmin=None, xmax=None, block=None, fmt='{}', logcb=False,
        annotate=False, annotate_offset=0, xlabel=None, ylabel=None,
        special_annotate='{}', special_annotate_idx=0, cmin=None, cmax=None):
    """
    Plots the spectra from archive_list on the same Axes

    Inputs:
        archive_list - list of str's or HDFArchive's, archives to read
            the MaxEnts and spectra from. Assumes standard layout and that
            the desired spectrum is in the last MaxEnt analysis.
        colors - list of colors (compatible with matplotlib). The color to draw
            each spectrum from archive_list. Lengths must match.
            Or None, in which case we use default color cycle
            Or a string, in which case we find a matplotlib cmap of that name.
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
        fmt - str, for formatting vals in legend. Default '{}'
        logcb - Boolean, log scale in colorbar. Default False.
        annotate - Boolean, draw vals directly on the curves. Default False
        annotate_offset - number, vertical offset for annotate text. Default 0.
        xlabel - str, optional. If provided, overrides plot xlabel
        ylabel - str, optional. If provided, overrides plot ylabel
        special_annotate - str, optional, for formatting vals in annotate with
            a special value for the one at special_annotate_idx. Is called
            after fmt.
        special_annotate_idx - int
        cmin, cmax - numbers, optional. If generating colors, normalise vals to
            be in this range.
    """
    # Get colors
    colors = _choose_colors(colors, vals, len(archive_list), logcb,
            cmin=cmin, cmax=cmax)
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
        # This also covers setting up the axes labels
        if i == 0 or offset == 0:
            maxents[i].plot_spectrum(choice=choice, ax=ax, inplace=False,
                    color=colors[i])
            if annotate and vals is not None:
                # We'll need this later
                A = maxents[i].get_spectrum(choice)
                omega = maxents[i].omega
        # Otherwise, need to do some manual work to get an offset.
        else:
            # Get data for the spectrum
            A = maxents[i].get_spectrum(choice)
            omega = maxents[i].omega
            # Offset
            A += offset * i
            # Plot
            ax.plot(omega, A, c=colors[i])
        if annotate and vals is not None:
            # Draw label on the curve (or near enough to it)
            # Where will we draw? Far right.
            if xmax is None:
                x = max(omega)
            else:
                x = xmax
            # Get y at far right
            if x >= max(omega):
                y = A[-1] + annotate_offset
            else:
                # Need to grab the visible y at the far right of the plot
                # Get the index of maximum omega
                idx = len(omega[omega < x])
                y = A[idx] + annotate_offset
            txt = fmt.format(vals[i])
            if i == special_annotate_idx%len(vals) and isinstance(special_annotate, str):
                txt = special_annotate.format(txt)
            ax.text(x, y, txt, c=colors[i], ha='right', va='bottom')
    # Adjust the maximum of the plot, because it defaults to the ylim of
    # the first spectrum plotted
    ax.autoscale()
    ax.set_ylim(bottom=0)
    ax.set_xlim(xmin, xmax)
    # Plot labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Create the legend
    if legend:
        if vals is None:
            ax.legend(title=legendlabel)
        else:
            fmtval = [fmt.format(v) for v in vals]
            ax.legend(fmtval, title=legendlabel)
    # Create the colorbar
    if colorbar:
        make_colorbar(ax=ax, vals=vals, colors=colors, legendlabel=legendlabel, logcb=logcb)

@wrap_plot
def plot_maxent_chi(archive_list, colors, vals=None, choice='Chi2Curvature',
        ax=None, colorbar=False, legend=False, offset=0, legendlabel='',
        xmin=None, xmax=None, block=None, normalise=True, fmt='{}', logcb=False,
        annotate=True, annotate_offset=0, special_annotate='{}',
        special_annotate_idx=0, cmin=None, cmax=None):
    """
    Plots metrics showing performance of MaxEnt

    Specifically, plots logchi^2 vs logalpha, and marks the choice of alpha.
    For Chi2Curvature or LineFit, you'll want the mark to be sitting at the
    join between the low-alpha plateau and the bit where chi^2 grows.

    Inputs:
        archive_list - list of str's or HDFArchive's, archives to read
            the MaxEnts. Assumes standard layout and that
            the desired spectrum is in the last MaxEnt analysis.
        colors - list of colors (compatible with matplotlib). The color to draw
            each line from archive_list. Lengths must match.
            Or None, in which case we use default color cycle
            Or a string, in which case we find a matplotlib cmap of that name.
        vals - list of numbers or strings (must be numbers if colorbar=True).
            Values to attach to each spectrum for legend purposes.
            If colorbar=True, these are numbers to order the color entries.
            If legend=True, these are values to put in the legend entries.
            Optional.
        choice - string. MaxEnt Analyzer to use. Default 'Chi2Curvature'
        colorbar - Boolean. Whether to draw a colorbar. Default False.
        legend - Boolean. Whether to draw a legend. Default False.
        offset - Non-negative number. Amount to shift each spectrum vertically
            so they can be distinguished. Default 0.
        legendlabel - string. Optional. Label to attach to colorbar or legend.
        xmin, xmax - numbers, Optional. Limits for x axis.
        block - string, optional. Block to read (e.g. 'up', 'down').
        normalise - Boolean. Normalise chi2 so start at same point?
        fmt - str, for formatting vals in legend
        logcb - Boolean, log scale in colorbar
        annotate - Boolean, draw vals directly on the curves. Default True
        annotate_offset - number, vertical offset for annotate text. Default 0.
        special_annotate - str, optional, for formatting vals in annotate with
            a special value for the one at special_annotate_idx. Is called
            after fmt.
        special_annotate_idx - int
        cmin, cmax - numbers, optional. If generating colors, normalise vals to
            be in this range.
    """
    # Get colors
    colors = _choose_colors(colors, vals, len(archive_list), logcb,
            cmin=cmin, cmax=cmax)
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
    # Modify choice is necessary
    try:
        maxents[0].data.analyzer_results[choice]
    except KeyError:
        # This means choice isn't here. Try choiceAnalyzer
        try:
            maxents[0].data.analyzer_results[choice+"Analyzer"]
        except KeyError:
            raise KeyError(f"{choice} and {choice}Analyzer not found in MaxEnt")
        else:
            # It is choiceAnalyzer. Update choice to match
            choice = choice+"Analyzer"
    # Plot the curves
    for i in range(len(maxents)):
        # Get the chi2 and alpha
        chi2 = maxents[i].data.chi2
        alpha = np.asarray(maxents[i].alpha)
        # Find the alpha corresponding with the choice
        i_alpha = maxents[i].data.analyzer_results[choice]["alpha_index"]
        # Normalise data
        if normalise:
            chi2 /= chi2.min()
        # Note that, because log scale, offset is multiplied, not added
        chi2 *= 10**(offset*i)
        # Plot
        ax.loglog(alpha, chi2, color=colors[i])
        # Plot the marker
        ax.scatter(alpha[i_alpha], chi2[i_alpha], color=colors[i], marker='o')
        if annotate and vals is not None:
            # Draw label on the curve (or near enough to it)
            # Where will we draw? Far left (Note, alpha is sorted descending)
            if xmin is None:
                x = alpha.min()
            else:
                x = xmin
            # Get y at far left
            if x <= min(alpha):
                y = chi2[-1] * 10**annotate_offset
            else:
                # Need to grab the visible y at the far left of the plot
                # Get the index of minimum alpha
                idx = len(alpha[alpha > x])
                y = chi2[idx] * 10**annotate_offset
            txt = fmt.format(vals[i])
            if i == special_annotate_idx%len(vals) and isinstance(special_annotate, str):
                txt = special_annotate.format(txt)
            ax.text(x, y, txt, c=colors[i], ha='left', va='bottom')
    # Adjust axes limits
    ax.set_xlim(xmin, xmax)
    # Annotate
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\chi^2$')
    # Create the legend
    if legend:
        handles = [mlines.Line2D([],[],color=c) for c in colors]
        if vals is None:
            fmtval = None
        else:
            fmtval = [fmt.format(v) for v in vals]
        ax.legend(labels=fmtval, handles=handles, title=legendlabel)
    # Create the colorbar
    if colorbar:
        make_colorbar(ax=ax, vals=vals, colors=colors, legendlabel=legendlabel, logcb=logcb)

def _choose_colors(colors, vals, n, log=False, cmin=None, cmax=None):
    """
    Processes the color argument

    n - integer, number of entries
    """
    # If colors is already a list of colours, return it
    if not (colors is None or isinstance(colors, str)):
        return colors
    # If we have non-numeric values, replace with indices
    if vals is None or not isinstance(vals[0],numbers.Number):
        vals = [i for i in range(n)]
    vals = np.asarray(vals)
    if cmin is None:
        cmin = vals.min()
    if cmax is None:
        cmax = vals.max()
    if colors is None:
        # If colors not given, default to color cycle
        return [f'C{i%10}' for i in range(n)]
    if isinstance(colors,str):
        # Bring out a matplotlib colormap
        cmap = cm.get_cmap(colors)
        # Take the logarithm of vals
        if log:
            vals = np.log10(vals)
            cmin = np.log10(cmin)
            cmax = np.log10(cmax)
        # Normalise vals on range 0 to 1
        if len(vals) > 1 and cmin < cmax:
            vals = np.maximum(np.minimum((vals - cmin)/(cmax - cmin),1),0)
        elif len(vals) == 1:
            vals = np.array([1])
        # Return colors
        return [cmap(v) for v in vals]

def get_divergent_colors(cmap, vals, mid, low=None, high=None):
    """
    Maps vals to a divergent colormap with a custom midpoint

    Inputs:
        cmap - str or matplotlib.colors.Colormap or None
        vals - list-like of numbers, to get colours for
        mid - number, midpoint for the colour map
        low - optional number, minimum value for colour map
            (default to min(vals))
        high - optional number, maximum value for colour map
            (default to max(vals))
            Values which lies outside the range will be pinned to edge.
    Output: list of colours
    """
    # Catch empty input
    if len(vals) == 0:
        return []
    # Load colourmap
    cmap = cm.get_cmap(cmap)
    # Go to numpy array for smarter operations and indexing
    vals = np.asarray(vals)
    # Get minimum and maximum if not set
    if low is None:
        low = vals.min()
    if high is None:
        high = vals.max()
    # Check error case
    if low > high:
        raise ValueError("Minimum is greater than maximum")
    # We need to map values into the range 0 to 0.5 and 0.5 to 1
    truevals = np.ones(vals.shape) * -1. # This ensures errors are obvious
    # Values at the midpoint will always be 0.5
    truevals[vals == mid] = 0.5
    # If low == high, we have a singular case, no variation allowed.
    if low == high:
        truevals[vals < mid] = 0
        truevals[vals > mid] = 1
    else:
        if low < mid:
            v = vals[vals < mid]
            # Map values to 0-0.5
            truevals[vals < mid] = np.maximum(0.5 - (mid - v) / (mid - low) * 0.5, 0)
        if high > mid:
            v = vals[vals > mid]
            # Map values to 0.5-1
            truevals[vals > mid] = np.minimum(0.5 + (v - mid) / (high - mid) * 0.5, 1)
    return [cmap(v) for v in truevals]

def make_colorbar(ax, colors, vals=None, legendlabel='', logcb=False):
    """
    Create a segmented colorbar from a list of values and colours.

    Inputs:
        ax - matplotlib Axes
        vals - list, or None. If None or a list of non-numeric values,
            values are ignored and we just go by the indices.
        colors - list of colors, or a str or None (if str or None provided,
            must include vals).
        legendlabel - str, title for colorbar.
    Output: colorbar
    """
    # Choose the colors, if necessary
    if isinstance(colors, str) or colors is None:
        colors = _choose_colors(colors, vals, len(vals), logcb)
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
        # If log, make true_vals a log
        if logcb:
            true_vals = np.array([np.log10(v) for v,c in val_col])
        else:
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
        if logcb:
            norm = mcolors.LogNorm(vmin=10**boundaries[0], vmax=10**boundaries[-1])
        else:
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
        # Generate norm
        norm = mcolors.Normalize(vmin=val-0.5, vmax=val+len(colors)-0.5)
        # Integers are centred on each color
    # Create the colorbar
    return fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
            label=legendlabel)


@sweep_plotter(ymin=0, ymax=2, ylabel='Density $n$')
def _help_plot_density(A):
    """Plots the density from different archives"""
    return density_from_archive(A)
@wrap_plot
def plot_density(*args, ax=None, halfline=True, **kwargs):
    """
    Keyword Argument:
        halfline - Boolean. Draw a horizontal line at density=1? Default True.
    """
    _help_plot_density(*args, ax=ax, **kwargs)
    if halfline:
        ax.axhline(1, color='k', linewidth=1)
plot_density.__doc__ = _help_plot_density.__doc__ + plot_density.__doc__

@wrap_plot
@sweep_plotter(ymin=0, ymax=1, ylabel=r'Quasiparticle Residue $Z$')
def plot_quasiparticle_residue(A):
    """Plots the quasiparticle residue from different archives"""
    return quasiparticle_residue_from_archive(A)

@wrap_plot
@sweep_plotter(ymin=0, ymax=0.5, ylabel='Effective spin')
def plot_effective_spin(A):
    """Plots the effective spin from different archives"""
    return effective_spin_from_archive(A)

@wrap_plot
@sweep_plotter(ymin=0, ymax=0.25, ylabel=r'$\chi$')
def plot_O_tau(A):
    """Plots the integrated O_tau (e.g. spin susceptibility) from different archives"""
    return integrate_O_tau_from_archive(A)

@wrap_plot
@sweep_plotter(ymin=0, ylabel=r'$1/\chi$')
def plot_chiinv(A):
    """Plots the inverse of the susceptibility from different archives"""
    return 1/integrate_O_tau_from_archive(A)

@wrap_plot
@sweep_plotter(ymin=0, ymax=0.25, ylabel=r'$T\chi$', logx=True)
def plot_chiT(A):
    """Plots the susceptibility times temperature from different archives"""
    return chiT_from_archive(A)


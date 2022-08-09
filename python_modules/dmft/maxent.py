"""
Runs basic MaxEnt processing
"""
import argparse
from warnings import warn
import functools
import logging

import numpy as np
import matplotlib.pyplot as plt

from dmft.faketriqs.importlogger import logger as triqslogger
try:
    import triqs.gf as gf # Needed to import the DMFT data
except ImportError:
    triqslogger.warning("triqs not found. Loading fake version")
    import dmft.faketriqs.triqs.gf as gf
try:
    from h5 import HDFArchive
except ImportError:
    triqslogger.warning("triqs/h5 not found. Loading fake version.")
    from dmft.faketriqs.h5 import HDFArchive
try:
    import triqs_maxent as me
except ImportError:
    triqslogger.warning("triqs_maxent not found. Loading fake version.")
    import dmft.faketriqs.triqs_maxent as me
try:
    import triqs.utility.mpi as mpi
except ImportError:
    import dmft.faketriqs.triqs.utility.mpi as mpi

import dmft.version
from dmft.utils import h5_write_full_path, h5_read_full_path, get_last_loop, archive_reader2
from dmft.logging.writelog import find_unique_name

def wrap_plot(func):
    """
    Boilerplate for a plotting function.
    Sets the Axes if provided. Saves and displaces the figure according
    to arguments. Returns Figure and Axes.
    Assumes that func will by some means grab the current axes or generate
    them if they do not exist, typically by calling plt.plot.
    """
    @functools.wraps(func)
    def plotter(*args, ax=None, inplace=True, save=None, **kwargs):
        if ax is not None:
            plt.sca(ax)
        func(*args, **kwargs)
        if save is not None:
            plt.savefig(save)
        if inplace:
            plt.show()
        return plt.gcf(), plt.gca()
    return plotter

def spectrum_plotter(func):
    """
    A decorator for functions which plot spectra.
    Handles things like setting the axes limits and labels and drawing the
    legends, as well as creating and returning the Axes.
    The function must take an argument ax=Axes, an Axes object, and do its
    plotting within that Axes object, modifying it in place.
    The function must be able to accept arbitrary arguments and kwargs,
    because I pass literally everything to the function just in case someone
    wants it (maybe it alters the behaviour? I don't know), but I do not expect
    you to do any processing with the default ones besides ax.
    Also updates the docstring to append the plotter information.
    """
    def plotter(*args, ax=None, emin=None, emax=None, amax=None, inplace=True, axeslabels=True, title=None, save=None, **kwargs):
        """
        Plotter keyword arguments:
            ax - matplotlib Axes
            emin, emax - floats, x-limits
            amax - float, y-limit (the other is 0)
            inplace - Boolean, whether or not to plt.show(). If True, returned
                objects have undefined behaviour.
            axeslabels - Boolean, whether or not to set the axes labels.
            title - string, plot title.
            save - string, file to save plot to.
        Outputs:
            Figure, Axes
        """
        # Generate the Axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        # Create the plots
        func(*args, **kwargs, ax=ax, emin=emin, emax=emax, amax=amax,
                inplace=inplace, axeslabels=axeslabels, title=title)
        # Adjust axes limits
        ax.set_xlim(emin, emax)
        ax.set_ylim(0, amax)
        ax.set_title(title)
        # Draw axes labels
        if axeslabels:
            ax.set_xlabel(r'$\omega$')
            ax.set_ylabel(r'$A(\omega)$')
        # Save the figure
        if save is not None:
            fig.savefig(save)
        # Show
        if inplace:
            plt.show()
        return fig, ax
    # Update the documentation to fit the original function
    # As I want to concatenate docstrings, I need to do it
    # manually rather than using functools.wrap
    plotter.__doc__ = func.__doc__ + plotter.__doc__
    plotter.__name__ = func.__name__
    return plotter

class MaxEnt():
    def __init__(self, cost_function='bryan', probability='normal', amin=None,
            amax=None, nalpha=None, **kwargs):
        """
        Class for generating, holding, and manipulating MaxEnt data.
        
        A lot of the keyword arguments can be set later.
        Keyword Arguments:
            cost_function - for TauMaxEnt
            probability - for TauMaxEnt
            alpha_mesh - an AlphaMesh for TauMaxEnt
            amin, amax, nalpha - float, float, integer - parameters to generate
                an AlphaMesh. Not to be used with alpha_mesh.
            omega_mesh
            All other kwargs - for TauMaxEnt
        """
        self.tm = me.TauMaxEnt(cost_function=cost_function, probability=probability, **kwargs)
        if isinstance(self.alpha, me.DataAlphaMesh) and 'scale_alpha' not in kwargs:
            # If a DataAlphaMesh has been provided, we probably don't need to scale
            self.scale_alpha = 1
        if amin is not None:
            self.generate_alpha(amin, amax, nalpha)
        # Whether to save the Default Model
        self._save_D = False
    @property
    def alpha(self):
        """The alpha mesh to sample MaxEnt along"""
        return self.tm.alpha_mesh
    @alpha.setter
    def alpha(self, value):
        self.tm.alpha_mesh = value
    def generate_alpha(self, amin, amax, nalpha, scale_alpha=1):
        """
        Sets a logarithmic alpha mesh

        Inputs: amin - float, minimum alpha
            amax - float, maximum alpha
            nalpha - positive integer, number of alpha points
            scale_alpha - float or str('ndata'). The scale_alpha argument
                for triqs_maxent.run. Defaults to 1 here because I assume that
                when you explicitly set alpha values you want them to be the 
                values you specify.
        Output: triqs_maxent.LogAlphaMesh
        """
        self.alpha = me.LogAlphaMesh(amin, amax, nalpha)
        self.scale_alpha = scale_alpha
        return self.alpha
    @property
    def scale_alpha(self):
        return self.tm.scale_alpha
    @scale_alpha.setter
    def scale_alpha(self, value):
        self.tm.scale_alpha = value
    @property
    def omega(self):
        """The omega mesh for the output spectrum"""
        return self.tm.omega
    @omega.setter
    def omega(self, value):
        self.tm.omega = value
    def generate_omega(self, omin, omax, nomega, cls=me.HyperbolicOmegaMesh):
        """
        Sets the omega mesh

        Inputs: omin - float, minimum omega
            omax - float, maximum omega
            nomega - positive integer, number of omega points
            cls - child of triqs_maxent.BaseOmegaMesh
        Output: cls instance
        """
        self.omega = cls(omin, omax, nomega)
        return self.omega
    @property
    def D(self):
        """The default model"""
        return self.tm.D
    @D.setter
    def D(self, value):
        self.tm.D = value
        # The Default Model is not saved with MaxEntResultsData,
        # so we need to save it separately. But only bother if not the default
        self._save_D = True
    def set_Gaussian_D(self, sigma, x0=0):
        """
        Sets the default model to be a Gaussian. Requires omega to be set.

        Inputs: sigma - float, Gaussian width
            x0 - float, center of peak
        Output: 1D array of len(omega), Default model
        """
        w = np.array(self.omega)
        # A normalized Gaussian of width sigma centered on x0
        f = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(w-x0)**2/(2*sigma**2))
        self.D = f
        return f
    def set_G_tau(self, G_tau):
        """Sets the G_tau for MaxEnt"""
        self.tm.set_G_tau(G_tau)
    def load_G_tau(self, archive, block='up', index=0):
        """Loads a G_tau from the last DMFT loop in a HDF5 archive"""
        G_tau_block = get_last_G_tau(archive)
        self.set_G_tau(G_tau_block[block][index,index])
    def set_error(self, err):
        self.tm.set_error(err)
    def run(self):
        """Runs MaxEnt and gets the results."""
        self.results = self.tm.run()
        self.data = self.results.data
        return self.results
    def save(self, archive, path):
        """
        Saves the results to path in a HDF5 archive
        """
        h5_write_full_path(archive, self.data, path)
    @classmethod
    @archive_reader2
    def load(cls, archive, path=None, block=None):
        """
        Reads saved MaxEnt results from path in a HDF5 archive

        If path is not provided, assumes default names and loads the latest
        MaxEnt (path="maxent/analysis_??/results"
        Or the latest MaxEnt which matches the given block.

        Compatible with any triqs_maxent.MaxEntResultData, not just those
        created by the class.
        Does not set the error, as that doesn't save.
        Also does not set self.results, which is a MaxEntResult object,
        as only MaxEntResultData is saved. But we only need the latter.
        """
        if path is not None:
            data = h5_read_full_path(archive, path)
        elif block is not None:
            # Find the latest maxent/analysis_??/results which matches block
            SG = archive['maxent']
            # Lexicographical sort of keys
            # Note: will give wrong results if have three-digit numbers
            anlist = sorted([k for k in SG if k[0:9] == 'analysis_'])
            if len(anlist) > 100:
                warn("More than 100 MaxEnt analyses. Sorting won't work properly")
            # Go through them backwards
            data = None
            for key in reversed(anlist):
                if 'block' in SG[key]:
                    if SG[key]['block'] == block:
                        data = SG[key]['results']
                        break
            if data is None:
                raise KeyError(f"No 'block' matching {block} found in archive.")
        else:
            data = get_last_maxent(archive)
        self = cls(alpha_mesh=data.alpha)
        self.data = data
        self.tm.set_G_tau_data(data.data_variable, data.G_orig)
        self.omega = data.omega
        return self
    def write_metadata(self, archive, name=None):
        """
        Writes version data and parameters not saved in MaxEntResult to a HDF5 archive
        
        Inputs:
            archive - HDF5 archive or string point to one. Will be written to.
            name - Group to record metadata to.
        """
        # Write version data
        if name is None:
            write_metadata(archive, 'code')
        else:
            write_metadata(archive, name)
        # Write the error we use (as that's also important)
        if name is None:
            name = 'params'
        if hasattr(self, 'tm') and hasattr(self.tm, 'err'):
            # We have an error value. self.tm.err is an array,
            # but we might be able to simplify it as a scalar.
            if self.tm.err.min() == self.tm.err.max():
                err = self.tm.err[0]
            else:
                err = self.tm.err
            h5_write_full_path(archive, err, name+'/maxent_error')
        if self._save_D:
            # The Default Model, not default value
            h5_write_full_path(archive, self.D, name+'/default_model')
    # Now we want some plotting scripts to show the results
    def get_spectrum(self, choice=None):
        """
        Returns the spectral function

        Input:
            choice - integer or string. If integer, is an alpha index.
                If string, is an analyzer name (possibly with the trailing
                "Analyzer" dropped).
                If unspecified, reverts to the default analyzer.
        Output:
            A 1D numeric array, the spectral function.
        """
        # Parse the choice
        if choice is None:
            # Default
            A = self.data.default_analyzer['A_out']
        elif isinstance(choice, str):
            # Choice is an Analyzer name
            if choice in self.data.analyzer_results.keys():
                A = self.data.analyzer_results[choice]['A_out']
            else:
                A = self.data.analyzer_results[choice+'Analyzer']['A_out']
        else:
            # Assume choice is an integer.
            A = self.data.A[choice]
        return A
    #
    @spectrum_plotter
    def plot_spectrum(self, choice=None, color=None, **kwargs):
        """
        Plots the spectral function

        Inputs:
            choice - integer or string. If integer, is an alpha index.
                If string, is an analyzer name (possibly with the trailing
                "Analyzer" dropped).
                If unspecified, reverts to the default analyzer.
            color - color argument for matplotlib.pyplot.plot
        """
        ax = kwargs['ax']
        A = self.get_spectrum(choice)
        # Plot
        ax.plot(self.omega, A, c=color)
    #
    @spectrum_plotter
    def plot_spectrum_fit_comparison(self, probability=False, legend=True, **kwargs):
        """
        Plots the spectral functions chosen by the different alpha fitting methods

        Namely, plots LineFit, Chi2Curvature, Entropy, and optionally Bryan
        and Classic. I leave the last two as optional because getting reliable
        results from them is challenging.

        Inputs:
            probability - Boolean, whether or not to plot the probability-based
                methods (Classic and Bryan). Default False.
            legend - Boolean (default True), whether or not to make a legend.
        """
        ax = kwargs['ax']
        # List the analyzers of interest
        analyzers = ['LineFit','Chi2Curvature','Entropy']
        if probability:
            # I'll plot Bryan and Classic underneath LineFit etc.
            analyzers = ['Bryan','Classic'] + analyzers
        # Plot
        for an in analyzers:
            ax.plot(self.data.omega, self.data.analyzer_results[an+'Analyzer']['A_out'], label=an)
        # Draw the legend
        if legend:
            ax.legend()
    #
    @spectrum_plotter
    def plot_spectrum_alpha_comparison(self, alpha, a_range=4, legend=True, **kwargs):
        """
        Plots the spectra for a few values of alpha around a given alpha

        Useful for determining the stability of a spectrum over an alpha window
        Inputs:
            alpha - string or integer. If string, is the name of an analyzer
                from which to take the optimal alpha. If an integer, is the
                index from which to take the alpha.
                String supports short form, where you omit trailing "Analyzer".
            a_range - positive integer (default 4). Number of alpha's on either
                side of the chosen alpha to also plot. Will automatically
                truncate if go beyond the allowed alpha values.
            legend - Boolean (default True), whether or not to make a legend.
        """
        ax = kwargs['ax']
        # If alpha is a string, convert to a number.
        if isinstance(alpha, str):
            if alpha in self.data.analyzer_results.keys():
                alpha = self.data.analyzer_results[alpha]['alpha_index']
            else:
                # Allow shortened names to work, where we omit 'Analyzer'.
                alpha = self.data.analyzer_results[alpha+'Analyzer']['alpha_index']
        # If the given alpha index was negative, convert to positive.
        if alpha < 0:
            alpha = len(self.alpha) + alpha
        # Get our range
        alpha_range = np.arange(max(alpha-a_range, 0), min(alpha+a_range+1, len(self.alpha)))
        # Plot
        for i in alpha_range:
            ax.plot(self.omega, self.data.A[i],
                    color=[max(i-alpha,0)/a_range, 0, max(alpha-i,0)/a_range],
                    label='{:.2e}'.format(self.alpha[i]))
        # Draw the legend
        if legend:
            ax.legend()
    # Mirror a few of the internal plotting functions
    @wrap_plot
    def plot_chi2(self, *args, **kwargs):
        self.data.plot_chi2(*args, **kwargs)
    @wrap_plot
    def plot_curvature(self, *args, **kwargs):
        self.data.analyzer_results['Chi2CurvatureAnalyzer'].plot_curvature(*args, **kwargs)
    @wrap_plot
    def plot_S(self, *args, **kwargs):
        self.data.plot_S(*args, **kwargs)
    @wrap_plot
    def plot_dS_dalpha(self, *args, **kwargs):
        self.data.analyzer_results['EntropyAnalyzer'].plot_dS_dalpha(*args, **kwargs)
    @wrap_plot
    def plot_linefit(self, *args, **kwargs):
        self.data.analyzer_results['LineFitAnalyzer'].plot_linefit(*args, **kwargs)
    @wrap_plot
    def plot_probability(self, *args, **kwargs):
        self.data.plot_probability(*args, **kwargs)
    def plot_metrics(self, chi2=True, linefit=True, curvature=True,
            probability=True, S=True, dS=True, mark_alphas=True,
            analyzer_list=['LineFitAnalyzer','Chi2CurvatureAnalyzer','ClassicAnalyzer','EntropyAnalyzer'],
            inplace=True, tight_layout=True, title=None, save=None):
        """
        Does a combined plot of all the relevant metrics.

        Inputs:
            chi2 - Boolean. Plot chi2?
            linefit - Boolean. Plot linefit on chi2? (If True, plots over chi2)
            curvature - Boolean. Plot chi2 curvature?
            probability - Boolean. Plot probability?
            S - Boolean. Plot entropy?
            dS - Boolean. Plot dS/dlogalpha?
            mark_alphas - Boolean. Whether to mark the alphas chosen by the
                analyzers.
            analyzer_list - list of strings. Analyzers for mark_alpha.
            inplace - Boolean. Whether to plt.show()
            tight_layout - Boolean. Call fig.tight_layout()?
            title - string, plot title
            save - string, file to save plot to
        Outputs:
            Figure, list of Axes.
        """
        if linefit:
            chi2 = True
        # Determine how many plots we have.
        n_plots = chi2 + (curvature or probability) + (S or dS)
        if n_plots == 0:
            raise TypeError("Cannot plot nothing.")
        # Make a figure
        fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True,
                squeeze=False, figsize=[6.4,1.5*n_plots+2])
        # Determine what order the plots go in.
        arrangement = [[('chi2',chi2),('linefit',linefit)],
                [('curvature',curvature),('probability',probability)],
                [('S',S),('dS',dS)]]
        # Delete rows which have nothing in them.
        for i in [2,1,0]:
            if (not arrangement[i][0][1]) and (not arrangement[i][1][1]):
                del arrangement[i]
            # and delete false elements
            elif not arrangement[i][1][1]:
                del arrangement[i][1]
            elif not arrangement[i][0][1]:
                del arrangement[i][0]
        # Arrangement now contains only those plots we wish to plot
        # Record which one is on the bottom so we know where to draw alpha
        bottom = arrangement[-1][0][0]
        # And record which one is the top so we know where to draw the title
        top = arrangement[0][0][0]
        # Have a mapping from string to y-variable
        ydict = dict(curvature=self.data.analyzer_results['Chi2CurvatureAnalyzer']['curvature'],
                probability=np.exp(self.data.probability-np.nanmax(self.data.probability)),
                S=self.data.S,
                dS=self.data.analyzer_results['EntropyAnalyzer']['dS_dalpha'])
        # Mapping from arrangement string to ylabel
        ylabels = dict(curvature='curvature', probability=r'$p(\alpha)$',
                S='$S$', dS=r'$dS/d{\rm log}\alpha$')
        # Go through the rows and make plots
        for ax, row in zip(axs[:,0], arrangement):
            # Chi2 and linefit needs different handling
            if row[0][0] == 'chi2':
                ax.loglog(self.alpha, self.data.chi2)
                if linefit:
                    ylims = ax.get_ylim()
                    # Manually reconstruct the lines (in log-space!)
                    linefit_params = self.data.analyzer_results['LineFitAnalyzer']['linefit_params']
                    for line in linefit_params:
                        if len(line) == 1:
                            ax.loglog(self.alpha, np.exp(line[0]) * np.ones(self.alpha.shape), zorder=1.9)
                        else:
                            ax.loglog(self.alpha, np.exp(line[1]) * np.power(self.alpha, line[0]), zorder=1.9)
                    ax.set_ylim(ylims)
                ax.set_ylabel(r'$\chi^2$')
            else:
                color1 = 'C0'
                color2 = 'C1'
                ax.semilogx(self.alpha, ydict[row[0][0]], c=color1)
                ax.set_ylabel(ylabels[row[0][0]])
                if len(row) > 1:
                    # Color the y axis
                    ax.set_ylabel(ylabels[row[0][0]], color=color1)
                    ax.tick_params(axis='y', labelcolor=color1)
                    # Create a secondary axis
                    ax2 = ax.twinx()
                    ax2.semilogx(self.alpha, ydict[row[1][0]], c=color2)
                    ax2.set_ylabel(ylabels[row[1][0]], color=color2)
                    ax2.tick_params(axis='y', labelcolor=color2)
            # Draw the x-axis label
            if row[0][0] == bottom:
                ax.set_xlabel(r'$\alpha$')
            if row[0][0] == top:
                ax.set_title(title)
        # Annotate alphas
        if mark_alphas:
            for an in analyzer_list:
                alpha = self.alpha[self.data.analyzer_results[an]['alpha_index']]
                for ax in axs.flat:
                    ax.axvline(alpha, c='0.5', lw=1)
                axs[0,0].annotate(an.replace('Analyzer',''), (alpha, 0.5),
                        xycoords=('data','axes fraction'), va='center', ha='right',
                        rotation='vertical')
        if tight_layout:
            fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        if inplace:
            plt.show()
        return fig, axs


def run_maxent(G_tau, err, alpha_mesh=None, omega_mesh=None, **kwargs):
    """
    Inputs:
        G_tau - triqs.gf.GfImTime
        err - the "error". If unsure, choose a number like 1e-4.
        alpha_mesh - triqs_maxent.alpha_meshes.BaseAlphaMesh or child
            (default LogAlphaMesh(alpha_min=0.0001, alpha_max=20, n_points=20),
            but it gets scaled by number of data points.).
        omega_mesh - triqs_maxent.omega_meshes.BaseOmegaMesh or child
            (default HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100))
        kwargs - keyword arguments for TauMaxEnt
    Output:
        MaxEntResult
    """
    tm = me.TauMaxEnt(cost_function='bryan', alpha_mesh=alpha_mesh, probability='normal', **kwargs)
    tm.set_G_tau(G_tau)
    tm.set_error(err)
    if omega_mesh is not None:
        tm.omega = omega_mesh
    return tm.run()


def get_last_G_tau(archive):
    """
    With a HDFArchive from dmft.dmft, with data from DMFT in 'loop-XXX' group,
    obtain the G_tau of the last loop.

    Input: archive - HDFArchive, or path to a hdf5 file.
    Output: saved G_tau, which is likely BlockGf of GfImTime.
    """
    # If archive is path to an archive, open it.
    if isinstance(archive, str):
        with HDFArchive(archive, 'r') as A:
            return get_last_G_tau(A)
    else:
        # Archive is a HDFArchive instance.
        # Extract the loop keys
        key = get_last_loop(archive)
        # The last key is the one we want
        return archive[key]['G_tau']

def get_last_maxent(archive):
    """
    With a HDFArchive from dmft.maxent, loads the last maxent analysis to be
    written.
    IF YOU WANT A MaxEnt instance from dmft.maxent, call MaxEnt.load without
    a path argument.
    
    Input: archive - HDFArchive, or path to a hdf5 file.
    Output: saved MaxEntResultsData object
    """
    # If archive is path to an archive, open it.
    if isinstance(archive, str):
        with HDFArchive(archive, 'r') as A:
            return get_last_maxent(A)
    # Open the maxent group
    SG = archive['maxent']
    # Find the last analysis
    # Lexicographical sort of keys
    anlist = sorted([k for k in SG if k[0:9] == 'analysis_'])
    # I set two digits for analysis. If no three-digit results, sorting is fine
    if len(anlist) <= 100:
        key = anlist[-1]
    else:
        # Otherwise, cut the shorter ones and take the latest three-digit result
        key = [k for k in anlist if len(k) > 11][-1]
        # I assume that we have no more than a thousand analyses.
    # Return the results
    return SG[key]['results']


def write_metadata(archive, name='code'):
    """Records maxent version information to a HDF5 archive."""
    h5_write_full_path(archive, me.version, name+'/triqs_maxent_version')
    h5_write_full_path(archive, dmft.version.get_git_hash(), name+'/dmft_maxent_version')



if __name__ == "__main__":
    # Put this import here to avoid circular import
    from dmft.measure import standard_deviation_from_archive
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Does a MaxEnt calculation ")
    parser.add_argument("input", help="Input archive containing G_tau from DMFT.")
    parser.add_argument("output", nargs="?",
            help="Archive to write MaxEntResult to. Defaults in input, which I strongly recommend.")
    parser.add_argument('-e','--error', type=float, default=1e-4, help="Error in G_tau")
    parser.add_argument('--amin', type=float, help="Minimum alpha")
    parser.add_argument('--amax', type=float, help="Maximum alpha")
    parser.add_argument('--nalpha', type=int, help="Number of alpha points")
    parser.add_argument('--omin', type=float, help="Minimum omega")
    parser.add_argument('--omax', type=float, help="Maximum omega")
    parser.add_argument('--nomega', type=int, help="Number of omega points")
    parser.add_argument('-b','--block', default='up', help="Block of G_tau to analytically continue.")
    parser.add_argument('-i','--index', type=int, default=0, help="(Diagonal) index of G_tau to analytically continue.")
    parser.add_argument('-d','--digits', type=int, default=2,
            help="Number of digits (with leading zeros) for the results index.")
    parser.add_argument('-n','--name', default='maxent/analysis',
            help="Group to record MaxEnt results in (will get numbers appended).")
    parser.add_argument('--nonumber', action='store_true',
            help="Do not automatically add a number to the name if name already unique")
    parser.add_argument('-l','--loops', nargs='+', type=int,
            help="Take G_tau from these loops and do MaxEnt on each of them (defaults to only the last one).")
    parser.add_argument('-s','--substrate', action='store_true',
            help="Read G_sub_tau instead of G_tau")
    parser.add_argument('--std', type=int,
            help="Rather than using a fixed error, read the standard deviation of the last few G_tau's.")
    parser.add_argument('-g','--gaussian', type=float,
            help="Rather than a flat default model, use a Gaussian of given width.")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input
    
    if args.substrate:
        gname = 'G_sub_tau'
    else:
        gname = 'G_tau'

    # Get G_tau and the loop numbers
    if args.loops is None:
        # Take the last one
        # Record the dmft_loop that is from, for our records.
        dmft_loop_list = [get_last_loop(args.input)]
        # Load G_tau, taking one of the blocks.
        G_tau_list = [h5_read_full_path(args.input, dmft_loop_list[0]+'/'+gname)[args.block]]
    else:
        # Gather up all the G_tau's for the specified loops.
        dmft_loop_list = []
        G_tau_list = []
        for loop in args.loops:
            dmft_loop = 'loop-{:03d}'.format(loop)
            try:
                G_tau = h5_read_full_path(args.input, dmft_loop+'/'+gname)[args.block][args.index,args.index]
            except KeyError:
                warn(dmft_loop+'/'+gname+' not found')
            else:
                dmft_loop_list.append(dmft_loop)
                G_tau_list.append(G_tau)
    for (G_tau, dmft_loop) in zip(G_tau_list, dmft_loop_list):
        # Let us go through the output and find a unique name to write to
        name = find_unique_name(args.output, args.name, digits=args.digits, always_number=(not args.nonumber))
        if mpi.is_master_node():
            print("Processing", gname, "from", dmft_loop)
            print("Writing MaxEnt data to", name)
        # Generate MaxEnt object with its alpha mesh
        maxent = MaxEnt(amin=args.amin, amax=args.amax, nalpha=args.nalpha)
        # Get the omega mesh
        if args.omin is not None:
            if args.omax is not None and args.nomega is not None:
                maxent.generate_omega(args.omin, args.omax, args.nomega)
            else:
                warn("Must specify all of omin, omax, and nomega; omega_mesh unset")
        elif args.omax is not None or args.nomega is not None:
            warn("Must specify all of omin, omax, and nomega; omega_mesh unset")
        # Generate the default model
        if args.gaussian is not None:
            maxent.set_Gaussian_D(args.gaussian)
        # Set other MaxEnt data
        maxent.set_G_tau(G_tau)
        if args.std is not None:
            error = standard_deviation_from_archive(args.input, args.std,
                        real=True, Gf='G_tau')
        else:
            error = args.error
        maxent.set_error(error)
        # Record the metadata
        if mpi.is_master_node():
            maxent.write_metadata(args.output, name)
            h5_write_full_path(args.output, dmft_loop, name+'/dmft_loop')
            if args.substrate:
                h5_write_full_path(args.output, args.block+'_sub', name+'/block')
            else:
                h5_write_full_path(args.output, args.block, name+'/block')
            h5_write_full_path(args.output, args.index, name+'/index')
        # Run the MaxEnt calculation
        maxent.run()
        # Record the result
        if mpi.is_master_node():
            maxent.save(args.output, name+'/results')

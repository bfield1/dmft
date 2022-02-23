"""
Runs basic MaxEnt processing
"""
import argparse
from warnings import warn
import functools

import numpy as np
import matplotlib.pyplot as plt

import triqs.gf as gf # Needed to import the DMFT data
from h5 import HDFArchive
import triqs_maxent as me
import triqs.utility.mpi as mpi

import dmft.version
from dmft.utils import h5_write_full_path, h5_read_full_path

def wrap_plot(func):
    @functools.wraps(func)
    def plotter(*args, ax=None, inplace=True, **kwargs):
        if ax is not None:
            plt.sca(ax)
        func(*args, **kwargs)
        if inplace is True:
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
    def plotter(*args, ax=None, emin=None, emax=None, amax=None, inplace=True, axeslabels=True, **kwargs):
        """
        Plotter keyword arguments:
            ax - matplotlib Axes
            emin, emax - floats, x-limits
            amax - float, y-limit (the other is 0)
            inplace - Boolean, whether or not to plt.show(). If True, returned
                objects have undefined behaviour.
            axeslabels - Boolean, whether or not to set the axes labels.
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
                inplace=inplace, axeslabels=axeslabels)
        # Adjust axes limits
        ax.set_xlim(emin, emax)
        ax.set_ylim(0, amax)
        # Draw axes labels
        if axeslabels:
            ax.set_xlabel(r'$\omega$')
            ax.set_ylabel(r'$A(\omega)$')
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
    def set_G_tau(self, G_tau):
        """Sets the G_tau for MaxEnt"""
        self.tm.set_G_tau(G_tau)
    def load_G_tau(self, archive, block='up', index=0):
        """Loads a G_tau from the last DMFT loop in a HDF5 archive"""
        G_tau_block = get_last_G_tau(archive)
        self.set_G_tau(G_tau_block[block][index])
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
    def load(cls, archive, path):
        """
        Reads saved MaxEnt results from path in a HDF5 archive

        Compatible with any triqs_maxent.MaxEntResultData, not just those
        created by the class.
        Does not set the error, as that doesn't save.
        Also does not set self.results, which is a MaxEntResult object,
        as only MaxEntResultData is saved. But we only need the latter.
        """
        data = h5_read_full_path(archive, path)
        self = cls(alpha_mesh=data.alpha)
        self.data = data
        self.tm.set_G_tau_data(data.data_variable, data.G_orig)
        self.omega = data.omega
        return self
    def write_metadata(self, archive):
        """Writes version data to a predetermined location in a HDF5 archive"""
        write_metadata(archive)
    # Now we want some plotting scripts to show the results
    @spectrum_plotter
    def plot_spectrum(self, choice=None, **kwargs):
        """
        Plots the spectral function

        Inputs:
            choice - integer or string. If integer, is an alpha index.
                If string, is an analyzer name (possibly with the trailing
                "Analyzer" dropped).
                If unspecified, reverts to the default analyzer.
        """
        ax = kwargs['ax']
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
        # Plot
        ax.plot(self.omega, A)
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
    if not isinstance(archive, HDFArchive):
        with HDFArchive(archive, 'r') as A:
            return get_last_G_tau(A)
    else:
        # Archive is a HDFArchive instance.
        # Extract the loop keys
        keys = sorted([k for k in archive if k[0:5] == 'loop-'])
        # The last key is the one we want
        return archive[keys[-1]]['G_tau']

def write_metadata(archive):
    """Records maxent version information to a HDF5 archive."""
    h5_write_full_path(archive, me.version, 'code/triqs_maxent_version')
    h5_write_full_path(archive, dmft.version.get_git_hash(), 'code/dmft_maxent_version')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input archive containing G_tau from DMFT.")
    parser.add_argument("output", help="Archive to write MaxEntResult to.")
    parser.add_argument('-n','--name', default="maxent_result", help="Label for MaxEntResult in the output")
    parser.add_argument('-e','--error', type=float, default=1e-4, help="Error in G_tau")
    parser.add_argument('--amin', type=float, help="Minimum alpha")
    parser.add_argument('--amax', type=float, help="Maximum alpha")
    parser.add_argument('--nalpha', type=int, help="Number of alpha points")
    parser.add_argument('--omin', type=float, help="Minimum omega")
    parser.add_argument('--omax', type=float, help="Maximum omega")
    parser.add_argument('--nomega', type=int, help="Number of omega points")
    parser.add_argument('-b','--block', default='up', help="Block of G_tau to analytically continue.")
    args = parser.parse_args()

    # Load G_tau, taking one of the blocks.
    G_tau = get_last_G_tau(args.input)[args.block]
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
    # Set outher MaxEnt data
    maxent.set_G_tau(G_tau)
    maxent.set_error(args.error)
    # Record the metadata
    if mpi.is_master_node():
        write_metadata(args.output)
    # Run the MaxEnt calculation
    maxent.run()
    # Record the result
    if mpi.is_master_node():
        maxent.save(args.output, args.name)

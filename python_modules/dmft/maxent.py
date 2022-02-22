"""
Runs basic MaxEnt processing
"""
import argparse
from warnings import warn

import numpy as np

import triqs.gf as gf # Needed to import the DMFT data
from h5 import HDFArchive
import triqs_maxent as me
import triqs.utility.mpi as mpi

import dmft.version
from dmft.utils import h5_write_full_path, h5_read_full_path

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
        return self.tm.alpha
    @alpha.setter
    def alpha(self, value):
        self.tm.alpha = value
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
        self.results = tm.run()
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
            maxent.generate_omega_mesh(args.omin, args.omax, args.nomega)
        else:
            warn("Must specify all of omin, omax, and nomega; omega_mesh unset")
    elif args.omax is not None or args.nomega is not None:
        warn("Must specify all of omin, omax, and nomega; omega_mesh unset")
    # Set outher MaxEnt data
    maxent.set_G_tau(G_tau)
    maxent.set_error(args.error)
    # Record the metadata
    write_metadata(args.output)
    # Run the MaxEnt calculation
    maxent.run()
    # Record the result
    if mpi.is_master_node():
        maxent.save(args.output, args.name)

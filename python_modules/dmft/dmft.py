"""
Main solver for DMFT
"""

import argparse
import warnings
from subprocess import CalledProcessError
import os.path

import numpy as np

# TRIQS libraries
from h5 import HDFArchive, HDFArchiveInert
import triqs.gf as gf
import triqs.operators as op
import triqs.utility.mpi as mpi
from triqs_cthyb import Solver
import triqs_cthyb.version
import triqs.version

import dmft.dos
import dmft.version

# Hide the code line in the warnings
# This line might come back to bite me if it affects the global scope
# outside this module, though
_original_showwarning = warnings.showwarning
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: _original_showwarning(message, category, filename, lineno, file, line='')

class EnvironmentWarning(UserWarning):
    """For when the execution environment is different to expected"""
    pass

class DMFTHubbard:
    """
    Sets up and performs DMFT loops in the Hubbard model.
    Also holds data, so good for analysis.
    """
    def __init__(self, beta, mu=None, solver_params={}, u=None, nl=None):
        self.last_loop = -1
        self.beta = beta
        if nl is None:
            self.S = Solver(beta = beta, gf_struct = [('up',[0]), ('down',[0])])
        else:
            self.S = Solver(beta = beta, gf_struct = [('up',[0]), ('down',[0])], n_l=nl)
        self.mu = mu # You need to set this manually.
        self.solver_params = solver_params # You need to set this manually too
        # n_cycles, length_cycle, n_warmup_cycles, measure_G_l
        if u is not None:
            self.U = u
    def set_dos(self, rho, energy, delta):
        """
        Record non-interacting DOS from output of dmft.dos.dos_from_band.
        """
        rho = np.asarray(rho)
        energy = np.asarray(energy)
        delta = np.asarray(delta)
        if rho.shape != energy.shape or rho.shape != delta.shape:
            raise ValueError("rho, energy, and delta must have the same shape.")
        self.rho = rho
        self.energy = energy
        self.delta = delta
    @property
    def U(self):
        """
        Hubbard U interaction parameter
        """
        return self._U
    @U.setter
    def U(self, u):
        self._U = u
        self.h_int = u * op.n('up',0) * op.n('down',0)
    def record_metadata(self, A):
        """Records metadata to a pre-opened archive A."""
        if 'params' not in A:
            A.create_group('params')
        SG = A['params']
        SG['U'] = self.U
        SG['mu'] = self.mu
        SG['beta'] = self.beta
        SG['solver_params'] = self.solver_params
        SG['dos'] = dict(rho=self.rho, energy=self.energy, delta=self.delta)
        SG['MPI_ranks'] = mpi.size
        if 'code' not in A:
            A.create_group('code')
        SG = A['code']
        SG['triqs_version'] = triqs.version.version
        SG['cthyb_version'] = triqs_cthyb.version.version
        try:
            SG['dmft_version'] = dmft.version.get_git_hash()
        except CalledProcessError:
            warnings.warn("Unable to get dmft_version")
    def loop(self, n_loops, archive=None, prior_loops=None, save_metadata_per_loop=False):
        """
        Perform DMFT loops, writing results for each loop to an archive

        Inputs:
            n_loops - positive integer, number of loops to do
            archive - str pointing to HDF5 archive to write results to
            prior_loops - positive integer, number of previous loops.
                Defaults to what is internally recorded.
                Used to properly label results in continuation jobs.
            save_metadata_per_loop - Boolean. If True, metadata is saved with each loop.
        Results are stored in the group loop-XXX, where XXX is the loop number.
        """
        if prior_loops is None:
            prior_loops = self.last_loop + 1
        if mpi.is_master_node():
            print("=================\nStarting DMFT loop\n====================")
            if prior_loops > 0:
                print(f"Continuation job from {prior_loops} prior loops.")
        # If we aren't doing a continuation job, set our initial guess for the self-energy
        if prior_loops == 0:
            self.S.Sigma_iw << self.mu
            # Record some metadata
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    self.record_metadata(A)
        # Create a local copy of G as a scratch variable
        G = self.S.G0_iw.copy()
        # Enter the DMFT loop
        for i_loop in range(n_loops):
            if mpi.is_master_node():
                print(f"\n Loop number {i_loop+prior_loops} \n")
            # Do the self-consistency condition
            G.zero()
            for name, _ in G:
                # Hilbert transform: an integral involving the DOS
                for i in range(len(self.rho)):
                    G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(gf.iOmega_n + self.mu - self.energy[i] - self.S.Sigma_iw[name])
            # Get the next impurity G0
            for name, g0 in self.S.G0_iw:
                g0 << gf.inverse(gf.inverse(G[name]) + self.S.Sigma_iw[name])
            # Run the solver
            self.S.solve(h_int=self.h_int, **self.solver_params)
            # record results
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    key = 'loop-{:03d}'.format(i_loop+prior_loops)
                    A.create_group(key)
                    SG = A[key]
                    SG['G_iw'] = self.S.G_iw
                    SG['Sigma_iw'] = self.S.Sigma_iw
                    SG['G0_iw'] = self.S.G0_iw
                    SG['G_tau'] = self.S.G_tau
                    if 'measure_G_l' in self.solver_params and self.solver_params['measure_G_l']:
                        SG['G_l'] = self.S.G_l
                    if save_metadata_per_loop:
                        self.record_metadata(SG)
            self.last_loop = i_loop + prior_loops
        if mpi.is_master_node():
            print("Finished DMFT loop.")
    @classmethod
    def load(cls, archive):
        """
        Load a DMFT instance from an archive.

        You must have run record_metadata and loop to have the right data
        recorded. Although record_metadata is done automatically.
        """
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                return cls.load(A)
        # Get the latest loop
        loop_index, loop = cls._load_get_latest_loop(archive)
        params, code = cls._load_get_params_and_code(archive)
        # Load the relevant data
        beta = params['beta']
        mu = params['mu']
        solver_params = params['solver_params']
        u = params['U']
        try:
            nl = params['nl']
        except KeyError:
            # I need to do this better, as I don't record nl yet.
            nl = None
        # Initialise
        self = cls(beta, mu, solver_params, u, nl)
        # DOS
        self.rho = np.asarray(params['dos']['rho'])
        self.energy = np.asarray(params['dos']['energy'])
        self.delta = np.asarray(params['dos']['delta'])
        # Set the latest loop
        self.last_loop = int(loop_index[5:])
        # Set the Green's functions
        # Especially the self-energy; all else could be skipped if needed
        self.S.Sigma_iw << loop['Sigma_iw']
        try:
            self.S.G_iw << loop['G_iw']
        except KeyError:
            pass
        # G0_iw, G_tau are not writable.
        try:
            self.S.G_l << loop['G_l']
        except KeyError:
            pass
        # Check if the environment has changed and throw appropriate warnings
        try:
            if mpi.size != params['MPI_ranks']:
                warnings.warn("Original calculations were with {0} MPI ranks, but current environment has {1} ranks.".format(mpi.size, params['MPI_ranks']), EnvironmentWarning)
        except KeyError:
            warnings.warn("Unknown MPI ranks in loaded data.", EnvironmentWarning)
        # Check versions
        try:
            if code['triqs_version'] != triqs.version.version:
                warnings.warn("Original calculations were with TRIQS version {0}, but current environment uses TRIQS version {1}.".format(code['triqs_version'], triqs.version.version),
                        EnvironmentWarning)
        except KeyError:
            warnings.warn("Unknown TRIQS version in loaded data.", EnvironmentWarning)
        try:
            if code['cthyb_version'] != triqs_cthyb.version.version:
                warnings.warn("Original calculations were with CTHYB version {0}, but current environment uses CTHYB version {1}.".format(code['cthyb_version'], triqs_cthyb.version.version),
                        EnvironmentWarning)
        except KeyError:
            warnings.warn("Unknown CTHYB version in loaded data.", EnvironmentWarning)
        try:
            current_dmft_version = dmft.version.get_git_hash()
        except CalledProcessError:
            warnings.warn("Unable to get dmft_version.", EnvironmentWarning)
        else:
            try:
                if code['dmft_version'] != current_dmft_version:
                    warnings.warn("Original calculations were with DMFT version {0}, but current environment uses DMFT version {1}.".format(code['dmft_version'], current_dmft_version), EnvironmentWarning)
            except KeyError:
                warnings.warn("Unknown DMFT version in loaded data.", EnvironmentWarning)
        # And we're done
        return self
    @staticmethod
    def _load_get_latest_loop(A):
        """
        Takes an already open archive. Gets latest loop index and group.
        Helper for load
        Input: open h5.HDFArchive
        Outputs: loop_index (str), loop (HDFArchive group)
        """
        loop_index = sorted([s for s in A if s[0:5] == 'loop-'])[-1]
        loop = A[loop_index]
        return loop_index, loop
    @classmethod
    def _load_get_params_and_code(cls, A):
        """
        Takes an already open archive. Gets the most up-to-date metadata. Helper to load
        Input: open h5.HDFArchive
        Outputs: params, code (HDFArchive groups)
        """
        # Get the latest loop
        loop_index, loop = cls._load_get_latest_loop(A)
        # Check if there is metadata in the latest loop.
        # Otherwise, use the default over-arching metadata.
        if 'params' in loop:
            params = loop['params']
        else:
            params = A['params']
        if 'code' in loop:
            code = loop['code']
        else:
            code = A['code']
        return params, code

class DMFTHubbardKagome(DMFTHubbard):
    def set_dos(self, t, offset, nk, bins=None, de=None):
        """Record non-interacting DOS for kagome lattice."""
        rho, energy, delta = dmft.dos.kagome_dos(t, offset, nk, bins, de)
        rho /= 3 # The solver requires it to be normalised to 1
        super().set_dos(rho, energy, delta)
        self.t = t
        self.offset = offset
        self.nk = nk
    def record_metadata(self, A):
        super().record_metadata(A)
        SG = A['params']
        SG['kagome_t'] = self.t
        SG['kagome_nk'] = self.nk
        SG['kagome_offset'] = self.offset
    @classmethod
    def load(cls, archive):
        # Open the archive if not already
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                return cls.load(A)
        # Do all the base class loading
        self = super().load(archive)
        # Load the kagome metadata
        params, code = cls._load_get_params_and_code(archive)
        # No big deal if it isn't there, as it is only for records.
        try:
            self.t = params['kagome_t']
        except KeyError:
            pass
        try:
            self.nk = params['kagome_nk']
        except KeyError:
            pass
        try:
            self.offset = params['kagome_offset']
        except KeyError:
            pass
        return self


class DMFTHubbardBethe(DMFTHubbard):
    def set_dos(self, t, offset, bins=None, de=None):
        """Record non-interacting DOS for Bethe lattice."""
        rho, energy, delta = dmft.dos.bethe_dos(t, offset, bins, de)
        super().set_dos(rho, energy, delta)
        self.t = t
        self.offset = offset
    def record_metadata(self, A):
        super().record_metadata(A)
        SG = A['params']
        SG['bethe_t'] = self.t
        SG['bethe_offset'] = self.offset
    @classmethod
    def load(cls, archive):
        # Open the archive if not already
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                return cls.load(A)
        # Do all the base class loading
        self = super().load(archive)
        # Load the kagome metadata
        params, code = cls._load_get_params_and_code(archive)
        # No big deal if it isn't there, as it is only for records.
        try:
            self.t = params['bethe_t']
        except KeyError:
            pass
        try:
            self.offset = params['bethe_offset']
        except KeyError:
            pass
        return self

if __name__ == "__main__":
    # Set up command line argument parser.
    # The point is to be able to run regular calculations from the command line
    # without having to write a whole script.
    parser = argparse.ArgumentParser(description="Perform a DMFT calculation on the Hubbard model.")
    parser.add_argument('-b','--beta', type=float, help="Inverse temperature.")
    parser.add_argument('-u', type=float, help="Hubbard U")
    parser.add_argument('-m','--mu', type=float, default=0, help="Chemical potential")
    parser.add_argument('-n','--nloops', type=int, required=True, help="Number of DMFT loops.")
    parser.add_argument('-c','--cycles', type=int, default=200000, help="Number of QMC cycles.")
    parser.add_argument('-l','--length', type=int, default=50, help="Length of QMC cycles.")
    parser.add_argument('-w','--warmup', type=int, default=10000, help="Number of warmup QMC cycles.")
    parser.add_argument('-a','--archive', help="Archive to record data to.")
    parser.add_argument('--nl', type=int, help="Number of Legendre polynomials to fit G_l in QMC (if any).")
    parser.add_argument('-o','--overwrite', action='store_true', help="Forcibly overwrite the existing archive. May act a bit unpredictably from merging the data.")

    subparsers = parser.add_subparsers(dest='lattice', help="Which lattice to solve. Or run a continuation job (which ignores all parameters except --archive and --nloops).")

    kagome_parser = subparsers.add_parser('kagome')
    kagome_parser.add_argument('-t', type=float, help="Hopping", default=1)
    kagome_parser.add_argument('--offset', type=float, default=0, help="Offset")
    kagome_parser.add_argument('--nk', type=int, default=2000, help="Number of k-points.")
    kagome_parser.add_argument('--bins', type=int, default=300, help="Number of DOS energy bins.")
    
    bethe_parser = subparsers.add_parser('bethe')
    bethe_parser.add_argument('-t', type=float, help="Hopping", default=1)
    bethe_parser.add_argument('--offset', type=float, default=0, help="Offset")
    bethe_parser.add_argument('--bins', type=int, default=200, help="Number of DOS energy bins.")
    
    subparsers.add_parser('continue')

    args = parser.parse_args()
    
    # Is this a continuation job?
    continuation = args.lattice == 'continue'
    changed = False
    # Check the existence of the archive and if we should overwrite it.
    if os.path.isfile(args.archive) and not args.overwrite and not continuation:
        raise FileExistsError(f"The archive {args.archive} already exists. Maybe you want to --overwrite it or 'continue' an existing job?")
    # Initialise the solver.
    if args.lattice == 'continue':
        # While I could go to the extent to extracting t and offset from the 
        # respective lattices, it isn't necessary, so I won't bother.
        with warnings.catch_warnings(record=True) as w:
            hubbard = DMFTHubbard.load(args.archive)
            # Count if any EnvironmentWarnings were raised.
            w2 = [warn for warn in w if issubclass(warn.category, EnvironmentWarning)]
            if len(w2) > 0:
                changed = True
            else:
                # There is another instance in which we want changed==True
                # When we are already using a modified parameter set
                with HDFArchive(args.archive, 'r') as A:
                    loop_index, loop = DMFTHubbard._load_get_latest_loop(A)
                    if 'params' in loop or 'code' in loop:
                        changed = True
                    # Close/un-bind the HDFArchive group
                    # (otherwise cannot write to archive later)
                    del loop
    # No continuation job. Go ahead with existing lattices.
    elif args.lattice == 'kagome':
        hubbard = DMFTHubbardKagome(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl)
        hubbard.set_dos(t=args.t, offset=args.offset, nk=args.nk, bins=args.bins)
    elif args.lattice == 'bethe':
        hubbard = DMFTHubbardBethe(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl)
        hubbard.set_dos(t=args.t, offset=args.offset, bins=args.bins)
    else:
        raise ValueError(f"Unrecognised lattice {args.lattice}.")
    # If a new job, set the solver params.
    if not continuation:
        hubbard.solver_params = dict(n_cycles=args.cycles, length_cycle=args.length, n_warmup_cycles=args.warmup)
        if args.nl is not None:
            hubbard.solver_params['measure_G_l'] = True
    # Run the loop
    hubbard.loop(args.nloops, archive=args.archive, save_metadata_per_loop=changed)

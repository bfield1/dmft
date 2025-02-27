"""
Main solver for DMFT

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
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
import dmft.utils

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
    def __init__(self, beta, mu=None, solver_params={}, u=None, nl=None, n_iw=1025, n_tau=10001):
        self.last_loop = -1
        self.beta = beta
        kwargs = dict(beta=beta, n_iw=n_iw, n_tau=n_tau)
        if nl is not None:
            kwargs['n_l'] = nl
        # Triqs version 3.1.0 changed how to specify gf_struct.
        version = tuple(int(x) for x in triqs.version.version.split('.'))
        if version < (3,1,0):
            kwargs['gf_struct'] = [('up',[0]), ('down',[0])]
        else:
            kwargs['gf_struct'] = [('up',1), ('down',1)]
        self.S = Solver(**kwargs)
        self.mu = mu # You need to set this manually.
        self.solver_params = solver_params # You need to set this manually too
        self._n_iw = n_iw # Record for ease of access
        self._n_tau = n_tau # Record for each of access
        self._nl = nl # Record for ease of access
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
        SG['n_iw'] = self._n_iw
        SG['n_tau'] = self._n_tau
        if self._nl is not None:
            SG['nl'] = self._nl
        if 'code' not in A:
            A.create_group('code')
        SG = A['code']
        SG['triqs_version'] = triqs.version.version
        SG['cthyb_version'] = triqs_cthyb.version.version
        try:
            SG['dmft_version'] = dmft.version.get_git_hash()
        except CalledProcessError:
            warnings.warn("Unable to get dmft_version")
    def loop(self, n_loops, archive=None, prior_loops=None, save_metadata_per_loop=False, enforce_spins=True):
        """
        Perform DMFT loops, writing results for each loop to an archive

        Inputs:
            n_loops - positive integer, number of loops to do
            archive - str pointing to HDF5 archive to write results to
            prior_loops - positive integer, number of previous loops.
                Defaults to what is internally recorded.
                Used to properly label results in continuation jobs.
            save_metadata_per_loop - Boolean. If True, metadata is saved with
                each loop.
            enforce_spins - Boolean. If True, forces spin up and down sectors
                to be the same. Enhances numerical stability.
        Results are stored in the group loop-XXX, where XXX is the loop number.
        """
        if prior_loops is None:
            prior_loops = self.last_loop + 1
        if mpi.is_master_node():
            print("====================\nStarting DMFT loop\n====================")
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
            # Symmetrise the spin
            if enforce_spins:
                sigma = 0.5 * (self.S.Sigma_iw['up'] + self.S.Sigma_iw['down'])
                sigma = dict(up=sigma, down=sigma)
            else:
                sigma = self.S.Sigma_iw
            # Do the self-consistency condition
            G.zero()
            for name, _ in G:
                # Hilbert transform: an integral involving the DOS
                for i in range(len(self.rho)):
                    G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(gf.iOmega_n + self.mu - self.energy[i] - sigma[name])
            # Get the next impurity G0
            for name, g0 in self.S.G0_iw:
                g0 << gf.inverse(gf.inverse(G[name]) + sigma[name])
            # Run the solver
            if 'random_seed' in self.solver_params:
                # I don't advise setting a fixed random seed in solver_params,
                # because it probably won't be different on each mpi rank
                self.S.solve(h_int=self.h_int, **self.solver_params)
            else:
                # I'm adding a slight change to the random seed in each loop
                # This ensures that different loops use different random numbers
                # But also allows consistency.
                self.S.solve(h_int=self.h_int, **self.solver_params,
                      random_seed=34788+928374*mpi.rank+17*(i_loop+prior_loops))
            # Symmetrise output
            if enforce_spins:
                g = self.S.G_iw['up'].copy()
                g << (self.S.G_iw['up'] + self.S.G_iw['down'])/2
                for name in ['up','down']:
                    self.S.G_iw[name] << g
                g << (self.S.Sigma_iw['up'] + self.S.Sigma_iw['down'])/2
                for name in ['up','down']:
                    self.S.Sigma_iw[name] << g
                g = self.S.G_tau['up'].copy()
                g << (self.S.G_tau['up'] + self.S.G_tau['down'])/2
                for name in ['up','down']:
                    self.S.G_tau[name] << g
                if 'measure_G_l' in self.solver_params and self.solver_params['measure_G_l']:
                    g = self.S.G_l['up'].copy()
                    g << (self.S.G_l['up'] + self.S.G_l['down'])/2
                    for name in ['up','down']:
                        self.S.G_l[name] << g
            # record results
            self._record_loop_data(archive, i_loop+prior_loops, save_metadata_per_loop)
            self.last_loop = i_loop + prior_loops
        if mpi.is_master_node():
            print("Finished DMFT loop.")
    #
    def _record_loop_data(self, archive, i_loop, *args, **kwargs):
        """
        Helper function for loop. Writes loop data to archive

        Inputs: archive - str, name of h5 archive
            i_loop - int, loop number to record to
            And any other arguments to pass on the _record_loop_data_inner.
        """
        if archive is not None and mpi.is_master_node():
            with HDFArchive(archive,'a') as A:
                key = 'loop-{:03d}'.format(i_loop)
                A.create_group(key)
                self._record_loop_data_inner(A[key], *args, **kwargs)
    def _record_loop_data_inner(self, SG, save_metadata):
        """
        Helper function for _record_loop_data

        For subclasses which modify the loop, you can call this with super().

        Inputs: SG - HDFArchiveGroup, the group to write to
            save_metadata - Boolean, whether to save metadata
        """
        SG['G_iw'] = self.S.G_iw
        SG['Sigma_iw'] = self.S.Sigma_iw
        SG['G0_iw'] = self.S.G0_iw
        SG['G_tau'] = self.S.G_tau
        if 'measure_G_l' in self.solver_params and self.solver_params['measure_G_l']:
            SG['G_l'] = self.S.G_l
        SG['average_sign'] = self.S.average_sign
        if 'measure_density_matrix' in self.solver_params and self.solver_params['measure_density_matrix']:
            SG['density_matrix'] = self.S.density_matrix
            SG['h_loc_diagonalization'] = self.S.h_loc_diagonalization
        if 'measure_pert_order' in self.solver_params and self.solver_params['measure_pert_order']:
            SG['perturbation_order'] = self.S.perturbation_order
            SG['perturbation_order_total'] = self.S.perturbation_order_total
        if 'measure_O_tau' in self.solver_params:
            SG['O_tau'] = self.S.O_tau
        if save_metadata:
            self.record_metadata(SG)
    @classmethod
    @dmft.utils.archive_reader2
    def load(cls, archive):
        """
        Load a DMFT instance from an archive.

        You must have run record_metadata and loop to have the right data
        recorded. Although record_metadata is done automatically.
        """
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
            # Default value
            nl = None
        try:
            n_iw = params['n_iw']
        except KeyError:
            # Default value
            n_iw = 1025
        try:
            n_tau = params['n_tau']
        except KeyError:
            # Default value
            n_tau = 10001
        # Initialise
        self = cls(beta, mu, solver_params, u, nl=nl, n_iw=n_iw, n_tau=n_tau)
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
                warnings.warn("Original calculations were with {1} MPI ranks, but current environment has {0} ranks.".format(mpi.size, params['MPI_ranks']), EnvironmentWarning)
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
    @dmft.utils.archive_reader2
    def load(cls, archive):
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
    @dmft.utils.archive_reader2
    def load(cls, archive):
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
    # Import here to reduce risk of circular dependencies,
    # and also because the substrates logic is exactly the same.
    import dmft.dmftsubstrate
    # Set up command line argument parser.
    # The point is to be able to run regular calculations from the command line
    # without having to write a whole script.
    parser = argparse.ArgumentParser(description="Perform a DMFT calculation on the Hubbard model.")
    parser.add_argument('-b','--beta', type=float, help="Inverse temperature.")
    parser.add_argument('-u', type=float, help="Hubbard U")
    parser.add_argument('-m','--mu', type=float, default=0, help="Chemical potential")
    parser.add_argument('-n','--nloops', type=int, required=True,
            help="Number of DMFT loops.")
    parser.add_argument('-c','--cycles', type=int, default=200000,
            help="Number of QMC cycles.")
    parser.add_argument('-l','--length', type=int, default=50,
            help="Length of QMC cycles.")
    parser.add_argument('-w','--warmup', type=int, default=10000,
            help="Number of warmup QMC cycles.")
    parser.add_argument('-a','--archive', help="Archive to record data to.")
    parser.add_argument('--nl', type=int,
            help="Number of Legendre polynomials to fit G_l in QMC (if any).")
    parser.add_argument('-o','--overwrite', action='store_true',
            help="Forcibly overwrite the existing archive. May act a bit unpredictably from merging the data.")
    parser.add_argument('-V', type=float,
            help="Coupling V between substrate and Hubbard. Only use if want a substrate.")
    parser.add_argument('--bandwidth', type=float,
            help="Substrate bandwidth. Only use if want a substrate.")
    parser.add_argument('-d','--density', action='store_true',
            help="Record the density matrix.")
    parser.add_argument('--spin', action='store_true',
            help="Measure the dynamical spin-spin susceptibility.")
    parser.add_argument('--motmi', type=int, default=100,
            help="measure_O_tau_min_ins")
    parser.add_argument('--niw', type=int, default=1025,
            help="Number of Matsubara frequencies used for the Green's function.")
    parser.add_argument('--ntau', type=int, default=10001,
            help="Number of imaginary time points used for the Green's function. Should be at least 6 times niw.")
    parser.add_argument('-p','--perturbation', action='store_true',
            help="Measure the perturbation order histograms.")

    subparsers = parser.add_subparsers(dest='lattice',
            help="Which lattice to solve. Or run a continuation job (which ignores all parameters except --archive and --nloops).")

    kagome_parser = subparsers.add_parser('kagome')
    kagome_parser.add_argument('-t', type=float, help="Hopping", default=1)
    kagome_parser.add_argument('--offset', type=float, default=0, help="Offset")
    kagome_parser.add_argument('--nk', type=int, default=2000, help="Number of k-points.")
    kagome_parser.add_argument('--bins', type=int, default=300, help="Number of DOS energy bins.")
    
    bethe_parser = subparsers.add_parser('bethe')
    bethe_parser.add_argument('-t', type=float, help="Hopping", default=1)
    bethe_parser.add_argument('--offset', type=float, default=0, help="Offset")
    bethe_parser.add_argument('--bins', type=int, default=200, help="Number of DOS energy bins.")

    impurity_parser = subparsers.add_parser('impurity')
    impurity_parser.add_argument('-e','--energy', type=float, help="Impurity energy", default=0)
    
    continue_parser = subparsers.add_parser('continue')
    continue_parser.add_argument('--substrate', action='store_true',
            help="Continuation job of a system with a substrate.")
    continue_parser.add_argument('--newparams', action='store_true',
            help="Create new solver_params.")

    args = parser.parse_args()
    
    # Is this a continuation job?
    continuation = args.lattice == 'continue'
    # Is this a substrate job?
    if (continuation and args.substrate) or (args.V is not None) or (args.bandwidth is not None) or (args.lattice == 'impurity'):
        substrate = True
        # Validation of supplied arguments
        if not continuation and (args.V is None or args.bandwidth is None):
            raise TypeError("Must include both -V and --bandwidth, not just one.")
    else:
        substrate = False
    changed = False
    # Have the parameters changed?
    if continuation and args.newparams:
        changed = True
    # Check the existence of the archive and if we should overwrite it.
    if os.path.isfile(args.archive) and not args.overwrite and not continuation and mpi.is_master_node():
        # It might not have DMFT data in it, though, just some logs. So check
        with HDFArchive(args.archive, 'r') as A:
            if 'loop-000' in A:
                raise FileExistsError(f"The archive {args.archive} already exists and contains DMFT data. Maybe you want to --overwrite it or 'continue' an existing job?")
    # Initialise the solver.
    if args.lattice == 'continue':
        # While I could go to the extent to extracting t and offset from the 
        # respective lattices, it isn't necessary, so I won't bother.
        with warnings.catch_warnings(record=True) as w:
            if substrate:
                cls = dmft.dmftsubstrate.DMFTHubbardSubstrate
            else:
                cls = DMFTHubbard
            hubbard = cls.load(args.archive)
            # Count if any EnvironmentWarnings were raised.
            w2 = [warn for warn in w if issubclass(warn.category, EnvironmentWarning)]
            if len(w2) > 0:
                changed = True
            else:
                # There is another instance in which we want changed==True
                # When we are already using a modified parameter set
                with HDFArchive(args.archive, 'r') as A:
                    loop_index, loop = cls._load_get_latest_loop(A)
                    if 'params' in loop or 'code' in loop:
                        changed = True
                    # Close/un-bind the HDFArchive group
                    # (otherwise cannot write to archive later)
                    del loop
    # No continuation job. Go ahead with existing lattices.
    elif args.lattice == 'kagome':
        if substrate:
            cls = dmft.dmftsubstrate.DMFTHubbardSubstrateKagome
        else:
            cls = DMFTHubbardKagome
        hubbard = cls(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl, n_iw=args.niw, n_tau=args.ntau)
        hubbard.set_dos(t=args.t, offset=args.offset, nk=args.nk, bins=args.bins)
    elif args.lattice == 'bethe':
        if substrate:
            cls = dmft.dmftsubstrate.DMFTHubbardSubstrateBethe
        else:
            cls = DMFTHubbardBethe
        hubbard = cls(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl, n_iw=args.niw, n_tau=args.ntau)
        hubbard.set_dos(t=args.t, offset=args.offset, bins=args.bins)
    elif args.lattice == 'impurity':
        cls = dmft.dmftsubstrate.DMFTHubbardSubstrateImpurity
        hubbard = cls(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl, n_iw=args.niw, n_tau=args.ntau)
        hubbard.set_dos(args.energy)
    else:
        raise ValueError(f"Unrecognised lattice {args.lattice}.")
    # If a new job or if requested, set the solver params
    if not continuation or (continuation and args.newparams):
        hubbard.solver_params = dict(n_cycles=args.cycles, length_cycle=args.length,
                n_warmup_cycles=args.warmup, measure_density_matrix=args.density,
                use_norm_as_weight=args.density)
        if args.nl is not None:
            hubbard.solver_params['measure_G_l'] = True
        # Dynamical spin susceptibility
        # https://triqs.github.io/cthyb/latest/guide/dynamic_susceptibility_notebook.html
        if args.spin:
            Sz = 0.5 * (op.n('up',0) - op.n('down',0))
            hubbard.solver_params['measure_O_tau'] = (Sz, Sz)
            hubbard.solver_params['measure_O_tau_min_ins'] = args.motmi
        if args.perturbation:
            hubbard.solver_params['measure_pert_order'] = args.perturbation
    # If a new job, set the substrate params
    if substrate and not continuation:
        hubbard.set_substrate(args.bandwidth)
        hubbard.V = args.V
    # Grab substrate-only kwargs for loop
    loops_kwargs = dict()
    if substrate:
        pass
    # Run the loop
    hubbard.loop(args.nloops, archive=args.archive, save_metadata_per_loop=changed, **loops_kwargs)

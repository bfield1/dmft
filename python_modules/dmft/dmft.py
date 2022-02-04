"""
Main solver for DMFT
"""

import argparse
import warnings
from subprocess import CalledProcessError

import numpy as np

# TRIQS libraries
from h5 import HDFArchive
import triqs.gf as gf
import triqs.operators as op
import triqs.utility.mpi as mpi
from triqs_cthyb import Solver
import triqs_cthyb.version
import triqs.version

import dmft.dos
import dmft.version

class DMFTHubbard:
    """
    Sets up and performs DMFT loops in the Hubbard model.
    Also holds data, so good for analysis.
    """
    def __init__(self, beta, mu=None, solver_params={}, u=None, nl=None):
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
        A.create_group('params')
        SG = A['params']
        SG['U'] = self.U
        SG['mu'] = self.mu
        SG['beta'] = self.beta
        SG['solver_params'] = self.solver_params
        SG['dos'] = dict(rho=self.rho, energy=self.energy, delta=self.delta)
        SG['MPI_ranks'] = mpi.size
        A.create_group('code')
        SG = A['code']
        SG['triqs_version'] = triqs.version.version
        SG['cthyb_version'] = triqs_cthyb.version.version
        try:
            SG['dmft_version'] = dmft.version.get_git_hash()
        except CalledProcessError:
            warnings.warn("Unable to get dmft_version")
    def loop(self, n_loops, archive=None, prior_loops=0):
        if mpi.is_master_node():
            print("=================\nStarting DMFT loop\n====================")
            if prior_loops > 0:
                print(f"Continuation job from {prior_loops} number of prior loops.")
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
                    if 'measure_G_l' in self.solver_params and self.solver_params['measure_G_l']:
                        SG['G_l'] = self.S.G_l
        if mpi.is_master_node():
            print("Finished DMFT loop.")

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

if __name__ == "__main__":
    # Set up command line argument parser.
    # The point is to be able to run regular calculations from the command line
    # without having to write a whole script.
    parser = argparse.ArgumentParser(description="Perform a DMFT calculation on the Hubbard model.")
    parser.add_argument('-b','--beta', type=float, required=True, help="Inverse temperature.")
    parser.add_argument('-u', type=float, required=True, help="Hubbard U")
    parser.add_argument('-m','--mu', type=float, default=0, help="Chemical potential")
    parser.add_argument('-n','--nloops', type=int, required=True, help="Number of DMFT loops.")
    parser.add_argument('-c','--cycles', type=int, default=20000, help="Number of QMC cycles.")
    parser.add_argument('-l','--length', type=int, default=50, help="Length of QMC cycles.")
    parser.add_argument('-w','--warmup', type=int, default=10000, help="Number of warmup QMC cycles.")
    parser.add_argument('-a','--archive', help="Archive to record data to.")
    parser.add_argument('--nl', type=int, help="Number of Legendre polynomials to fit G_l in QMC (if any).")

    subparsers = parser.add_subparsers(dest='lattice', help="Which lattice to solve.")

    kagome_parser = subparsers.add_parser('kagome')
    kagome_parser.add_argument('-t', type=float, help="Hopping", default=1)
    kagome_parser.add_argument('--offset', type=float, default=0, help="Offset")
    kagome_parser.add_argument('--nk', type=int, default=1000, help="Number of k-points.")
    kagome_parser.add_argument('--bins', type=int, default=50, help="Number of DOS energy bins.")
    
    args = parser.parse_args()

    # Initialise the solver.
    if args.lattice == 'kagome':
        hubbard = DMFTHubbardKagome(beta=args.beta, u=args.u, mu=args.mu, nl=args.nl)
        hubbard.set_dos(t=args.t, offset=args.offset, nk=args.nk, bins=args.bins)
    else:
        raise ValueError(f"Unrecognised lattice {args.lattice}.")
    hubbard.solver_params = dict(n_cycles=args.cycles, length_cycle=args.length, n_warmup_cycles=args.warmup)
    if args.nl is not None:
        hubbard.solver_params['measure_G_l'] = True
    # Run the loop
    hubbard.loop(args.nloops, archive=args.archive)

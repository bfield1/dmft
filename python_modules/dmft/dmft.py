"""
Main solver for DMFT
"""

import numpy as np
from h5 import HDFArchive
import triqs.gf as gf
import triqs.operators as op
import triqs.utility.mpi as mpi
from triqs_cthyb import Solver
import triqs_cthyb.version
import triqs.version

import dmft.dos

class DMFTHubbard:
    """
    Sets up and performs DMFT loops in the Hubbard model.
    Also holds data, so good for analysis.
    """
    def __init__(self, beta, mu=None, solver_params={}, u=None):
        self.beta = beta
        self.S = Solver(beta = beta, gf_struct = [('up',[0]), ('down',[0])])
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
        A['U'] = self.U
        A['mu'] = self.mu
        A['beta'] = self.beta
        A['solver_params'] = self.solver_params
        A['dos'] = dict(rho=self.rho, energy=self.energy, delta=self.delta)
        A['triqs_version'] = triqs.version.version
        A['cthyb_version'] = triqs_cthyb.version.version
    def loop(self, n_loops, archive=None, prior_loops=0):
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
            # Do the self-consistency condition
            G.zero()
            for name, _ in G:
                # Hilbert transform: an integral involving the DOS
                for i in range(len(self.rho)):
                    G[name] += self.rho[i] * self.delta[i] * gf.inverse(gf.iOmega_n + self.mu - self.energy[i] - self.S.Sigma_iw[name])
            # Get the next impurity G0
            for name, g0 in self.S.G0_iw:
                g0 << gf.inverse(gf.inverse(G[name]) + self.S.Sigma_iw[name])
            # Run the solver
            self.S.solve(h_int=self.h_int, **self.solver_params)
            # record results
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    A[f'G_iw-{i_loop+prior_loops}'] = self.S.G_iw
                    A[f'Sigma_iw-{i_loop+prior_loops}'] = self.S.Sigma_iw
                    A[f'G0_iw-{i_loop+prior_loops}'] = self.S.G0_iw

class DMFTHubbardKagome(DMFTHubbard):
    def set_dos(self, t, offset, nk, bins=None, de=None):
        """Record non-interacting DOS for kagome lattice."""
        super().set_dos(*dmft.dos.kagome_dos(t, offset, nk, bins, de))
        self.t = t
        self.offset = offset
        self.nk = nk
    def record_metadata(self, A):
        super().record_metadata(A)
        A['kagome_t'] = self.t
        A['kagome_nk'] = self.nk
        A['kagome_offset'] = self.offset


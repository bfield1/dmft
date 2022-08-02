"""
DMFT solver for a system with a substrate
"""

import numpy as np

import triqs.gf as gf
import triqs.operators as op
import triqs.utility.mpi as mpi
from triqs_cthyb import Solver
from h5 import HDFArchive, HDFArchiveGroup

import dmft.dmft
import dmft.utils

class DMFTHubbardSubstrate(dmft.dmft.DMFTHubbard):
    """
    Sets up and performs DMFT loops in a Hubbard model coupled to a substrate

    I assume the substrate Green's function is local. And substrate DOS is flat.
    """
    def set_substrate(self, bandwidth):
        self.bandwidth = bandwidth
    @property
    def V(self):
        """Substrate coupling V"""
        return self._V
    @V.setter
    def V(self, v):
        self._V = v
    def record_metadata(self, A):
        super().record_metadata(A)
        # Open the params group
        SG = A['params']
        # Create the substrate sub-group if it doesn't exist
        if 'substrate' not in SG:
            SG.create_group('substrate')
        SG2 = SG['substrate']
        # Record substrate metadata
        SG2['bandwidth'] = self.bandwidth
        SG2['V'] = self.V
    @classmethod
    @dmft.utils.archive_reader2
    def load(cls, archive):
        # Do all the base class loading
        self = super().load(archive)
        # Load the metadata
        params, code = cls._load_get_params_and_code(archive)
        # Load the substrate parameters
        self.set_substrate(params['substrate']['bandwidth'])
        self.V = params['substrate']['V']
        return self
    def loop(self, n_loops, archive=None, prior_loops=None,
            save_metadata_per_loop=False, enforce_spins=True, save_Gsub=True):
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
            save_Gsub - Boolean. If True, save full substrate Green's function
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
        # Generate the non-interacting substrate Green's function
        Gsub = self.S.G0_iw['up'].copy()
        Gsub << gf.Flat(self.bandwidth/2)
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
                    # Do the Reimann sum of the system.
                    G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(gf.iOmega_n + self.mu - self.energy[i] - sigma[name] - self.V**2 * Gsub)
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
            self._record_loop_data(archive, i_loop+prior_loops,
                    save_metadata=save_metadata_per_loop, save_Gsub=save_Gsub)
            self.last_loop = i_loop + prior_loops
        if mpi.is_master_node():
            print("Finished DMFT loop.")
    def _record_loop_data_inner(self, SG, save_metadata, save_Gsub):
        super()._record_loop_data_inner(SG, save_metadata)
        if save_Gsub:
            SG['G_sub_iw'] = self.Gsub_full()
            g = self.S.G_tau.copy()
            g << gf.Fourier(self.Gsub_full())
            SG['G_sub_tau'] = g

    def Gsub_full(self):
        """
        Return the full substrate Green's function in ImFreq.
        """
        G = self.S.G_iw.copy()
        # Generate the non-interacting substrate Green's function
        Gsub = self.S.G0_iw['up'].copy()
        Gsub << gf.Flat(self.bandwidth/2)
        # Do the self-consistency condition
        G.zero()
        for name, _ in G:
            # Hilbert transform: an integral involving the DOS
            for i in range(len(self.rho)):
                # Do the Reimann sum of the system.
                G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(gf.inverse(Gsub) - self.V**2 * gf.inverse(gf.iOmega_n + self.mu - self.energy[i] - self.S.Sigma_iw[name]))
        return G


class DMFTHubbardSubstrateBethe(DMFTHubbardSubstrate, dmft.dmft.DMFTHubbardBethe):
    """
    DMFT with Bethe Hubbard model coupled to a flat substrate
    """
    # Inheritance covers all the necessary steps
    pass

class DMFTHubbardSubstrateKagome(DMFTHubbardSubstrate, dmft.dmft.DMFTHubbardKagome):
    """
    DMFT with Kagome Hubbard model coupled to a flat substrate
    """
    # Inheritance covers all the necessary steps
    pass

class DMFTHubbardSubstrateImpurity(DMFTHubbardSubstrate):
    """
    DMFT with Anderson Impurity Model coupled to a flat substrate
    """
    def set_dos(self, ed, width=0.01):
        """
        Record non-interacting DOS for an atomic impurity at energy ed.

        width: DOS is rectangular, but this parameter doesn't actually matter
            because the summation makes the width cancel out.
        """
        super().set_dos([1/width], [ed], [width])
        self.ed = ed
    def record_metadata(self, A):
        super().record_metadata(A)
        SG = A['params']
        SG['impurity_energy'] = self.ed
    @classmethod
    @dmft.utils.archive_reader2
    def load(cls, archive):
        self = super().load(archive)
        params, code = cls._load_get_params_and_code(archive)
        try:
            self.ed = params['impurity_energy']
        except KeyError:
            pass


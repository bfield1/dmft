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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need to re-do the setting of the solver's structure
        # Extract beta
        if len(args) >= 1:
            beta = args[0]
        elif 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            raise TypeError("Required argument 'beta' not supplied")
        # Extract nl
        if len(args) >= 5:
            nl = args[4]
        elif 'nl' in kwargs:
            nl = kwargs['nl']
        else:
            nl = None
        # Re-define the solver to include space for a substrate
        if nl is None:
            self.S = Solver(beta=beta, gf_struct = [('up',[0,1]), ('down',[0,1])])
        else:
            self.S = Solver(beta=beta, gf_struct = [('up',[0,1]), ('down',[0,1])], nl=nl)
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
            save_metadata_per_loop=False, enforce_spins=True,
            enforce_sigma_hubbard_only=False):
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
            enforce_sigma_hubbard_only - Boolean. If True, only the Hubbard 
                part of the self-energy enters the self-consistency condition.
                Otherwise, the full self-energy enters it.
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
            self.S.Sigma_iw['up'][0,0] << self.mu
            self.S.Sigma_iw['down'][0,0] << self.mu
            # Record some metadata
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    self.record_metadata(A)
        # Generate the non-interacting substrate Green's function
        Gsub = self.S.G0_iw['up'][1,1].copy()
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
            # Enforce self-energy entering only into the interacting part
            # for the self-consistency condition.
            if enforce_sigma_hubbard_only:
                for name in ['up','down']:
                    sigma[name][1,0] << 0
                    sigma[name][0,1] << 0
                    sigma[name][1,1] << 0
            # Do the self-consistency condition
            G.zero()
            dG = G.copy()
            for name, _ in G:
                # dG is an in-between variable, the matrix before we invert it
                dG[name][0,1] << -self.V
                dG[name][1,0] << -self.V
                dG[name][1,1] << gf.inverse(Gsub)
                # Hilbert transform: an integral involving the DOS
                for i in range(len(self.rho)):
                    # Update the matrix element which has energy-dependence
                    dG[name][0,0] << gf.iOmega_n + self.mu - self.energy[i]
                    # Do the Reimann sum of the system.
                    G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(dG[name] - sigma[name])
            # Get the next impurity G0
            for name, g0 in self.S.G0_iw:
                g0 << gf.inverse(gf.inverse(G[name]) + sigma[name])
            # Run the solver
            self.S.solve(h_int=self.h_int, **self.solver_params)
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
                    SG['average_sign'] = self.S.average_sign
                    if save_metadata_per_loop:
                        self.record_metadata(SG)
            self.last_loop = i_loop + prior_loops
        if mpi.is_master_node():
            print("Finished DMFT loop.")

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

class DMFTHubbardSubstrateRotated(DMFTHubbardSubstrate):
    """
    DMFT with Hubbard model coupled to a flat substrate, evaluated in rotated
    basis.
    Specifically, normally we work in the basis of (HubbardLattice, Substrate).
    But here we rotate into the basis of (HubbardLattice+Substrate,
    HubbardLattice-Substrate)/sqrt(2).
    This makes the off-diagonal elements in the hybridisation function
    non-constant (and thus non-zero), which allows them to be evaluated
    properly by CTHYB's solver. (At least, that's the theory.)
    """
    @classmethod
    @dmft.utils.archive_reader2
    def load(cls, archive):
        self = super().load(archive)
        # Rotate into the proper basis
        rotmat = np.array([[1,1],[1,-1]])/np.sqrt(2)
        for name in ['up','down']:
            self.S.Sigma_iw[name].from_L_G_R(rotmat, self.S.Sigma[name], rotmat)
            try:
                self.S.G_iw[name].from_L_G_R(rotmat, self.S.G_iw[name], rotmat)
            except AttributeError:
                pass
            try:
                self.S.G_l[name].from_L_G_R(rotmat, self.S.G_l[name], rotmat)
            except AttributeError:
                pass
        return self
    @property
    def U(self):
        return super().U
    @U.setter
    def U(self, u):
        self._U = u
        # The interaction Hamiltonian is a bit more complicated in this basis
        nup = (op.c_dag('up',0)+op.c_dag('up',1)) * (op.c('up',0)+op.c('up',1)) / 2
        ndown = (op.c_dag('down',0)+op.c_dag('down',1)) * (op.c('down',0)+op.c('down',1)) / 2
        self.h_int = u * nup * ndown
    def loop(self, n_loops, archive=None, prior_loops=None,
            save_metadata_per_loop=False, enforce_spins=True):
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
        # Matrix for basis transformation
        rotmat = np.array([[1,1],[1,-1]])/np.sqrt(2)
        # If we aren't doing a continuation job,
        # set our initial guess for the self-energy
        if prior_loops == 0:
            for name in ['up','down']:
                self.S.Sigma_iw[name][0,0] << self.mu
                self.S.Sigma_iw[name].from_L_G_R(rotmat, self.S.Sigma_iw[name], rotmat)
            # Record some metadata
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    self.record_metadata(A)
        # Generate the non-interacting substrate Green's function
        Gsub = self.S.G0_iw['up'][1,1].copy()
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
            dG = G.copy()
            dGrot = dG.copy()
            for name, _ in G:
                # dG is an in-between variable, the matrix before we invert it
                dG[name][0,1] << -self.V
                dG[name][1,0] << -self.V
                dG[name][1,1] << gf.inverse(Gsub)
                # Hilbert transform: an integral involving the DOS
                for i in range(len(self.rho)):
                    # Update the matrix element which has energy-dependence
                    dG[name][0,0] << gf.iOmega_n + self.mu - self.energy[i]
                    # Rotate into the rotated basis
                    dGrot[name].from_L_G_R(rotmat, dG[name], rotmat)
                    # Do the Reimann sum of the system.
                    G[name] << G[name] + self.rho[i] * self.delta[i] * gf.inverse(dGrot[name] - sigma[name])
            # Get the next impurity G0
            for name, g0 in self.S.G0_iw:
                g0 << gf.inverse(gf.inverse(G[name]) + sigma[name])
            # Run the solver
            self.S.solve(h_int=self.h_int, **self.solver_params)
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
            if archive is not None and mpi.is_master_node():
                with HDFArchive(archive,'a') as A:
                    key = 'loop-{:03d}'.format(i_loop+prior_loops)
                    A.create_group(key)
                    SG = A[key]
                    # We need to rotate into the original basis to save
                    G.zero()
                    for name, g in G:
                        g.from_L_G_R(rotmat, self.S.G_iw[name], rotmat)
                    SG['G_iw'] = G
                    for name, g in G:
                        g.from_L_G_R(rotmat, self.S.Sigma_iw[name], rotmat)
                    SG['Sigma_iw'] = G
                    for name, g in G:
                        g.from_L_G_R(rotmat, self.S.G0_iw[name], rotmat)
                    SG['G0_iw'] = G
                    for name, g in G:
                        g.from_L_G_R(rotmat, self.S.G_tau[name], rotmat)
                    SG['G_tau'] = G
                    if 'measure_G_l' in self.solver_params and self.solver_params['measure_G_l']:
                        for name, g in G:
                            g.from_L_G_R(rotmat, self.S.G_l[name], rotmat)
                        SG['G_l'] = G
                    SG['average_sign'] = self.S.average_sign
                    if save_metadata_per_loop:
                        self.record_metadata(SG)
            self.last_loop = i_loop + prior_loops
        if mpi.is_master_node():
            print("Finished DMFT loop.")

class DMFTHubbardSubstrateBetheRotated(DMFTHubbardSubstrateRotated, dmft.dmft.DMFTHubbardBethe):
    """
    DMFT with Bethe Hubbard model coupled to a flat substrate
    """
    # Inheritance covers all the necessary steps
    pass

class DMFTHubbardSubstrateKagomeRotated(DMFTHubbardSubstrateRotated, dmft.dmft.DMFTHubbardKagome):
    """
    DMFT with Kagome Hubbard model coupled to a flat substrate
    """
    # Inheritance covers all the necessary steps
    pass

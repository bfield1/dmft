#   Copyright (C) 2022 Bernard Field, GNU GPL v3+
# A manual Python reverse engineering of triqs.gf.meshes, which was originally C++

import numpy as np

from .mesh_point import MeshValueGenerator, MeshPoint
from dmft.faketriqs.h5.formats import register_class

statistic_enum = ['Boson','Fermion']

class Mesh_Generic():
    """
    A linear mesh between two values
    """
    def __init__(self, x_min, x_max, n):
        self.x_min = x_min
        self.x_max = x_max
        self.n = n
        self.delta = (self.x_max - self.x_min)/(self.n-1)
    def index_to_linear(self, i):
        return i
    def __iter__(self):
        self._idx = 0
        return self
    def __next__(self):
        if self._idx < len(self):
            i = self._idx
            self._idx += 1
            return MeshPoint(linear_index = i, value = i*self.delta + self.x_min)
        else:
            raise StopIteration
    def __len__(self):
        return self.n
    def values(self):
        return MeshValueGenerator(self)
    def __eq__(self, other):
        return (self.x_min == other.x_min) and (self.x_max == other.x_max) and (self.n == other.n)
    def __ne__(self, other):
        return not self == other
    def __reduce_to_dict__(self):
        raise NotImplementedError

class MeshImTime(Mesh_Generic):
    def __init__(self, beta, S, n_max):
        self.beta = beta
        if S in statistic_enum:
            self.statistic = S
        else:
            raise ValueError("S must be 'Boson' or 'Fermion'.")
        super().__init__(x_min=0, x_max=beta, n=n_max)
    def copy(self):
        return MeshImTime(self.beta, self.statistic, self.n)
    @classmethod
    def __factory_from_dict__(cls, name, D):
        """ this handles reading from h5 """
        if 'domain' in D:
            # Backwards compatibility with TRIQS 3.0
            beta = D['domain']['beta']
            S = D['domain']['statistic']
        else:
            # In TRIQS 3.1, they removed 'domain'.
            beta  = D['beta']
            S = D['statistic']
        if S == 'F':
            S = 'Fermion'
        elif S == 'B':
            S = 'Boson'
        n = D['size']
        return cls(beta, S, n)

register_class(MeshImTime)

class MeshReFreq(Mesh_Generic):
    def __init__(self, omega_min, omega_max, n_max):
        super().__init__(x_min=omega_min, x_max=omega_max, n=n_max)
    def copy(self):
        return MeshReFreq(self.x_min, self.x_max, self.n)
    @classmethod
    def __factory_from_dict__(cls, name, D):
        """This handles reading from h5"""
        return cls(D["min"], D["max"], D["size"])

register_class(MeshReFreq)

class MeshReTime(Mesh_Generic):
    def __init__(self, t_min, t_max, n_max):
        super().__init__(x_min=t_min, x_max=t_max, n=n_max)
    def copy(self):
        return MeshReTime(self.x_min, self.x_max, self.n)
    @classmethod
    def __factory_from_dict__(cls, name, D):
        """This handles reading from h5"""
        # For MeshReTime, this is currently a guess from MeshReFreq.
        return cls(D["min"], D["max"], D["size"])

register_class(MeshReTime)

class MeshImFreq(Mesh_Generic):
    def __init__(self, beta, S, n_max=1025):
        self.beta = beta
        if S in statistic_enum:
            self.statistic = S
        else:
            raise ValueError("S must be 'Boson' or 'Fermion'.")
        # Matsubara frequencies depend on whether we're bosons or fermions
        # Boson = 0, Fermion = 1
        n = 2*n_max - 1 + statistic_enum.index(S)
        delta = 2*np.pi/beta
        xmax = complex(0,delta*(n_max-1) + statistic_enum.index(S)*delta/2)
        super().__init__(x_min = -xmax, x_max = xmax, n=n)
    def copy(self):
        n_max = self.last_index() + 1
        return MeshImFreq(self.beta, self.statistic, n_max)
    def __call__(self, n):
        S = statistic_enum.index(self.statistic)
        return complex(0, (2*n + S) * np.pi / self.beta)
    def last_index(self):
        return int(self.n/2) - statistic_enum.index(self.statistic)
    def first_index(self):
        return -int(self.n/2)
    def index_to_linear(self, i):
        return i - self.first_index()
    def positive_only(self):
        return False
    @classmethod
    def __factory_from_dict__(cls, name, D):
        """ this handles reading from h5 """
        if 'domain' in D:
            # Backwards compatibility with TRIQS 3.0
            beta = D['domain']['beta']
            S = D['domain']['statistic']
        else:
            # In TRIQS 3.1, they removed 'domain'.
            beta  = D['beta']
            S = D['statistic']
        positive_freq_only = bool(D['positive_freq_only'])
        n = D['size']
        if positive_freq_only:
            raise NotImplementedError("faketriqs doesn't currently implement positive_freq_only")
        if S == 'F':
            S = 'Fermion'
        elif S == 'B':
            S = 'Boson'
        n_max = int(n/2) - statistic_enum.index(S) + 1
        return cls(beta, S, n_max)

register_class(MeshImFreq)

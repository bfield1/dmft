import numpy as np

from .mesh_point import MeshValueGenerator, MeshPoint


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

class MeshReFreq(Mesh_Generic):
    def __init__(self, omega_min, omega_max, n_max):
        super().__init__(x_min=omega_min, x_max=omega_max, n=n_max)
    def copy(self):
        return MeshReFreq(self.x_min, self.x_max, self.n)

class MeshReTime(Mesh_Generic):
    def __init__(self, t_min, t_max, n_max):
        super().__init__(x_min=t_min, x_max=t_max, n=n_max)
    def copy(self):
        return MeshReTime(self.x_min, self.x_max, self.n)

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


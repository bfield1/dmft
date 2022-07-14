# TRIQS application maxent
# Copyright (C) 2018 Gernot J. Kraberger
# Copyright (C) 2018 Simons Foundation
# Authors: Gernot J. Kraberger and Manuel Zingl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from .omega_meshes import HyperbolicOmegaMesh
import numpy as np
import copy


class TauMaxEnt(object):
    r""" Perform MaxEnt with a :math:`G(\tau)` kernel.

    The methods and properties of :py:class:`.MaxEntLoop` are, in general,
    shadowed by ``TauMaxEnt``, i.e., they can be used in a ``TauMaxEnt``
    object as well.

    Parameters
    ----------
    cov_threshold : float
        when setting a covariance using :py:meth:`.TauMaxEnt.set_cov`, this threshold
        is used to ignore small eigenvalues
    **kwargs :
        are passed on to :py:class:`.MaxEntLoop`
    """

    # this is needed to make the getattr/setattr magic work
    maxent_loop = None

    def __init__(self, cov_threshold=1.e-14, alpha_mesh=None, scale_alpha='Ndata', **kwargs):

        omega = HyperbolicOmegaMesh()
        # N.B.: can only set omega after having initialized a kernel
        self.omega = omega
        self.cov_threshold = cov_threshold
        self.alpha_mesh = alpha_mesh
        if self.alpha_mesh is None:
            self.alpha_mesh = LogAlphaMesh()
        self.scale_alpha = scale_alpha

    def set_G_tau_data(self, tau, G_tau):
        r""" Set :math:`G(\tau)` from array.

        Parameters
        ==========
        tau : array
            tau-grid
        G_tau : array
            The data for the analytic continuation.
        """

        assert len(tau) == len(G_tau), \
            "tau and G_tau don't have the same dimension"
        self.tau = tau
        self.G = G_tau

    def set_G_tau_file(self, filename, tau_col=0, G_col=1, err_col=None):
        r""" Set :math:`G(\tau)` from data file.

        Parameters
        ==========
        filename : str
            the name of the file to load.
            The first column (see ``tau_col``) is the :math:`\tau`-grid,
            the second column (see ``G_col``) is the :math:`G(\tau)`
            data.
        tau_col : int
            the 0-based column number of the :math:`\tau`-grid
        G_col : int
            the 0-based column number of the :math:`G(\tau)`-data
        err_col : int
            the 0-based column number of the error-data or None if the
            error is not supplied via a file
        """

        dat = np.loadtxt(filename)
        self.tau = dat[:, tau_col]
        self.G = dat[:, G_col]
        if err_col is not None:
            self.err = dat[:, err_col]

    def set_error(self, error):
        r""" Set error """
        self.err = error
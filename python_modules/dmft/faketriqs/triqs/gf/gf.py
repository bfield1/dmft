# Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2018-2020 Simons Foundation
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
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Michel Ferrero, Manuel, Olivier Parcollet, Hugo U. R. Strand, Nils Wentzell
# This version with edits (mostly stuff removed) by Bernard Field

import itertools, warnings, numbers
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
import numpy as np
from . import mesh_product
from .mesh_product import MeshProduct
from dmft.faketriqs.triqs.plot.protocol import clip_array
from . import meshes
from . import plot 
from . import gf_fnt
from .gf_fnt import GfIndices
from .mesh_point import MeshPoint

# list of all the meshes
all_meshes = (MeshProduct,) + tuple(c for c in list(meshes.__dict__.values()) if isinstance(c, type) and c.__name__.startswith('Mesh'))

# For IO later
def call_factory_from_dict(cl,name, dic):
    """Given a class cl and a dict dic, it calls cl.__factory_from_dict__(dic)"""
    return cl.__factory_from_dict__(name, dic)

# a metaclass that adds all functions of gf_fnt as methods 
# the C++ will take care of the dispatch
def add_method_helper(a,cls): 
    def _(self, *args, **kw):
       return a(self, *args, **kw)
    _.__doc__ =  a.__doc__
    #_.__doc__ = 50*'-' + '\n' + a.__doc__
    _.__name__ = a.__name__
    return _

class AddMethod(type): 
    def __init__(cls, name, bases, dct):
        super(AddMethod, cls).__init__(name, bases, dct)
        for a in [f for f in list(gf_fnt.__dict__.values()) if callable(f)]:
            if not hasattr(cls, a.__name__):
                setattr(cls, a.__name__, add_method_helper(a,cls))

class Idx:
    def __init__(self, *x):
        self.idx = x[0] if len(x)==1 else x

class Gf(metaclass=AddMethod):
    r""" TRIQS Greens function container class

    Parameters
    ----------

    mesh: Types defined in triqs.gf beginning with 'Mesh'
          The mesh on which the Green function is defined.

    data: numpy.array, optional
          The data of the Greens function.
          Must be of dimension ``mesh.rank + target_rank``.

    target_shape: list of int, optional
                  Shape of the target space.

    is_real: bool
             Is the Greens function real valued?
             If true, and target_shape is set, the data will be real.
             Mutually exclusive with argument ``data``.

    name: str 
          The name of the Greens function for plotting.

    Notes
    -----

    One of ``target_shape`` or ``data`` must be set, and the other must be `None`. If passing ``data``
    and ``indices`` the ``data.shape`` needs to be compatible with the shape of ``indices``.

    """
    
    _hdf5_data_scheme_ = 'Gf'
    
    def __init__(self, **kw): # enforce keyword only policy 
        
        #print "Gf construct args", kw

        def delegate(self, mesh, data=None, target_shape=None, indices = None, name = '', is_real = False):
            """
            target_shape and data  : must provide exactly one of them
            """
            self.name = name

            # input check
            assert (target_shape is None) or (data is None), "data and target_shape : one must be None"
            assert (data is None) or (is_real is False), "is_real can not be True if data is not None"
            if target_shape : 
                for i in target_shape : 
                    assert i>0, "Target shape elements must be >0"
     
            # mesh
            assert isinstance(mesh, all_meshes), "Mesh is unknown. Possible type of meshes are %s" % ', '.join([m.__name__ for m in all_meshes])
            self._mesh = mesh

            # indices
            # if indices is not a list of list, but a list, then the target_rank is assumed to be 2 !
            # backward compatibility only, it is not very logical (what about vector_valued gf ???)
            assert isinstance(indices, (type(None), list, GfIndices)), "Type of indices incorrect : should be None, Gfindices, list of str, or list of list of str"
            # I added the not isinstance(indices, GfIndices) because I made
            # GfIndices a subclass of list in my fake implementation.
            if isinstance(indices, list) and not isinstance(indices, GfIndices):
                if not isinstance(indices[0], list): indices = [indices, indices]
                # indices : transform indices into string
                indices = [ [str(x) for x in v] for v in indices]
                indices = GfIndices(indices)
            self._indices = indices # now indices are None or Gfindices 

            # data
            if data is None:
                # if no data, we get the target_shape. If necessary, we find it from of the list of indices
                if target_shape is None : 
                    assert indices, "Without data, target_shape, I need the indices to compute the shape !"
                    target_shape = [ len(x) for x in indices.data]
                # we now allocate the data
                l = mesh.size_of_components() if isinstance(mesh, MeshProduct) else [len(mesh)]
                data = np.zeros(list(l) + list(target_shape), dtype = np.float64 if is_real else np.complex128)
            else:
                l = tuple(mesh.size_of_components()) if isinstance(mesh, MeshProduct) else (len(mesh),)
                assert l == data.shape[0:len(l)], "Mismatch between data shape %s and sizes of mesh(es) %s\n " % (data.shape, l)

            # Now we have the data at correct size. Set up a few short cuts
            self._data = data
            len_data_shape = len(self._data.shape) 
            self._target_rank = len_data_shape - (self._mesh.rank if isinstance(mesh, MeshProduct) else 1)  
            self._rank = len_data_shape - self._target_rank 
            assert self._rank >= 0

            # target_shape. Ensure it is correct in any case.
            assert target_shape is None or tuple(target_shape) == self._data.shape[self._rank:] # Debug only
            self._target_shape = self._data.shape[self._rank:]

            # If no indices was given, build the default ones
            if self._indices is None: 
                self._indices = GfIndices([list(str(i) for i in range(n)) for n in self._target_shape])

            # Check that indices  have the right size
            if self._indices is not None: 
                d,i =  self._data.shape[self._rank:], tuple(len(x) for x in self._indices.data)
                assert (d == i), "Indices are of incorrect size. Data size is %s while indices size is %s"%(d,i)
            # Now indices are set, and are always a GfIndices object, with the
            # correct size
            
            # check all invariants. Debug.
            self.__check_invariants()
            

        delegate(self, **kw)
    
    def __check_invariants(self):
        """Check various invariant. Mainly for debug"""
        # rank
        assert self.rank == self._mesh.rank if isinstance (self._mesh, MeshProduct) else 1
        # The mesh size must correspond to the size of the data
        assert self._data.shape[:self._rank] == tuple(len(m) for m in self._mesh.components) if isinstance (self._mesh, MeshProduct) else (len(self._mesh),)
    
    def density(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def rank(self):
        r"""int : The mesh rank (number of meshes)."""
        return self._rank

    @property
    def target_rank(self): 
        """int : The rank of the target space."""
        return self._target_rank

    @property
    def target_shape(self): 
        """(int, ...) : The shape of the target space."""
        return self._target_shape

    @property
    def mesh(self):
        """gf_mesh : The mesh of the Greens function."""
        return self._mesh

    @property
    def data(self):
        """ndarray : Raw data of the Greens function.

           Storage convention is ``self.data[x,y,z, ..., n0,n1,n2]``
           where ``x,y,z`` correspond to the mesh variables (the mesh) and 
           ``n0, n1, n2`` to the ``target_space``.
        """
        return self._data

    @property
    def indices(self):
        """GfIndices : The index object of the target space."""
        return self._indices

    def copy(self) : 
        """Deep copy of the Greens function.

        Returns
        -------
        G : Gf
            Copy of self.
        """
        return Gf (mesh = self._mesh.copy(), 
                   data = self._data.copy(), 
                   indices = self._indices.copy(), 
                   name = self.name)

    def copy_from(self, another):
        """Copy the data of another Greens function into self."""
        self._mesh.copy_from(another.mesh)
        assert self._data.shape == another._data.shape, "Shapes are incompatible: " + str(self._data.shape) + " vs " + str(another._data.shape)
        self._data[:] = another._data[:]
        self._indices = another._indices.copy()
        self.__check_invariants()
    
    def __repr__(self):
        return "Greens Function %s with mesh %s and target_shape %s: \n"%(self.name, self.mesh, self.target_shape)
 
    def __str__ (self): 
        return self.__repr__()
	
    #--------------  Bracket operator []  -------------------------
    
    _full_slice = slice(None, None, None) 

    def __getitem__(self, key):

        # First case : g[:] = RHS ... will be g << RHS
        if key == self._full_slice:
            return self

        # Only one argument. Must be a mesh point, idx or slicing rank1 target space
        if not isinstance(key, tuple):
            if isinstance(key, (MeshPoint, Idx)):
                return self.data[key.linear_index if isinstance(key, MeshPoint) else self._mesh.index_to_linear(key.idx)]
            else: key = (key,)

        # If all arguments are MeshPoint, we are slicing the mesh or evaluating
        if all(isinstance(x, (MeshPoint, Idx)) for x in key):
            assert len(key) == self.rank, "wrong number of arguments in [ ]. Expected %s, got %s"%(self.rank, len(key))
            return self.data[tuple(x.linear_index if isinstance(x, MeshPoint) else m.index_to_linear(x.idx) for x,m in zip(key,self._mesh._mlist))]

        # If any argument is a MeshPoint, we are slicing the mesh or evaluating
        elif any(isinstance(x, (MeshPoint, Idx)) for x in key):
            assert len(key) == self.rank, "wrong number of arguments in [[ ]]. Expected %s, got %s"%(self.rank, len(key))
            assert all(isinstance(x, (MeshPoint, Idx, slice)) for x in key), "Invalid accessor of Greens function, please combine only MeshPoints, Idx and slice"
            assert self.rank > 1, "Internal error : impossible case" # here all == any for one argument
            mlist = self._mesh._mlist 
            for x in key:
                if isinstance(x, slice) and x != self._full_slice: raise NotImplementedError("Partial slice of the mesh not implemented")
            # slice the data
            k = tuple(x.linear_index if isinstance(x, MeshPoint) else m.index_to_linear(x.idx) if isinstance(x, Idx) else x for x,m in zip(key,mlist)) + self._target_rank * (slice(0, None),)
            dat = self._data[k]
            # list of the remaining lists
            mlist = [m for i,m in filter(lambda tup_im : not isinstance(tup_im[0], (MeshPoint, Idx)), zip(key, mlist))]
            assert len(mlist) > 0, "Internal error" 
            mesh = MeshProduct(*mlist) if len(mlist)>1 else mlist[0]
            sing = None 
            r = Gf(mesh = mesh, data = dat)
            r.__check_invariants()
            return r

        # In all other cases, we are slicing the target space
        else : 
            assert self.target_rank == len(key), "wrong number of arguments. Expected %s, got %s"%(self.target_rank, len(key))

            # Assume empty indices (scalar_valued)
            ind = GfIndices([])

            # String access: transform the key into a list integers
            if all(isinstance(x, str) for x in key):
                warnings.warn("The use of string indices is deprecated", DeprecationWarning)
                assert self._indices, "Got string indices, but I have no indices to convert them !"
                key_tpl = tuple(self._indices.convert_index(s,i) for i,s in enumerate(key)) # convert returns a slice of len 1

            # Slicing with ranges -> Adjust indices
            elif all(isinstance(x, slice) for x in key): 
                key_tpl = tuple(key)
                ind = GfIndices([ v[k]  for k,v in zip(key_tpl, self._indices.data)])

            # Integer access
            elif all(isinstance(x, int) for x in key):
                key_tpl = tuple(key)

            # Invalid Access
            else:
                raise NotImplementedError("Partial slice of the target space not implemented")

            dat = self._data[ self._rank*(slice(0,None),) + key_tpl ]
            r = Gf(mesh = self._mesh, data = dat, indices = ind)

            r.__check_invariants()
            return r

    def __setitem__(self, key, val):

        # Only one argument and not a slice. Must be a mesh point, Idx
        if isinstance(key, (MeshPoint, Idx)):
            self.data[key.linear_index if isinstance(key, MeshPoint) else self._mesh.index_to_linear(key.idx)] = val

        # If all arguments are MeshPoint, we are slicing the mesh or evaluating
        elif isinstance(key, tuple) and all(isinstance(x, (MeshPoint, Idx)) for x in key):
            assert len(key) == self.rank, "wrong number of arguments in [ ]. Expected %s, got %s"%(self.rank, len(key))
            self.data[tuple(x.linear_index if isinstance(x, MeshPoint) else m.index_to_linear(x.idx) for x,m in zip(key,self._mesh._mlist))] = val

        else:
            self[key] << val
    
    # -------------- Various operations -------------------------------------
    
    @property
    def real(self): 
        """Gf : A Greens function with a view of the real part."""
        return Gf(mesh = self._mesh, data = self._data.real, name = ("Re " + self.name) if self.name else '') 

    @property
    def imag(self): 
        """Gf : A Greens function with a view of the imaginary part."""
        return Gf(mesh = self._mesh, data = self._data.imag, name = ("Im " + self.name) if self.name else '') 
    
    # -------------- call -------------------------------------
    
    def __call__(self, *args) : 
        raise NotImplementedError
    
    #----------------------------- other operations -----------------------------------
    
    def total_density(self, *args, **kwargs):
        raise NotImplementedError
    
	#-----------------------------  IO  -----------------------------------
    
    def __reduce__(self):
        return call_factory_from_dict, (Gf, self.name, self.__reduce_to_dict__())

    def __reduce_to_dict__(self):
        d = {'mesh' : self._mesh, 'data' : self._data}
        if self.indices : d['indices'] = self.indices 
        return d

    _hdf5_format_ = 'Gf'

    @classmethod
    def __factory_from_dict__(cls, name, d):
        # Backward compatibility layer
        # Drop singularity from the element and ignore it
        d.pop('singularity', None)
        #
        r = cls(name = name, **d)
        # Backward compatibility layer
        # In the case of an ImFreq function, old archives did store only the >0
        # frequencies, we need to duplicate it for negative freq.
        # Same code as in the C++ h5_read for gf.
        need_unfold = isinstance(r.mesh, meshes.MeshImFreq) and r.mesh.positive_only() 
        if need_unfold:
            raise NotImplementedError
        #return r if not need_unfold else wrapped_aux._make_gf_from_real_gf(r)
        return r
	
	#-----------------------------plot protocol -----------------------------------

    def _plot_(self, opt_dict):
        """ Implement the plot protocol"""
        return plot.dispatcher(self)(self, opt_dict)

    def x_data_view(self, x_window=None, flatten_y=False):
        """Helper method for getting a view of the data.

        Parameters
        ----------

        x_window : optional
            The window of x variable (omega/omega_n/t/tau) for which data is requested.
        flatten_y: bool, optional
            If the Greens function is of size (1, 1) flatten the array as a 1d array.

        Returns
        -------

        (X, data) : tuple
            X is a 1d numpy array of the x variable inside the window requested.
            data is a 3d numpy array of dim (:,:, len(X)), the corresponding slice of data.
            If flatten_y is True and dim is (1, 1, *) it returns a 1d numpy array.
        """
        
        X = [x.imag for x in self.mesh] if isinstance(self.mesh, meshes.MeshImFreq) \
            else [x for x in self.mesh]
        
        X, data = np.array(X), self.data
        if x_window:
            # the slice due to clip option x_window
            sl = clip_array(X, *x_window) if x_window else slice(len(X))
            X, data = X[sl],  data[sl, :, :]
        if flatten_y and data.shape[1:3] == (1, 1):
            data = data[:, 0, 0]
        return X, data

#---------------------------------------------------------

from dmft.faketriqs.h5.formats import register_class, register_backward_compatibility_method
register_class (Gf)

# A backward compatility function
def bckwd(hdf_scheme):
    # we know scheme is of the form GfM1_x_M2_s/tv3
    m, t= hdf_scheme[2:], '' # get rid of Gf
    for suffix in ['_s', 'Tv3', 'Tv4'] : 
        if m.endswith(suffix) :
            m, t = m[:-len(suffix)], suffix
            break
    return { 'mesh': 'Mesh'+m, 'indices': 'GfIndices'}

register_backward_compatibility_method("Gf", "Gf", bckwd)

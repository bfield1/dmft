import itertools, warnings, numbers
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
import numpy as np
from . import mesh_product
from .mesh_product import MeshProduct
from triqs.plot.protocol import clip_array
from . import meshes
from . import plot 

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
            if isinstance(indices, list):
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
           
            # NB : at this stage, enough checks should have been made in Python in order for the C++ view 
            # to be constructed without any INTERNAL exceptions.
            # Set up the C proxy for call operator for speed. The name has to
            # agree with the wrapped_aux module, it is of only internal use
            s = '_x_'.join( m.__class__.__name__[4:] for m in self.mesh._mlist) if isinstance(mesh, MeshProduct) else self._mesh.__class__.__name__[4:]
            proxyname = 'CallProxy%s_%s%s'%(s, self.target_rank,'_R' if data.dtype == np.float64 else '') 
            try:
                self._c_proxy = all_call_proxies.get(proxyname, CallProxyNone)(self)
            except:
                self._c_proxy = None

            # check all invariants. Debug.
            self.__check_invariants()

        delegate(self, **kw)
	
	def __repr__(self):
        return "Greens Function %s with mesh %s and target_shape %s: \n"%(self.name, self.mesh, self.target_shape)
 
    def __str__ (self): 
        return self.__repr__()
	
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
        return r if not need_unfold else wrapped_aux._make_gf_from_real_gf(r)
	
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
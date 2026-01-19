"""
A h5py implementation of triqs/h5 which does the bare minimum necessary for
reading data recorded by dmft and maxent.
"""

# Original version Copyright (c) 2019-2020 Simons Foundation
# This modified version Copyright (c) 2022 Bernard Field
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified such that calls to read h5 archives use h5py, and disables writing
# to file.


from importlib import import_module

import numpy as np

import h5py

from .formats import register_class, register_backward_compatibility_method, get_format_info

# -------------------------------------------
#
#  Various wrappers for basic python types.
#
# --------------------------------------------
class List:
    def __init__(self,ob) :
        self.ob = ob
    def __reduce_to_dict__(self) :
        return {str(n):v for n,v in enumerate(self.ob)}
    @classmethod
    def __factory_from_dict__(cls, name, D) :
        return [x for n,x in sorted([(int(n), x) for n,x in list(D.items())])]

class Tuple:
    def __init__(self,ob) :
        self.ob = ob
    def __reduce_to_dict__(self) :
        return {str(n):v for n,v in enumerate(self.ob)}
    @classmethod
    def __factory_from_dict__(cls, name, D) :
        return tuple(x for n,x in sorted([(int(n), x) for n,x in list(D.items())]))

class Dict:
    def __init__(self,ob) :
        self.ob = ob
    def __reduce_to_dict__(self) :
        return {str(n):v for n,v in list(self.ob.items())}
    @classmethod
    def __factory_from_dict__(cls, name, D) :
        return {n:x for n,x in list(D.items())}

register_class(List)
register_backward_compatibility_method('PythonListWrap', 'List')

register_class(Tuple)
register_backward_compatibility_method('PythonTupleWrap', 'Tuple')

register_class(Dict)
register_backward_compatibility_method('PythonDictWrap', 'Dict')

class HDFArchiveGroup():
    """
    Expected behaviour:
        archive[group] returns HDFArchiveGroup
        unless it has attributes saying it should be another datatype
        or if archive[X] is a dataset rather than a group, in which case it
        should return an appropriate datatype.
    """
    def __init__(self, parent, subpath):
        self._group = parent._group[subpath] if subpath else parent._group
        if not isinstance(self._group, h5py.Group):
            raise TypeError("Subpath %s does not point to a group"%subpath)
        self.is_top_level = False
    def keys(self):
        return list(self._group.keys())
    def __contains__(self, key):
        return key in list(self.keys())
    def values(self):
        """Generator returning values in the group"""
        def res() :
            for name in list(self.keys()) :
                yield self[name]
        return res()
    def items(self):
        """Generator returning couples (key, values) in the group"""
        def res() :
            for name in list(self.keys()):
                yield name, self[name]
        return res()
    def __iter__(self):
        """Returns the keys, like a dictionary"""
        def res() :
            for name in list(self.keys()) :
                yield name
        return res()
    def __len__(self) :
        """Returns the length of the keys list """
        return  len(list(self.keys()))
    def is_group(self,p):
        """Is p a subgroup?"""
        assert len(p)>0 and p[0]!='/'
        return p in self.keys() and isinstance(self._group[p],h5py.Group)
    def is_data(self,p):
        """Is p a dataset?"""
        assert len(p)>0 and p[0]!='/'
        return p in self.keys() and isinstance(self._group[p],h5py.Dataset)
    def get_raw(self,key):
        """Similar to __getitem__ but it does NOT reconstruct the python object,
        it presents it as a subgroup"""
        return self.__getitem1__(key,False)
    def __getitem__(self,key):
        """Return the object key, possibly reconstructed as a python object if
        it has been properly set up"""
        # If the key contains /, grabs the subgroups
        if '/' in key:
            a,l = self, key.split('/')
            for s in l[:-1]: a = a.get_raw(s)
            return a[l[-1]]
        return self.__getitem1__(key,True)
    #-------------------------
    def __getitem1__(self, key, reconstruct_python_object, hdf5_format = None) :

        if key not in self :
            raise KeyError("Key %s does not exist."%key)

        if self.is_group(key) :
            SubGroup = HDFArchiveGroup(self,key) # View of the subgroup
            bare_return = lambda: SubGroup
        elif self.is_data(key):
            dset = self._group[key]
            # If it is a string, return as a string
            if dset.dtype == 'O' or dset.dtype == 'S':
                bare_return = lambda: dset.asstr()[()]
            # Otherwise, return as a numeric type.
            else:
                if "__complex__" in dset.attrs and dset.attrs["__complex__"] == "1":
                    # We have a special case: the last dimension of this dataset
                    # is 2, and encodes complex values
                    if dset.shape[-1] != 2:
                        bare_return = lambda: dset[()]
                    else:
                        bare_return = lambda: np.array(dset[...,0] + 1j*dset[...,1])
                else:
                    bare_return = lambda: dset[()]
        else :
            raise KeyError("Key %s is of unknown type !!"%Key)

        if not reconstruct_python_object : return bare_return()

        # try to find the format
        if hdf5_format is None:
            attrs = self._group[key].attrs
            if "Format" in attrs:
                hdf5_format = attrs["Format"]
            else:
                hdf5_format = ""
            if hdf5_format == "":
                return bare_return()

        try :
            fmt_info = get_format_info(hdf5_format)
        except KeyError:
            print("Warning : The hdf5 format %s is not recognized. Returning as a group. Hint : did you forgot to import this python class ?"%hdf5_format)
            return bare_return()

        r_class_name  = fmt_info.classname
        r_module_name = fmt_info.modulename
        r_readfun = fmt_info.read_fun
        if not (r_class_name and r_module_name) : return bare_return()
        try:
            r_class = getattr(import_module(r_module_name),r_class_name)
        except KeyError:
            raise RuntimeError("I cannot find the class %s to reconstruct the object !"%r_class_name)
        if r_readfun:
            return r_readfun(self._group, key)
        if hasattr(r_class,"__factory_from_dict__"):
            assert self.is_group(key), "__factory_from_dict__ requires a subgroup"
            reconstruct = lambda k: SubGroup.__getitem1__(k, reconstruct_python_object, fmt_info.backward_compat.get(k, None))
            values = {k: reconstruct(k) for k in SubGroup}
            return r_class.__factory_from_dict__(key, values)

        raise ValueError("Impossible to reread the class %s for group %s and key %s"%(r_class_name,self, key))
    #------------------------
    def __str__(self):
        def pr(name) :
            if self.is_group(name) :
                return "%s : subgroup"%name
            elif self.is_data(name) : # can be an array of a number
                return "%s : data "%name
            else :
                raise ValueError("oopps %s"%name)

        s= "HDFArchive%s with the following content:\n"%(" (partial view)" if self.is_top_level else '')
        s+='\n'.join([ '  '+ pr(n) for n in list(self.keys()) ])
        return s
    def __repr__(self) :
        return self.__str__()
    # These two methods are necessary for "with"
    def __enter__(self): return self
    def __exit__(self, type, value, traceback): pass


class HDFArchive(HDFArchiveGroup):
    """
    Expected behaviour:
        HDFArchive(archive, 'r') loads archive in read-only form into the object
    """
    def __init__(self, descriptor, open_flag):
        if open_flag != 'r':
            raise ValueError("Fake HDFArchive can only open files in read-only mode.")
        self._group = h5py.File(descriptor, 'r')
        self.is_top_level = True
    def __del__(self):
        # We must ensure the root group is closed before closing the file
        if hasattr(self, '_group'):
            del self._group
    def __exit__(self, type, value, traceback):
        del self._group

class HDFArchiveInert:
    """
    A fake class for the node in MPI. It does nothing, but
    permits to write simply :
       a= mpi.bcast(H['a']) # run on all nodes
    -[] : __getitem__ returns self so that H['a']['b'] is ok...
    - setitem : does nothing.
    """
    def HDFArchive_Inert(self):
        pass
    def __getitem__(self,x)   : return self
    def __setitem__(self,k,v) : pass

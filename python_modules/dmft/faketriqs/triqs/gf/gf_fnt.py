#   Copyright (C) 2022 Bernard Field, GNU GPL v3+

class GfIndices(list):
    def __reduce_to_dict__(self):
        raise NotImplementedError
    @classmethod
    def __factory_from_dict__(cls, name, D):
        return [[D['left']], [D['right']]]
    @property
    def data(self):
        return list(self)

from dmft.faketriqs.h5.formats import register_class
register_class(GfIndices)

def density(self, *args, **kwargs):
    raise NotImplementedError("density requires some sort of integration which I'm not putting in the fake version")

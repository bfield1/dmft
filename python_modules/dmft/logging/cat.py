"""
For reading and printing logs saved to HDF5 archives

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""
import argparse

import h5py

def read_string(archive, name, encoding='utf-8'):
    if isinstance(archive, str):
        with h5py.File(archive, 'r') as A:
            return read_string(A, name)
    try:
        item = archive[name].asstr(encoding=encoding)[()]
        # I want some more human-readable error messages
    except TypeError as E:
        if str(E) == "dset.asstr() can only be used on datasets with an HDF5 string datatype":
            raise TypeError("Chosen dataset is not a string dataset") from None
        else:
            raise
    except AttributeError as E:
        if str(E) == "'Group' object has no attribute 'asstr'":
            raise TypeError("Groups cannot be read as strings") from None
        else:
            raise
    # There is also a KeyError if name doesn't exist, but its message is satisfactory
    # We have several scenarios.
    # First, item is a scalar dataset
    if isinstance(item, str):
        return item
    # Second, item is a list of lines of text
    # They may or may not be newline terminated
    if item[0][-1] == '\n':
        return ''.join(item)
    return '\n'.join(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prints text in a HDF5 archive")
    parser.add_argument("archive", help="Archive to read from.")
    parser.add_argument("name", help="Name of text dataset to print.")
    args = parser.parse_args()

    print(read_string(args.archive, args.name))

"""
Renames a HDF5 object
"""
import argparse
import sys

import h5py

def move(archive, source, destination, force=False):
    """
    Inside HDF5 archive, moves source to destination

    By default does not overwrite destination

    Inputs:
        archive - h5py.File, or str pointing to HDF5 archive
        source - str, path in archive of original file
        destination - str, destination path
        force - Boolean, if True overwrites destination if it exists.
    """
    if isinstance(archive, str):
        with h5py.File(archive, 'a') as A:
            move(A, source, destination, force)
    else:
        if destination in archive:
            if force:
                # First we'll test that the source actually exists
                archive[source]
                # If that passes without error, then we can delete the destination
                del archive[destination]
            else:
                raise KeyError("Destination name "+destination+" already exists. Use force to overwrite.")
        archive[destination] = archive[source]
        del archive[source]
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename a HDF5 entry")
    parser.add_argument("archive", help="HDF5 archive")
    parser.add_argument("source", help="Name of source object")
    parser.add_argument("destination", help="Destination name")
    parser.add_argument("-f","--force", action="store_true", help="Overwrite destination if it exists")
    args = parser.parse_args()

    move(args.archive, args.source, args.destination, args.force)


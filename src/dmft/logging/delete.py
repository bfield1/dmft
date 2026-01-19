"""
For deleting entries in a HDF5 archive

NOTE: HDF5 doesn't support true deletion. Only deleting the names. It doesn't
free up file space.

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""
import argparse
import sys

import h5py


def delete(archive, name):
    """Deletes name from archive"""
    if isinstance(archive, str):
        with h5py.File(archive, 'r+') as A:
            delete(A, name)
    else:
        del archive[name]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete entries from a HDF5 archive")
    parser.add_argument("archive", help="HDF5 archive")
    parser.add_argument("names", nargs="+", help="Items to delete from the archive")
    args = parser.parse_args()

    # Open HDF5 archive now for efficiency
    try:
        with h5py.File(args.archive, 'r+') as A:
            for name in args.names:
                try:
                    delete(A, name)
                except KeyError:
                    print("Error:", name, "not in", args.archive, file=sys.stderr)
    except FileNotFoundError:
        print("Error: file", args.archive, "does not exist.", file=sys.stderr)
    except OSError:
        print("Error: unable to open file", args.archive, file=sys.stderr)

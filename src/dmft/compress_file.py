"""
A utility to compress a HDF5 archive by cutting out unnecessary data

    Copyright (C) 2022 Bernard Field, GNU GPL v3+

Currently, this is just G_rec in the maxent results, because it's big
And, optionally, old loops
"""

import argparse
import sys
import os
import tempfile
import shutil

import h5py


def compress_archive(archive, keep_loops=None, verbose=False):
    """
    Deletes some of the unnecessary data in the archives to free up space
    
    Inputs:
        archive - string, name of hdf5 archive created by dmft
        keep_loops - optional integer
        verbose - Boolean, whether to print debug messages
    """
    with h5py.File(archive, 'r+') as A:
        delete_something = False
        # Delete the groups we want
        if 'maxent' in A:
            if verbose: print("maxent is in archive", archive)
            for group in A['maxent']:
                if verbose: print("maxent/"+group, "is in archive", archive)
                name = 'maxent/'+group+'/results/G_rec'
                if name in A:
                    if verbose: print(name, "is in archive", archive+". Deleting.")
                    del A[name]
                    delete_something = True
        # Now go over the loops
        if keep_loops is not None:
            loop_list = sorted([k for k in A if k[0:5] == 'loop-'])
            # Trim out unwanted loops:
            loop_list = loop_list[-keep_loops:]
            loop_list.insert(0, 'loop-000')
            if verbose: print("Keeping loops", loop_list)
            for group in A:
                if ('loop-' in group) and (group not in loop_list):
                    del A[group]
                    delete_something = True
    # Rebuild the file to reclaim file space
    # But only if we actually did something
    if delete_something:
        repack(archive, verbose)
        
def repack(archive, verbose=False):
    if verbose: print("Reclaiming file space in", archive)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        try:
            # Copy everything from archive to tmp
            with h5py.File(archive, 'r') as A:
                with h5py.File(tmp, 'w') as f:
                    for group in A:
                        A.copy(group, f)
        finally:
            tmp.close()
        shutil.copyfile(tmp.name, archive)
        if verbose: print("Successfully reclaimed file space in", archive)
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete G_rec from archives made by dmft")
    parser.add_argument("archives", nargs="+", help="HDF5 Archives")
    parser.add_argument("-l", "--loops", type=int, help="Number of loops to keep (not counting loop-000)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose printing")
    args = parser.parse_args()

    if (args.loops is not None) and (args.loops < 1):
        raise ValueError("--loops must be positive")
    for A in args.archives:
        try:
            compress_archive(A, keep_loops=args.loops, verbose=args.verbose)
        except FileNotFoundError:
            print("Error: file", A, "does not exist.", file=sys.stderr)
        #except OSError:
        #    print("Error: unable to open file", A, ". Is it a HDF5 file?", file=sys.stderr)


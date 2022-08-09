import functools
import warnings
import logging

from dmft.faketriqs.importlogger import logger as triqslogger
try:
    from h5 import HDFArchive, HDFArchiveGroup
except ImportError:
    triqslogger.warning("triqs/h5 not found. Loading fake version.")
    from dmft.faketriqs.h5 import HDFArchive, HDFArchiveGroup

def archive_reader(func):
    """
    For a function which reads a HDFArchive as the first argument,
    it opens the archive (read-only) if a string was passed.
    """
    @functools.wraps(func)
    def reader(archive, *args, **kwargs):
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                try:
                    return func(A, *args, **kwargs)
                except Exception as e:
                    raise type(e)(f"Exception while reading {archive}.") from e
        else:
            return func(archive, *args, **kwargs)
    return reader
def archive_reader2(func):
    """
    For a function which reads a HDFArchive as the second argument,
    it opens the archive (read-only) if a string was passed.
    """
    @functools.wraps(func)
    def reader(self, archive, *args, **kwargs):
        if isinstance(archive, str):
            with HDFArchive(archive, 'r') as A:
                try:
                    return func(self, A, *args, **kwargs)
                except Exception as e:
                    raise type(e)(f"Exception while reading {archive}.") from e
        else:
            return func(self, archive, *args, **kwargs)
    return reader

def h5_write_full_path(archive, item, path):
    """
    Helper function for appending data in a HDFArchive using path specification

    Inputs:
        archive - HDFArchive, or path of a hdf5 file (opens in append mode).
        item - object you want to record
        path - list of strings. Each string is a group/subgroup, except the
            last which is the desired label for your item.
            Alternatively, path is a single string with groups delimited by
            forward slashes '/'.
    """
    # Convert a string-like path to a list-like path
    if isinstance(path, str):
        path = path.split('/')
    # If archive is path to an archive, open it.
    # HDFArchiveGroup is parent of HDFArchive and behaves the same
    if not isinstance(archive, HDFArchiveGroup):
        with HDFArchive(archive, 'a') as A:
            h5_write_full_path(A, item, path)
    else:
        # Archive is a HDFArchive instance.
        A = archive
        # iterate through the path
        for group in path[:-1]:
            # Create the group if it does not exist
            if group not in A:
                A.create_group(group)
            # Move to the subgroup
            A = A[group]
        # We are now at the end of the path
        A[path[-1]] = item

@archive_reader
def h5_read_full_path(archive, path):
    """
    Helper function for appending data in a HDFArchive using path specification

    Inputs:
        archive - HDFArchive, or path of a hdf5 file (opens in read mode).
        path - list of strings. Each string is a group/subgroup, except the
            last which is the desired label for your item.
            Alternatively, path is a single string with groups delimited by
            forward slashes '/'.
    Output:
        Whatever is recorded at that archive and path.
    """
    # Convert a string-like path to a list-like path
    if isinstance(path, str):
        path = path.split('/')
    # iterate through the path
    for group in path[:-1]:
        # Move to the subgroup
        archive = archive[group]
    # We are now at the end of the path
    return archive[path[-1]]

@archive_reader
def get_last_loop(archive):
    """
    With a HDFArchive from dmft.dmft, finds the 'loop-XXX' with
    the highest number.

    Input: archive - HDFArchive, or path to a hdf5 file.
    Output: string, the label/key
    """
    # We go over all the keys in archive which start with 'loop-'
    # Then we sort them
    # Then we take the last one
    return sorted([k for k in archive if k[0:5] == 'loop-'])[-1]

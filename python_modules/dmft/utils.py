from h5 import HDFArchive, HDFArchiveGroup

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
    # If archive is path to an archive, open it.
    # HDFArchiveGroup is parent of HDFArchive and behaves the same
    if not isinstance(archive, HDFArchiveGroup):
        with HDFArchive(archive, 'r') as A:
            return h5_read_full_path(A, path)
    else:
        # Archive is a HDFArchive instance.
        A = archive
        # iterate through the path
        for group in path[:-1]:
            # Move to the subgroup
            A = A[group]
        # We are now at the end of the path
        return A[path[-1]]

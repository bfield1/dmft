"""
Records a text file to a HDF5 archive.
Handles duplicate entries by appending a number.

I use h5py here rather than TRIQS' h5 so you can run these scripts
without installing TRIQS.
It also lets me write attributes, if I ever decide that is useful.
h5 is specialised for reading/writing TRIQS data, but h5py lets me use the
full power of HDF5.
"""
import argparse

import h5py

def write(archive, name, textfile):
    """
    Writes contents of textfile to a HDF5 archive at name

    Overwrites existing items.

    Inputs:
        archive - h5py.File or string pointing to a HDF5 archive; opens in append mode
        name - string, path in HDF5 archive to write to.
        textfile - string. Points to a text file.
    """
    if isinstance(archive, str):
        with h5py.File(archive, 'a') as A:
            write(A, name, textfile)
    else:
        with open(textfile, 'r') as f:
            text = f.readlines()
        archive[name] = text

def writeunique(archive, name, textfile, always_number=True, digits=2):
    """
    Writes contents of textfile to a HDF5 archive at name, appending a unique index

    Inputs:
        archive - h5py.File or string pointing to a HDF5 archive; opens in append mode
        name - string, path in HDF5 archive to write to.
        textfile - string. Points to a text file.
        always_number - Boolean. If true, will always append a unique index
            even if 'name' is itself unique. (Useful for consistent formatting)
        digits - positive integer. Minimum number of digits for format the
            unique index as. Puts in leading zeros.
    Output:
        String - the actual name written to
    """
    # Open the Archive
    if isinstance(archive, str):
        with h5py.File(archive, 'a') as A:
            return writeunique(A, name, textfile)
    else:
        # If name doesn't already exist, can just write
        if not always_number and name not in archive:
            write(archive, name, textfile)
            return name
        # Initialise the index
        if always_number:
            i = 0
        else:
            i = 1
        newname = name + '_{{:0{0}d}}'.format(digits).format(i)
        # Keep incrementing the index until we get a unique name
        while newname in archive:
            i += 1
            newname = name + '_{{:0{0}d}}'.format(digits).format(i)
        write(archive, newname, textfile)
        return newname

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a text file to a HDF5 archive, uniquely")
    parser.add_argument('text', help='Text file')
    parser.add_argument('archive', help='HDF5 Archive')
    parser.add_argument('name', help='Name/path to target')
    parser.add_argument('-n','--nonumber', action='store_true',
            help="Turns off automatic numbering when name is unique.")
    parser.add_argument('-d','--digits', type=int, default=2,
            help='Number of digits to format the unique index with (using leading zeros)')
    parser.add_argument('-o','--overwrite', action='store_true',
            help="Overwrites any entries at 'name' rather than making a unique name.")
    args = parser.parse_args()
    
    if args.overwrite:
        write(args.archive, args.name, args.text)
        print("Saved to", args.name)
    else:
        newname = writeunique(args.archive, args.name, args.text,
                always_number=(not args.nonumber), digits=args.digits)
        print("Saved to", newname)
    

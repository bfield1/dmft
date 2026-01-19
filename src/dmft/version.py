"""
Return the current git hash

    Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""

import os.path
import subprocess

import dmft.version

def get_git_hash():
    # We need to spawn a subprocess, which goes to the source directory, then
    # uses git to read the git hash
    dirname = os.path.dirname(dmft.version.__file__)
    output = subprocess.run(['git','rev-parse','HEAD'], cwd=dirname, check=True, text=True, capture_output=True).stdout
    return output.rstrip('\n')

if __name__ == "__main__":
    print(get_git_hash())

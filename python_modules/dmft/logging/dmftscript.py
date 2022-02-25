"""
Write the DMFT jobscript
"""

from dmft.logging.writelog import commandline_process_autoname

if __name__ == "__main__":
    commandline_process_autoname("code/jobscript")

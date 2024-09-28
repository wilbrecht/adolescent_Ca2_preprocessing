import os

"""
Note: when scanning for multiple folders, each folder name should be UNIQUE
since output file will be named after the folder name, not the entire path

FILE_CUTOFF (int): number of .tif files needed to designate as a valid dataset
"""
FILE_CUTOFF = 10


def scan_folder(input_folder):
    """
    input_folder: root of file tree to scan
    returns a set of folders within the directory tree rooted
    at input_folder containing >= 10 .tif files
    """
    valid_folders = set()
    for dirpath, dirnames, filenames in os.walk(input_folder):
        if check_files(filenames):
            dirpath.replace('\\', '/')
            valid_folders.add(dirpath)
    return valid_folders


def check_files(files):
    """
    files: list of strings representing file names
    return true if the number of .tif files is >= 10
    """
    total = 0
    for file in files:
        if file.endswith('.tif'):
            total += 1
            if total >= FILE_CUTOFF:
                return True
    return False

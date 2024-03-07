import os

def dir_check(path):

    exists = os.path.exists(path)

    if not exists:
        os.makedirs(path)


def get_file_names(dirname):
    
    file_names = []
    for dirpath, dirnames, files in os.walk(dirname):
        for name in files:
            file_names.append(name)
    return file_names    


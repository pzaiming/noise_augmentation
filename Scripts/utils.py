import os
import yaml
import torch
import torchaudio

def check(path: str, check_type: str = "exists") -> bool:
    """
    Checks a path based on the specified check type.

    Args:
        path (str): Path to be checked.
        check_type (str): The type of check to perform (default: "exists").
            - "exists": Checks if the path exists (file or directory).
            - "dir": Checks if the path exists as a directory.
            - "wav": Checks if the file is a .WAV file.
            - "txt": Checks if the file is a .txt file.
    
    Returns:
        bool: True if the path passes the check, False otherwise.
    """
    if check_type == "exists":
        return os.path.exists(path)
    elif check_type == "dir":
        return os.path.isdir(path)
    else:
        try:
            return os.path.isfile(path) and path.lower().endswith(f".{check_type}")
        except:
            raise ValueError(f"Invalid check_type: '{check_type}'.")
        
def file_check(path: str, check_type: str = "exists") -> str:
    if check(path, check_type):
        return path
    else:
        raise ValueError("Path does not exist.")

def path_builder(base_pth: str, link_pth: str, type: str = None) -> str:
    """
    Constructs a normalized path and optionally creates the directory.

    Args:
        base_pth (str): Base path to start from.
        link_pth (str): Path to be appended to the base path.
        type (str): Type of path to handle (default: None).
            - "DIR": Creates the directory if it doesn't exist.
    
    Returns:
        str: Constructed and normalized path.
    """
    new_pth = os.path.normpath(os.path.join(base_pth, link_pth))
    
    if type == "DIR":
        if not os.path.exists(new_pth):
            print(f"Making new directory: {new_pth}")
            os.makedirs(new_pth)
    return new_pth

def read_config(self, config_path: str):
    """
    Reads and loads configuration from a YAML file into the object's attributes.
    Assuuming that a self.obj has been instantiated previously, the value of this variable will be overwritten.

    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    for key, value in config_data.items():
        lowercase_key = key.lower()
        if hasattr(self, lowercase_key):
            setattr(self, lowercase_key, value)

def rm_ext(filename: str) -> str:
    """
    Removes the extension from a given filename.

    Args:
        filename (str): The filename from which to remove the extension.
    
    Returns:
        str: The filename without its extension.
    """
    root, _ = os.path.splitext(filename)
    return root

def read_audio(pth: str):
    """
    Reads an audio file using torchaudio.

    Args:
        pth (str): Path to the audio file.
    
    Returns:
        tuple: A tuple containing the waveform tensor, sample rate, and length of the audio in seconds.
    """
    waveform, sample_rate = torchaudio.load(pth)
    length = waveform.size(1) / sample_rate
    return waveform[0], sample_rate, length

def mk_file(given_pth: str, name: str, ext: str = ".txt", pth_only: bool = False) -> str:
    """
    Creates an empty file given a name, extension, and directory.

    Args:
        given_pth (str): Path to make the empty file in.
        name (str): Name of the file.
        ext (str): Extension that will be added to the end of the file (default: ".txt").
        pth_only (bool): If True, only the path will be returned without creating the file (default: False).
    
    Returns:
        str: Absolute path of the constructed file.
    """
    file_pth = os.path.join(given_pth, name + ext)
    if not pth_only:
        with open(file_pth, "a") as file:
            pass
    return file_pth

def write_template(given_pth: str, template: list, joinby: any):
    """
    Facilitates the writing of data into files following a given template.

    Args:
        given_pth (str): Path of the file to write data into.
        template (list): List of all the elements to be written into the file following a particular order.
        joinby (any): Separator string to be inserted between elements. Can be '\n' or ' '.
    """
    with open(given_pth, 'a') as file:
        file.write(joinby.join(template))
        file.write("\n")

def mk_dir(given_pth: str, name: str) -> str:
    """
    Creates a directory given a name and a directory to construct in.

    Args:
        given_pth (str): Path to make a directory in.
        name (str): Name of the directory to be made.
    
    Returns:
        str: Absolute path of the constructed directory.
    
    Raises:
        ValueError: If the given path does not exist or if the directory creation fails.
    """
    if not check(given_pth, "exists"):
        raise ValueError(f"Path given does not exist: '{given_pth}'.")
    else:
        dir_pth = os.path.join(given_pth, name)
        try:
            os.mkdir(dir_pth)
            if not check(dir_pth, "dir"):
                raise ValueError(f"Failed to create directory: '{dir_pth}'.")
        except FileExistsError:
            pass
        return dir_pth

def check_subdir(dir : str, subdir_lst: list):
    """
    Checks a given directory if it has the required subdirectories.

    Args:
        dir (str) : The directory you wish to check for existence of subdirs.
        subdir_lst (list) :  A list of subdirectories to validate for existence.

    Returns:
        Prints missing directories, if any.
    """
    for subdir in subdir_lst:
        try:
            check(path_builder(dir, subdir, "dir"))
        except:
            print(f"{subdir} cannot be found.")

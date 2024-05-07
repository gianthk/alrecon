import os
from os import getlogin, path
import pandas as pd


def collect_recon_paths(root_dir):
    """
    check for .Trash or empty dirs?
    could be improved for speed, I have the impression that "it walks too much"
    """

    dict_recon_paths = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Temporarily store directories that match 'recon'
        temp_recon_dirs = [d for d in dirnames if d.startswith("recon")]

        # For each directory that starts with 'recon', find the first parent that doesn't
        for dirname in temp_recon_dirs:
            full_path = os.path.join(dirpath, dirname)
            # Split the path to get all parts
            parts = full_path.split(os.sep)
            # Find the first parent directory not starting with 'recon'
            for part in reversed(parts[:-1]):
                if not part.startswith("recon"):
                    if part not in dict_recon_paths:
                        dict_recon_paths[part] = []
                    dict_recon_paths[part].append(full_path)
                    break

        # Ensure we still traverse into subdirectories of 'recon'
        # reference to dirnames
        dirnames[:] = [d for d in dirnames if d in temp_recon_dirs or not d.startswith("recon")]
        # print("dirnames", dirnames)

    df_rows = [(key, path) for key, paths in dict_recon_paths.items() for path in paths]
    df = pd.DataFrame(df_rows, columns=["Parent Directory", "recon_dir"])

    return df


from pathlib import Path


def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)


def settings_file():
    # check if user settings file exists
    user_settings_file = get_project_root() + "/settings/" + getlogin() + ".yml"

    if path.isfile(user_settings_file):
        return user_settings_file
    else:
        return get_project_root() + "/settings/default.yml"
        # return get_project_root() + "/settings/default_test_locally.yml"

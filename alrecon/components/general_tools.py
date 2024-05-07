import os
from os import getlogin, path
import pandas as pd

import solara


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


# ===== # ====== # INPUT AND OUTPUT

import yaml


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    pass


# Define a function that simply ignores complex tags
def ignore_complex_objects(loader, node):
    return None  # We're ignoring any complex objects


# Register the function to ignore tags
SafeLoaderIgnoreUnknown.add_constructor(None, ignore_complex_objects)


def save_app_settings(self, filename):
    self.update_settings_dictionary()

    print(self.settings)

    # write YAML settings file
    with open(filename, "w") as file:
        yaml.dump(self.settings, file)


def init_settings(self, filename):
    with open(filename, "r") as file_object:
        self.settings_file = solara.reactive(path.basename(filename))
        self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)
        self.check_settings_paths()

        # initialize app settings from YAML file
        for key, val in self.settings.items():
            exec("self." + key + "=solara.reactive(val)")


def load_app_settings(self, filename):
    with open(filename, "r") as file_object:
        self.settings_file.set(path.basename(filename))
        # self.settings = yaml.load(file_object, Loader=yaml.SafeLoader)
        self.settings = yaml.load(file_object, Loader=SafeLoaderIgnoreUnknown)

        # some app settings
        for key, value in self.settings.items():
            try:
                exec("self." + key + ".set(value)")
            except Exception as e:
                print(e)
                print(key, "    ", value)

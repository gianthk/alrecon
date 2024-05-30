import os
from os import getlogin, path
import pandas as pd

import solara


import re

from pathlib import Path


def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)


def sorted_alphanum(listOfStrings):
    """Sorts the given iterable in the way that is expected. Converts each given list to a list of strings

    Required arguments:
    listOfStrings -- The iterable to be sorted.

    """

    listOfStrings = [str(i) for i in listOfStrings]

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(listOfStrings, key=alphanum_key)


def collect_recon_paths(root_dir, search_string):
    """
    check for .Trash or empty dirs?
    could be improved for speed, I have the impression that "it walks too much"
    """

    dict_recon_paths = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Temporarily store directories that match 'recon'
        temp_recon_dirs = [d for d in dirnames if d.startswith(search_string)]

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
        dirnames[:] = [d for d in dirnames if d in temp_recon_dirs or not d.startswith(search_string)]
        # print("dirnames", dirnames)

    return dict_recon_paths


def get_dict_files_with_ending(source_dir, file_ending):
    found_files = [os.path.join(d, x) for d, dirs, files in os.walk(source_dir) for x in files if x.endswith(file_ending)]
    dict_found_files = {os.path.splitext(os.path.basename(file))[0]: file for file in found_files}
    return dict_found_files


def message_if_list_empty(key, should_be_a_list):
    len_of_list = len(should_be_a_list)
    if len_of_list > 0:
        keys_corresponding_to_path = [""] * len_of_list
        keys_corresponding_to_path[0] = key
        return [keys_corresponding_to_path, should_be_a_list]
    else:
        return [[key], ["no reconstructions so far"]]
        # return [[key], ["      "]]


def return_number_of_files_in_folder(path):
    if os.path.isdir(path):
        return len(os.listdir(path))
    return ""


def does_its_path_contain(list_of_strings, path):
    """
    list_of_strings must be a list!!!!
    """

    for string_ in list_of_strings:
        if string_ in path:
            return "completed"
    return "     "


def provide_unified_data_frame_from_dicts(dict_exp_files, dict_recon_paths, list_of_marker_for_completed_recons):
    """
    should contain:
    exp name1 - recon name - elements in folder - paths
                recon name - elements in folder - paths
                recon name - elements in folder - paths
    exp name2 - recon name - elements in folder - paths
                recon name - elements in folder - paths

    # für jeden key der experimente brauchen wir mindestens einen recon gegenwert, ansonsten schreiben wir ein "missing oä". alle anderen kommen in den key "others"
    """

    keys_exp = sorted_alphanum(dict_exp_files.keys())

    df_rows = [
        (key_corresponding_to_path, return_number_of_files_in_folder(path), path, does_its_path_contain(list_of_marker_for_completed_recons, path))
        for key in keys_exp
        for key_corresponding_to_path, path in zip(*message_if_list_empty(key, sorted_alphanum(dict_recon_paths.pop(key, []))))
    ]

    remaining_datasets = dict_recon_paths.values()  # unclassified

    def flatten_concatenation(list_of_lists):
        flat_list = []
        for row in list_of_lists:
            flat_list += row
        return flat_list

    remaining_datasets = flatten_concatenation(remaining_datasets)

    df_rows_remaining = [
        (key_corresponding_to_path, return_number_of_files_in_folder(path), path, does_its_path_contain(list_of_marker_for_completed_recons, path))
        for key_corresponding_to_path, path in zip(*message_if_list_empty("others", sorted_alphanum(remaining_datasets)))
    ]

    df_rows += df_rows_remaining

    df = pd.DataFrame(df_rows, columns=["exp dataset", "number_of_files", "recon_dir", "recon complete?"])

    return df


def create_dataoverview_BEATS(exp_dir, recon_dir, list_of_marker_for_completed_recons=["_finished_reconstructions"]):
    dict_exp_files = get_dict_files_with_ending(source_dir=exp_dir, file_ending=".h5")

    dict_recon_paths = collect_recon_paths(root_dir=recon_dir, search_string="recon")

    df = provide_unified_data_frame_from_dicts(dict_exp_files, dict_recon_paths, list_of_marker_for_completed_recons)

    return df


# ==== # ==== # ====
# ==== # ==== # ====
# ==== # ==== # ====
# ==== # ==== # ====
# ==== # ==== # ====


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

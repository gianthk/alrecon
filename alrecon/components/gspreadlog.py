import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe


scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

creds = ServiceAccountCredentials.from_json_keyfile_name("./../keys/alrecon_v1-f048b037c351.json", scopes=scopes)

def log_to_gspread(settings):
    # access master
    file = gspread.authorize(creds)
    workbook = file.open(settings['master'])
    sheet = workbook.sheet1

    # update master when button is pushed
    master = get_as_dataframe(sheet).dropna(axis=0, how='all')

    # check if dataset is already in master
    if np.any(master['scan'].isin([settings['dataset']])):
        print("Found dataset in master.")

        # check if dataset was already reconstructed
        if master.loc[master["scan"] == settings['dataset'], "reconstructed"][0]:

            # if already reconstructed create a new line
            newline = master.loc[master["scan"] == settings['dataset']]

            # if multiple rows exist for 1 scan take the first one
            if len(newline) > 1:
                newline = newline.loc[[0]]

            # log reconstruction settings
            newline["cor"] = COR
            newline["pad"] = pad
            newline["alpha"] = alpha
            master = pd.concat([master, newline])
        else:
            # log settings as new reconstruction line
            master.loc[master["scan"] == filename, "cor"] = COR
            master.loc[master["scan"] == filename, "pad"] = pad
            master.loc[master["scan"] == filename, "alpha"] = alpha

    else:
        # log everything (including dataset name) to a new line
        print("Dataset is not in master")
        master.loc[len(master), ["scan", "cor", "pad", "alpha"]] = [filename, COR, pad, alpha]

    # update master sheet
    set_with_dataframe(sheet, master)

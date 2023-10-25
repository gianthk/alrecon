import pandas as pd
import numpy as np
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# key_file = '/home/gianthk/PycharmProjects/BEATS/alrecon/keys/alrecon-f048b037c351.json'
key_file = 'keys/alrecon-f048b037c351.json'
# key_file = './../keys/alrecon_v1-f048b037c351.json'

creds = ServiceAccountCredentials.from_json_keyfile_name(filename=key_file, scopes=scopes)

def spreadsheetname(experiment_name=''):
    if experiment_name == '':
        return 'master'
    else:
        return experiment_name+'_master'

def add_to_master(master, settings):
    """Add entry to master dict based on settings dict

    Parameters
    ----------
    master : dict
        2D master table dicitonary
    settings : dict
        1D reconstruction settings dictionary
    """

    masterSet = set(master)
    settingsSet = set(settings)

    dataset_row = len(master)
    for name in masterSet.intersection(settingsSet):
        # print(settings[name].len())
        # print(master.loc[dataset_row, name])
        master.loc[dataset_row, name] = settings[name]

def log_to_gspread(settings):
    # access master spreadsheet
    file = gspread.authorize(creds)
    # print(settings['experiment_name'])
    # print(spreadsheetname(settings['experiment_name']))
    workbook = file.open(spreadsheetname(settings['experiment_name']))
    sheet = workbook.sheet1

    # pull master from spreadsheet
    master = get_as_dataframe(sheet).dropna(axis=0, how='all')

    # add reconstruction entry to master
    add_to_master(master, settings)

    # update master sheet
    set_with_dataframe(sheet, master)

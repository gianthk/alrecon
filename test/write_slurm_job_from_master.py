#!/usr/bin/env python
from alrecon.components import gspreadlog, slurm
from os import path
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials

def master_line_to_settings(dataframe):
    settings = []
    settings.recon_script = 'BEATS_recon.py'
    settings.h5file = path.join("/PETRA/SED/BEATS/SEM_6/20235081/ExpData/20235081", dataframe['dataset'][0] + '.h5')
    settings.recon_dir = dataframe['recon_dir'][0]

key = '/home/beatsbs/PycharmProjects/alrecon/keys/alrecon-e50b00b0b6f6.json' # beatsbs@BL-BEATS-WS01
key = '/home/beats/PycharmProjects/BEATS/alrecon/keys/alrecon-e50b00b0b6f6.json' # beats laptop
master_spreadsheet = 'foo'

scopes = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive'
          ]

creds = ServiceAccountCredentials.from_json_keyfile_name(filename=key, scopes=scopes)

# access master spreadsheet
file = gspread.authorize(creds)

# load spreadsheet contents
workbook = file.open(master_spreadsheet)
sheet = workbook.sheet1

# pull master from spreadsheet
master = get_as_dataframe(sheet).dropna(axis=0, how='all')

# select lines marked for run
master_run = master.loc[master['run_recon'] == 1.0]

#

print('here')
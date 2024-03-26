"""
Class for integration of online logging to Google spreadsheet using gspread.
For more information, visit the project homepage:
	https://github.com/gianthk/alrecon
"""

import logging

try:
    import gspread
    from gspread_dataframe import get_as_dataframe, set_with_dataframe
    from oauth2client.service_account import ServiceAccountCredentials
except ImportError:
    # raise ImportError('gspread not available for dataframe logging.')
    logging.error('gspread not available for dataframe logging.')

def spreadsheetname(experiment_name=''):
    if experiment_name == '':
        return 'master'
    else:
        return experiment_name+'_master'

def add_to_master_dict(master, settings):
    """Add entry to master dictionary based on settings dictionary

    Parameters
    ----------
    master : dict
        2D master table dictionary
    settings : dict
        1D reconstruction settings dictionary
    """

    masterSet = set(master)
    settingsSet = set(settings)

    dataset_row = len(master)
    for name in masterSet.intersection(settingsSet):
        master.loc[dataset_row, name] = settings[name]

class logger:
    def __init__(self, key='', experiment_name=''):
        self.glogging = True
        self.key = key
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets',
                       'https://www.googleapis.com/auth/drive'
                       ]

        try:
            import gspread, gspread_dataframe, oauth2client
        except:
            self.glogging = False

        if self.glogging:
            self.creds = ServiceAccountCredentials.from_json_keyfile_name(filename=self.key, scopes=self.scopes)
        else:
            self.creds = None

    def log_to_gspread(self, settings, master=None, sheet=None):
        master, sheet = self.read_gspread_master(settings)

        if master is not None:
            # add reconstruction entry to master
            add_to_master_dict(master, settings)

            # update master sheet
            set_with_dataframe(sheet, master)

    def read_gspread_master(self, settings):
        if self.glogging:
            # access master spreadsheet
            file = gspread.authorize(self.creds)

            # modify the following line to store spreadsheet name at object init ??
            try:
                workbook = file.open(settings['master_spreadsheet'])
                sheet = workbook.sheet1

                # pull master from spreadsheet
                master = get_as_dataframe(sheet).dropna(axis=0, how='all')

                return master, sheet

            except:
                logging.error('Master spreadsheet {0} does not exist. Check that spreadsheet exists and is shared.'.format(settings['master_spreadsheet']))
                return None
        else:
            logging.error('gspread not available for dataframe logging.')
            return None
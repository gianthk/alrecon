# Alrecon (الريكُن)
Pure Python Computed Tomography reconstruction web application. Built with [Solara](https://solara.dev/).

[![GitHub license](https://img.shields.io/github/license/gianthk/alrecon)](https://github.com/gianthk/alrecon/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/701693534.svg)](https://zenodo.org/doi/10.5281/zenodo.10535211)

![](docs/alrecon_home.gif)

### How to cite
If you use `alrecon` please cite:
>  Iori et al., (2024). Alrecon: computed tomography reconstruction web application based on Solara. Open Research Europe, 4(54). https://doi.org/10.12688/openreseurope.16863.1  <br>

## Installation
<!-- Install `alrecon` using pip. The flag `[all]` will install optional dependencies required for integration with [`napari`](https://napari.org) and logging to google spreadsheets.
```commandline
pip install alrecon[all]
``` -->

1. Checkout this repository:
```commandline
git clone https://github.com/gianthk/alrecon.git
```
2. Navigate to the repository and create a virtual environment with all necessary dependencies:
```commandline
cd alrecon
conda env create --file envs/alrecon-base.yml
conda activate alrecon-base
```
> [!NOTE]
> If you already have a destination virtual environment for alrecon, you can install manually the requirements listed in file [alrecon.yml](envs/alrecon-base.yml).

> [!NOTE]
> To use TomoPy with CUDA features, install TomoPy from conda following [these instructions](https://tomopy.readthedocs.io/en/stable/install.html).

3. Build the `alrecon` app:
```commandline
pip install -e .
```

## Usage
### Run the `alrecon` [solara](https://solara.dev/api/file_browser) web application on your localhost
```commandline
solara run alrecon.pages --host localhost
```
### Run `alrecon` within [jupyter](https://solara.dev/api/file_browser)
1. Make sure that the `alrecon` virtual environment is activated and `ipykernel` installed:
```commandline
conda activate alrecon
pip install --user ipykernel
```
or:
```commandline
conda install -c anaconda ipykernel 
```
2. Install [ipykernel](https://github.com/ipython/ipykernel) with the `alrecon` virtual environment:
```commandline
python -m ipykernel install --user --name=alrecon
```
3. Launch Jupyter Notebook or Jupyter Lab instance:
```commandline
jupyter lab
```
4. Open and run the cells of the notebook [launch_within_jupyter.ipynb](launch_within_jupyter.ipynb). [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](launch_within_jupyter.ipynb)

You can take a look at [solara's documnetation](https://solara.dev/api) for more information on the different ways of running solara applications.

---
## Integration with Google Sheets
- `Alrecon` allows you to keep a consistent log of all reconstruction jobs on an online master Google spreadsheet.
- The integration uses [gspread](https://docs.gspread.org/en/v5.12.0/), a Python API for Google Sheets.
- You will need to setup your Google Cloud account, enable [Google Sheets API](https://developers.google.com/sheets/api/guides/concepts), and create a Secret Key with read/write permission to your online master Google sheet. Follow [these instructions](https://www.youtube.com/watch?v=hyUw-koO2DA) to setup the integration with Google Sheets API.

## Use with [napari](https://napari.org/stable/)
[napari](https://napari.org/stable/) is a powerful pure Python multi-dimensional image viewer. Alrecon supports napari only when [running the app through Jupyter](#run-al-recon-within-jupyter).

## Setup [ImageJ](https://imagej.net/software/fiji/) launcher
To launch [ImageJ](https://imagej.net/software/fiji/) from the alrecon web app follow these steps:
1. Modify the path to your ImageJ executable in the alrecon general settings
![alrecon imagej executable path setting](docs/pictures/alrecon_imagej.png)

2. Copy the [FolderOpener_virtual.ijm](/imagej_macros/FolderOpener_virtual.ijm) ImageJ macro contained in `/alrecon/imagej_macros/` to the plugin folder of your ImageJ installation. On Linux this is something like `/opt/Fiji.app/macros/`.

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No [[822535]](https://cordis.europa.eu/project/id/822535)(Beamline for Tomography at SESAME [BEATS](https://beats-sesame.eu/)). 

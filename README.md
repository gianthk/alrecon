# Al-recon
Pure Python Computed Tomography reconstruction web application. Built with [Solara](https://solara.dev/).

[![GitHub license](https://img.shields.io/github/license/gianthk/alrecon)](https://github.com/gianthk/alrecon/blob/master/LICENSE)

<!-- Micro Finite Element (microFE) models can be derived from micro Computed Tomography (microCT) 3D images to non-destructively assess mechanical properties of biological or artificial specimens. <br />
**ciclope** provides fully open-source pipelines from microCT data preprocessing to microFE model generation, solution and postprocessing. <br /> -->

![](docs/alrecon_home.gif)

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
conda env create --file alrecon.yml
conda activate alrecon
```
> [!NOTE]
> If you already have a destination virtual environment for alrecon, you can install manually the requirements listed in file [alrecon.yml](alrecon.yml).

3. Build the `alrecon` app:
```commandline
cd alrecon
pip install -e .
```

## Usage
### Run the `al-recon` [solara](https://solara.dev/api/file_browser) web application on your localhost
```commandline
solara run alrecon.pages --host localhost
```
### Run `al-recon` within [jupyter](https://solara.dev/api/file_browser)
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
- You will need to setup your Google account, enable Google Sheets API, and create a Secret Key for accessing your online master. Follow [these instructions](https://www.youtube.com/watch?v=hyUw-koO2DA) to setup the integration with Google Sheets.

## Use with [napari](https://napari.org/stable/)
napari is a pure Python powerfull viewer for multi-dimensional images. Until now, alrecon supports napari only when [running the app through Jupyter](#run-al-recon-within-jupyter).

## Acknowledgements
This work was performed within the [BEATS](https://beats-sesame.eu/) project and has received funding from the EU’s H2020 framework programme for research and innovation under grant agreement n° [822535](https://cordis.europa.eu/project/id/822535).
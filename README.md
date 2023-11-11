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
2. Navigate to the repository and create a virtual environment with the dependencies required by the application:
```commandline
cd alrecon
conda env create --file alrecon.yml
conda activate alrecon
```
[!NOTE]
If you already have a virtual environment, you can install manually the requirements listed in file [alrecon.yml](alrecon.yml).

3. Build the `alrecon` app:
```commandline
cd alrecon
pip install -e .
```

4. Run the `alrecon` web application on localhost:
```commandline
solara run alrecon.pages --host localhost
```

### How to contribute
If you want to contribute to this project, please install `alrecon` following the [development guide](development.md).

## Usage
### Run the `al-recon` [solara](https://solara.dev/api/file_browser) web app on your localhost
```commandline
solara run alrecon.pages --host localhost
```
### Run `al-recon` within [jupyter](https://solara.dev/api/file_browser)
1. Install Jupyter kernel with the alrecon environment:

2. Launch a Jupyter Notebook or Jupyter Lab instance:
```commandline
jupyter lab
```

3. Open and run the cells of the notebook [launch_within_jupyter.ipynb](launch_within_jupyter.ipynb). [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](launch_within_jupyter.ipynb)

You can take a look at [solara's documnetation](https://solara.dev/api) for more information on the different ways of running solara applications.

## Notes on ciclope
* Tetrahedra meshes are generated with [pygalmesh](https://github.com/nschloe/pygalmesh) (a Python frontend to [CGAL](https://www.cgal.org/))
* High-resolution surface meshes for visualization are generated with the [PyMCubes](https://github.com/pmneila/PyMCubes) module.
* All mesh exports are performed with the [meshio](https://github.com/nschloe/meshio) module.
* **ciclope** handles the definition of material properties and FE analysis parameters (e.g. boundary conditions, simulation steps..) through separate template files. The folders [material_properties](/material_properties) and [input_templates](/input_templates) contain a library of template files that can be used to generate FE simulations.
  * Additional libraries of [CalculiX](https://github.com/calculix) examples and template files can be found [here](https://github.com/calculix/examples) and [here](https://github.com/calculix/mkraska)

## Acknowledgements
This project was partially developed during the Jupyter Community Workshop [“Building the Jupyter Community in Musculoskeletal Imaging Research”](https://github.com/JCMSK/2022_JCW) sponsored by [NUMFocus](https://numfocus.org/).
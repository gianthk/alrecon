## the following commands were used to create a virtual environment for alrecon with full GPU functionalities using astra-toolbox and tomopy
```commandline
conda create --name tomopy python=3.8
conda activate tomopy
conda install -c astra-toolbox astra-toolbox
conda install -c conda-forge tomopy
conda install dxchange pandas tifffile
conda install -c conda-forge dxchange pandas tifffile
pip install pypng matplotlib gspread gspread-dataframe oauth2client
pip install solara
```

---
## the following is an attempt to install tomopy and alrecon in development mode in a virtual environment with the astra-toolbox

### 1. clone the TomoPy GitHub repository
### 2. create virtual env
```commandline
conda env create -f /tomopy/envs/linux-cuda.yml --name alrecon python=3.8
conda activate alrecon
```

### 3. install astra-toolbox
```commandline
conda install -c astra-toolbox astra-toolbox
```

### 4. build TomoPy
```commandline
cd tomopy/
pip install . --no-deps
mkdir build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_INSTALL_LIBDIR=lib
cmake --build .
cmake --install .
```

### 5. Clone the alrecon GitHub repository
### 6. Install the alrecon dependencies
```commandline
cd alrecon/
conda install -c plotly plotly
conda install pandas tifffile 
pip install pypng matplotlib gspread gspread-dataframe oauth2client
pip install solara
```
### 7. Build alrecon
```commandline
pip install -e .
```

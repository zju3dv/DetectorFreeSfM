# Installation

## Environment
Download third party libraries:
```shell
git clone https://github.com/zju3dv/DetectorFreeSfM.git
cd DetectorFreeSfM

# Clone third party modules:
git submodule update --init
```

Install python environment:
```shell
# Be sure that you are currently under the repo directory:
# NOTE: we assume the cuda version is 11.7 by default. If you are using other cuda version, please modify the environment.yaml accordingly (Line. 12).
conda env create -f environment.yaml
conda activate detectorfreesfm
```

Install third party moduels:
```shell
# Install RoIAlign, which is usded in our multi-view refinement matching phase:
cd third_party/RoIAlign.pytorch && pip install .
```
Install [multi-view evaluation tool](https://github.com/ETH3D/multi-view-evaluation) follow their instruction (used to evaluate ETH3D's triangulation metrics).

Download pretrained weights from [here](https://drive.google.com/file/d/1phP6U1CQ7jo1ZfUZ0xRYDf0IBZX_t9qb/view?usp=sharing) and place it under repo directory. Then unzip it by running the following command:
```shell
# Be sure that you are currently under the repo directory:
unzip weight.zip
rm -rf weight.zip
```

## Modified COLMAP version installation
### Why?
Our project is partly based on COLMAP. We modified the COLMAP to implement geometry refinement module, which is a core part in our SfM refinement pipeline.

### Install 
```shell
apt-get install -y git \
    	cmake \
    	build-essential \
    	libboost-program-options-dev \
    	libboost-filesystem-dev \
    	libboost-graph-dev \
    	libboost-system-dev \
    	libboost-test-dev \
    	libeigen3-dev \
    	libsuitesparse-dev \
    	libfreeimage-dev \
        libgoogle-glog-dev \
    	libgflags-dev \
    	libglew-dev \
    	qtbase5-dev \
    	libqt5opengl5-dev \
    	libcgal-dev \
		libmetis-dev \
    	&& apt-get install -y libcgal-qt5-dev \
        && apt-get install -y libatlas-base-dev libsuitesparse-dev 
```

```shell
# Install ceres-solver:
cd path/to/your/desired/ceres/installation/directory
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver && git checkout 1.14.x
mkdir build && cd build && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && make -j && make install
```

```shell
# Install the folk-and-modified COLMAP:
cd path/to/your/desired/colmap/installation/directory
git clone https://github.com/hxy-123/colmap.git
cd colmap && mkdir build && cd build && cmake .. && make -j 

# If you have sudo permission, you can install COLMAP by:
sudo make install

# If you do not have sudo permission, keep in mind your colmap exe path and export it to your environment variable:
export COLMAP_PATH=/your/path/colmap/build/src/exe/colmap
```
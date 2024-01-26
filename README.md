logo - badges

## Basic overview
cm_biocytin is a simple way to extract data from neuron 3D reconstructions (from SNT fiji plugin) and build a model to predict their firing properties from their morphology.

## key features

## Usage

### Input

1. Deposit your images and traces from the SNT semi-automated reconstruction in the "input" folder.

### Running the script

1. Activate pyimagej virtual environment before running the script. Then run the script "filling_extraction.py".
'''bash
mamba activate pyimagej
python filling_extraction.py
'''

2. Remember to deactivate your pyimagej environment when done.
'''bash
mamba deactivate
'''

3. Use Jupyter notebook to investigate the best scikit-learn model to fit your dataset.

## Installation

The script necessitate python 3.8 as well as Maven and openjdk 11.0 in order to work (works with Jython).
The script for tracing and extraction of the data from the neuron trace needs to run from a pyimagej environnement.

### On Linux

1. If you don't have miniconda installed nor mamba to set up the environment you shall install it calling:

'''bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
'''

or

'''bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
'''

2. If you don't have java / openjdk installed: install it and check on your installation.

'''bash
sudo apt-get install openjdk-11-jdk
java --version
'''

3. Install pyimagej into a new virtual environment.

'''bash
mamba create -n pyimagej pyimagej openjdk=11
'''

4. Testing your installation:

'''bash
mamba activate pyimagej
python -c 'import imagej; ij = imagej.init("2.14.0"); print(ij.getVersion())'
mamba deactivate
'''

## Configuration options & troubleshoot

- Often the java heap size is not sufficient enought to run the script on large images/datasets. You can simply increase the java heap size if so you get the following error:
'''bash
java.lang.OutOfMemoryError: Java heap space
'''
or if the script breaks before the end.

## Contribution

## Licence

## Aknowledgements

This project wouldn't be possible without SNT add-on for ImageJ developped by Tiago Ferreira nor the constent enrichments that the community of https://forum.image.sc provides.

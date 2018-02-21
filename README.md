# SAMkNN

## Installing nearest neighbor library:
Before using SAM you have to install a C++ library for the nearest neighbor calculations.
- Install the C++ library by using the common setup.py script e.g. "python setup.py install --user".
- Edit the nearestNeighbor/setup.py to adapt it to your local environment (usually not necessary).


## Using SAM
In SAMKNN/testSAMKNN.py you can find a test script which applies the SAMKNN model on a dataset. Simply execute "python testSAMKNN.py". 

## Datasets
Two exemplary datasets "weather" and "moving squares" can be found in the datasets folder. More drift datasets can be found in the repository https://github.com/vlosing/driftDatasets

# mBLA_UF
The modified boundary layer approximation (mBLA) of the concentration polarization layer in the cross-flow ultrafiltration proposed by Park and Naegele [1, 2]. The first version (v1.0) of this source code (used in [1]) is also published on Zenodo with DOI number https://doi.org/10.5281/zenodo.3895786 . The third version (specifically v3.1) used in [2] is published on Zenodo with DOI number https://doi.org/10.5281/zenodo.5658545 . The comments inside code explain related equations in Refs. [1, 2].

[1] Park and Naegle, Journal of Chemical Physics, 2020 (https://doi.org/10.1063/5.0020986)
[2] Park and Nagele, Membranes, 2021 (https://doi.org/10.3390/membranes11120960)

# Version and environment
## Version 3.x
Now, we account the additional geometries composed of the flat sheet of membrane. Many of cross-reference is added inside the code compared to the paper [1] and [2]. The proper definition is provided if the one is different from Ref. [2].

## Version 2.0
With the major revision of code and adjusting the numerical methodology slightly for better readability and stability, now we launched version 2.0. This code is compatible with Python 3.8.x environment and requires Scipy, Numpy, and Matplotlib.

## Version 1.0
This is the original code used in the paper (Park and Naegele, JCP, 2020) with the tag name of v1.0 (stands for version 1.0). The v1.0 was developed/tested under Python 2.7.x environments with the additional support of libraries Scipy, Numpy, and Matplotlib.



# Usage
Run the below code to see basic arguments and output:
'''
Python scripts/cal_phiw_from_input.py
'''

The example is located at "test_example". Try to run below example inside "test_example" folder:
'''
python ../scripts/cal_phiw_from_input.py inp_v3.py inp_v3.py.dat
'''



# mBLA_UF
These scripts provide the calculation of matched asymptotic solution of the concentration-polarization layer in the cross-flow ultrafiltration proposed by Park and Naegele (JCP, 2020, https://doi.org/10.1063/5.0020986). The first version (v1.0) of this source code is also published on Zenodo with DOI number https://doi.org/10.5281/zenodo.3895786 . 

# Version and environment
## Version 3.0
Now, we account the additional geometries composed of the flat sheet of membrane. The details of such studies will be explained soon when the manuscript is accepted.

## Version 2.0
With the major revision of code and adjusting the numerical methodology slightly for better readability and stability, now we launched version 2.0. This code is compatible with Python 3.8.x environment and requires Scipy, Numpy, and Matplotlib.

## Version 1.0
This is the original code used in the paper (Park and Naegele, JCP, 2020) with the tag name of v1.0 (stands for version 1.0). The v1.0 was developed/tested under Python 2.7.x environments with the additional support of libraries Scipy, Numpy, and Matplotlib.



# Usage
"test_examples/run_test.sh" contains the basic test setup for computing semi-analytic solution of the model and reporting its result. The code can be directly used by the main function located in "scripts/cal_phiw_from_input.py". In the default folder, the basic test can be run by
"python scripts/cal_phiw_from_input.py inp_ref.py test.dat"
where inp_ref.py is the python script with pre-given system parameters, and test.dat is the output file once the computation is done. The aditional log file will be provided with the extension of .log. This log file contains a summary of conditions and logs of reports during iteration steps.

# Inter-link of codes with notations and equations in paper
This is provided inside the help-doc or comments where the exact location of the equation in the manuscript. 

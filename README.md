# mBLA_UF
This scripts are providing the calculation of matched asymptotic solution of the concentration-polarization layer in the cross-flow ultrafiltration. The manuscript is accepted and now in production for publication steps. The links will be embedded soon after the paper is published online.

# Version and environment
The original v1 code is applicable under Python 2.7.x whereas new version (v2, not tagged yet)is avaliable with Python 3.8.x. It contains basic scientific libraries for Python such as Scipy, Numpy, and Matplotlib. On top of that, several Python packages will be used.

# Usage
"inp_ref.py" contains the basic test setup for computing semi-analytic solution of the model. This can be used directly by the main function located in "scripts/cal_phiw_from_input.py". In the default folder, the basic test can be run by
"python scripts/cal_phiw_from_input.py inp_ref.py test.dat"
where test.dat is for the output file. Note that there will be one additional output file usually put additional extension ".log" after the output file. In this example, the file name for the log file will be "test.dat.log" which contains summary of conditions and iteration log at each fixed-point iteration (FPI) to solve the semi-analytic solution

# Inter-link between codes and paper
This is provided inside the help-doc or comments where the exact location of equation in the manuscript. 



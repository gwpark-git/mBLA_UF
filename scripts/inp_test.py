#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


import scipy.constants as const
from numpy import *

# Parameters for numerical computation

N_iter = 100          # Maximum iteration for FPI
weight = 0.1          # Relaxation factor for FPI
Nz = 100              # Z-discritization number
Nr = 100              # Bulk R-discritization number (note that the boundary-layer R-disc. has a different number)
IDENT_parallel = True # True/False : parallel computation

# Methodology
IDENT_modification = True   # True for newly proposed version and False for the original version 

# Additional option
IDENT_verbose = False  # If True, it record all the result steps

# System parameters

R_channel = 0.5/1000. # m          : Radius of channel
L_channel = 0.5       # m          : Length of channel
Lp = 6.7e-10          # m/(Pa*sec) : Solvent-permeability of membrane
eta0 = 1e-3           # Pa*sec     : Solvent viscosity
T = 293.15            # K          : Temperature
ref_Pout = 101325     # Pa         : Outlet pressure
ref_DTP = 5000        # Pa         : Transmembrane pressure (linear approximation)
DLP = 5000            # Pa         : Longitudinal pressure difference

# Dispersion properties
phi_bulk = 1e-3   # Inlet volume fraction of particle
gamma = 1.0       # Solvent-permeability to hard spheres
a_particle = 1e-8 # m : Radius of particle

# Transport properties
from osmotic_pressure_CS import *
fcn_Pi_given = Pi_CS

from transport_properties_SPHS import *

fcn_eta_given = eta_div_eta0_SPHS
fcn_Dc_given = Dc_short_div_D0_SPHS

# Initial conditions
phiw_arr = ones(Nz)*phi_bulk 


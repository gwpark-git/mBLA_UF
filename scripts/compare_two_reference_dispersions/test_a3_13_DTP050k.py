#############################################################################
#   Test input file for mBLA_UF code                                        #
#                                                                           #
#   Used in the paper:                                                      #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   doi: 10.1063/5.0020986                                                  #
#   Code Developer: Park, Gun Woo    (g.park@fz-juelich.de)                 #
#   MIT Open License (see LICENSE file in the main directory)               #
#############################################################################

import scipy.constants as const
from numpy import *


membrane_geometry = 'HF'
                          # Concerning about the membrane geometry
                          # At this moment (8 JUN 20201), there are three possible geometries
                          # (1) 'HF' (== 'hollow fiber')
                          # (2) 'FMM' (== 'flat membrane/membrane')
                          # (3) 'FMS' (== 'flat membrane/substrate')
                          # Default is 'HF' as we did in past

R_channel = 0.5/1000.     # m          : Radius of channel
L_channel = 0.5           # m          : Length of channel

                          # Which values will describe membrane permeability?
                          # (1) 'membrane' means the membrane permeability Lp (see our paper)
                          # (2) 'Darcy' means the Darcy's permeability kappa_Darcy (see our paper)
                          # Note that the Lp comes from Darcy-Straling permeate flux: vw = Lp (P - Pper - Pi)
                          # whereas Darcy's permeability kappa_Darcy will calculate LP by using
                          # integration over the thickness of membrane (homogeneous membrane only)
                          # The required input for (1) is 'Lp' without additional parameter
                          # The required input for (2) is 'kappa_Darcy' and 'h_membrane' which is thickness of membrane
                          # In the case (2), it eventually calculate Lp, then there is no difference

define_permeability = 'membrane'
# kappa_Darcy = 1.3583e-16  # m^2 : Darcy's permeability of membrane
# h_membrane = R_channel/2. # the reference thickness is half of channel-radius
                          # When darcy's permeability is given, Lp will be calculated in according to the membrane geometry
Lp = 6.7e-10          # m/(Pa*sec) : Solvent-permeability of membrane
                          # Parameters for numerical computation

BC_inlet = 'velocity'
                          # this can be either 'pressure' or 'velocity'
                          # the case 'pressure': P_in will be re-calculated based on the DLP and ref_Pout. Then u_ast will be set with u_HP
                          # the case 'velocity': u(0,0) will be specified based on u_ast value. Then, u_ast is replaced by the value itself.
                          # for 'pressure':
# DLP = 1300                 # Pa         : Longitudinal pressure difference
                          # fore 'velocity':
# u_inlet = 0.01625 # m/s: the reference value
u_inlet = 0.1625
ref_Pout = 101325         # Pa         : Outlet pressure

BC_perm = 'DTP'  # either DTP or Pperm
ref_DTP = 50000            # Pa         : Transmembrane pressure (linear approximation)

# BC_perm = 'Pper'  # either DTP or Pperm
# ref_Pperm = 96381 # test
# ref_DTP = 5000            # Pa         : Transmembrane pressure (linear approximation)


N_iter = 100              # Maximum iteration for FPI
weight = 0.1              # Relaxation factor for FPI
TOL_chi_A = 1e-3          # Convergence criterion before meet n == N_iter
Nz = 100                  # Z-discritization number
Nr = 200                  # Bulk R-discritization number (note that the boundary-layer R-disc. has a different number)
N_PROCESSES = 4           # Number of processes for multiprocessing


eta0 = 1e-3               # Pa*sec     : Solvent viscosity
T = 293.15                # K          : Temperature

# Dispersion properties
phi_bulk = 1e-3   # Inlet volume fraction of particle
phi_freeze = 0.494
a_particle = 3.13e-9 # m : Hard-core radius of particle
gamma = 1.0       # Solvent-permeability to hard spheres (a_H/a_particle)

# Transport properties
from osmotic_pressure_CS import *
fcn_Pi_given = Pi_CS

from transport_properties_SPHS import *

fcn_eta_given = eta_div_eta0_SPHS
fcn_Dc_given = Dc_short_div_D0_SPHS

# Initial conditions
phiw_arr = ones(Nz)*phi_bulk 


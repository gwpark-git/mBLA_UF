#############################################################################
#   Particle-contributed osmotic pressure using Carnahan-Starling equation  #
#                                                                           #
#   Used in the paper:                                                      #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   doi: 10.1063/5.0020986                                                  #
#                                                                           #
#   Used in the paper (to be submitted):                                    #
#   (tentative title) Geometrical influence on particle transport in        #
#   cross-flow ultrafiltration: cylindrical and flat sheet membranes        #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   doi: TBD                                                                #
#                                                                           #
#                                                                           #
#   Code Developer: Park, Gun Woo    (g.park@fz-juelich.de)                 #
#   MIT Open License (see LICENSE file in the main directory)               #
#                                                                           #
#   Update (June 2021):                                                     #
#   The original code only applicable for the hollow fiber                  #
#   New version support for the channel between two flat sheets:            #
#   1. FMM: channel flow between flat membrane (top) / membrane (bottom)    #
#   2. FMS: channel flow between flat membrane (top) / substrate (bottom)   #
#   For this reason, the hollow fiber expression will be renamed as HF      #
#                                                                           #
#   Important note:                                                         #
#   The new updte is based on the coordination y (in the new manuscript)    #
#   This is exactly the same treatment with r in the code                   #
#############################################################################

from numpy import *

# calculating osmotic pressure
import scipy.constants as const

def fcn_zero(phi, cond_GT):
    re = zeros(size(phi))
    return re

def Pi_CS(phi, cond_GT):
    return get_Pi(phi, cond_GT['Va'], cond_GT['kT'])

def Pi_VH(phi, cond_GT):
    rho = phi/cond_GT['Va']
    kBT = cond_GT['kT']
    return rho * kBT * 1.0

def CS_Z(phi):
    return (1 + phi + phi**2. - phi**3.)/(1.-phi)**3.0

def get_Pi(phi, V_a, kBT):
    rho = phi / V_a # converting volume fraction phi into number density n
    return rho*kBT*CS_Z(phi)

def CS_S0(phi):
    # using CS_Z
    return (1. - phi)**4.0 / ((1. + 2.*phi)**2.0 + phi**3.0 *(phi - 4.))

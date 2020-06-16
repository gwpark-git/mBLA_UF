#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


from numpy import *

# calculating osmotic pressure
import scipy.constants as const

def fcn_zero(phi, cond_GT):
    re = zeros(size(phi))
    return re

def Pi_CS(phi, cond_GT):
    return get_Pi(phi, cond_GT['Va'], cond_GT['kT'])


def CS_Z(phi):
    return (1 + phi + phi**2. - phi**3.)/(1.-phi)**3.0

def get_Pi(phi, V_a, kBT):
    rho = phi / V_a # converting volume fraction phi into number density n
    return rho*kBT*CS_Z(phi)

def CS_S0(phi):
    # using CS_Z
    return (1. - phi)**4.0 / ((1. + 2.*phi)**2.0 + phi**3.0 *(phi - 4.))

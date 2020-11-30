#############################################################################
#   Transport properties of solvent-permeable hard spheres                  #
#   The basic theory is based on Riest et al. Soft Matter (2015)            #
#   and the code Used in the paper:                                         #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   doi: 10.1063/5.0020986                                                  #
#   Code Developer: Park, Gun Woo    (g.park@fz-juelich.de)                 #
#   MIT Open License (see LICENSE file in the main directory)               #
#############################################################################


from numpy import *

# two main functions

def fcn_unity(phi, cond_GT):
    re = ones(size(phi))
    return re

def eta_div_eta0_SPHS(phi, cond_GT):
    gamma = cond_GT['gamma']
    return eta_inf_div_eta0_SPHS(phi, gamma)*(1 + (1/Gamma_S_SPHS(phi, gamma))*Del_eta_noHI_div_eta0_SPHS(phi))

def eta_div_eta0_SPHS_2(phi, gamma):
    return eta_inf_div_eta0_SPHS(phi, gamma)*(1 + (1/Gamma_S_SPHS(phi, gamma))*Del_eta_noHI_div_eta0_SPHS(phi))


def Dc_short_div_D0_SPHS(phi, cond_GT):
    gamma = cond_GT['gamma']
    return K_SPHS(phi, gamma)/S0_CS(phi)

def Dc_short_div_D0_SPHS_2(phi, gamma):
    return K_SPHS(phi, gamma)/S0_CS(phi)

# Auxilary functions

def Ds_div_D0_SPHS(phi, gamma):
    return 1 + lambda_t_SPHS(gamma)*phi*(1 + 0.12*phi - 0.70*phi**2)

def lambda_t_SPHS(gamma):
    return -1.8315 + 7.820 *(1-gamma) - 14.231 *(1-gamma)**2 + 14.908* (1-gamma)**3 - 9.383* (1-gamma)**4 + 2.717* (1-gamma)**5




def eta_inf_div_eta0_SPHS(phi, gamma):
    return 1 + 2.5*gamma**3*phi*(1 + Saito_fcn_SPHS(phi, gamma))*(1 - gamma**3 * phi * (1 + Saito_fcn_SPHS(phi, gamma)))


def Saito_fcn_SPHS(phi, gamma):
    return phi*(lambda_V_SPHS(gamma)/(2.5*gamma**3) - gamma**3)


def lambda_V_SPHS(gamma):
    return 5.0021 - 39.279* (1-gamma) + 143.179 *(1-gamma)**2 - 288.202* (1-gamma)**3 + 254.581 *(1- gamma)**4


def Del_eta_noHI_div_eta0_SPHS(phi):
    # (Note: phi_RCP ~ 0.64 is already applied)    
    return (12/5.)*phi**2*(1 - 7.085*phi + 20.182*phi**2)/(1 - phi/0.64)
    

def Gamma_S_SPHS(phi, gamma):
    return Ds_div_D0_SPHS(phi, gamma)*eta_inf_div_eta0_SPHS(phi, gamma)




def K_SPHS(phi, gamma):
    gp = gamma*phi
    return 1 + lambda_K_SPHS(gamma)*phi*(1 - 3.348*(gp) + 7.426*(gp)**2 - 10.034*(gp)**3 + 5.882*(gp)**4)


def lambda_K_SPHS(gamma):
    return -6.5464 + 8.592*(1-gamma) - 3.901*(1-gamma)**2 + 2.011*(1-gamma)**3 - 0.142*(1-gamma)**4


def S0_CS(phi):
    return (1-phi)**4/((1+2*phi)**2 + phi**3*(phi - 4))

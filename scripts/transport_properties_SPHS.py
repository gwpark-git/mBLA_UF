#############################################################################
#   Transport properties of solvent-permeable hard spheres                  #
#   The basic theory is based on Riest et al. Soft Matter (2015)            #
#   and simulation data from Abade et al. (2012)                            #
#                                                                           #
#   Used in the paper:                                                      #
#   [1] Park and N{\"a}gele, JCP, 2020                                      #
#   doi: 10.1063/5.0020986                                                  #
#                                                                           #
#   [2] Park and N{\"a}gele, Membranes, 2021                                #
#   doi: https://doi.org/10.3390/membranes11120960                          #
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

def fcn_unity(phi, cond_GT):
    re = ones(size(phi))
    return re

    
def eta_inf_div_eta0_HS_S3(phi, gamma):
    """ High-frequency viscosity reduced by solvent (or dilution-limit) viscosity using the third order Saito function on phi.
    The expression is already reported by Riest et al. Soft Matter (2015) although their actual use with solvent-permeable hard sphere is based on the linear fit with phi (see Saito_fcn_SPHS function).
    """
    s = Saito_fcn_HS_S3(phi)
    return 1 + (5./2.)*phi*(1.+s)/(1. - phi*(1.+s))

def Gamma_S_HS_S3(phi, gamma):
    """ Generalized Stokes-Einstein function for self-diffusion coefficient of hard spheres
    """
    return Ds_div_D0_SPHS(phi, gamma)*eta_inf_div_eta0_HS_S3(phi, gamma)

def eta_div_eta0_HS_S3(phi, cond_GT):
    """ High-frequency viscosity reduced by solvent (or dilution-limit) viscosity using the first order Saito function on phi.
    The main reference is Riest et al. Soft Matter (2015).
    It is based on the "first-order" polynomial fit for permeable hard spheres based on the simulation result reported by Abade et al. JCP (2012).
    If we consider solvent-impermeable hard sphere case (i.e., gamma=1), the better polynomial fit is avaliable.
    For such a normal hard sphere case, see Saito_fcn_HS_S3 function.
    
    Remark: In the case of a typical (solvent-impermeable) hard sphere dispersions, we would recommend the function "Saito_fcn_HS_S3".
    """
    gamma = cond_GT['gamma']
    return eta_inf_div_eta0_HS_S3(phi, gamma)*(1 + (1/Gamma_S_HS_S3(phi, gamma))*Del_eta_noHI_div_eta0_SPHS(phi))


# Below are the originally used functions
def eta_div_eta0_SPHS(phi, cond_GT):
    gamma = cond_GT['gamma']
    return eta_inf_div_eta0_SPHS(phi, gamma)*(1 + (1/Gamma_S_SPHS(phi, gamma))*Del_eta_noHI_div_eta0_SPHS(phi))

def eta_div_eta0_SPHS_2(phi, gamma):
    return eta_inf_div_eta0_SPHS(phi, gamma)*(1 + (1/Gamma_S_SPHS(phi, gamma))*Del_eta_noHI_div_eta0_SPHS(phi))


def Dc_short_div_D0_SPHS(phi, cond_GT):
    """ Short-time collective diffusion coefficient of solvent-permeable hard spheres.

    Arguments:
        phi: volume fraction of particles
        cond_GT: dictionary for basic condition. Here, the only necessary key is cond_GT['gamma']

    Return:
        Dc/D0: Reduced short-time collective diffusion coefficient (reducing factor is the Stokes-Einstein self-diffusion coefficient)
    """
    gamma = cond_GT['gamma']
    return K_SPHS(phi, gamma)/S0_CS(phi)

def Dc_short_div_D0_SPHS_2(phi, gamma):
    """ The same as Dc_short_div_D0_SPHS except the arguments.
    """
    return K_SPHS(phi, gamma)/S0_CS(phi)

# Auxilary functions

def Ds_div_D0_SPHS(phi, gamma):
    """ Self-diffusion coefficient of solvent-permeable hard spheres.
    """
    return 1 + lambda_t_SPHS(gamma)*phi*(1 + 0.12*phi - 0.70*phi**2)

def lambda_t_SPHS(gamma):
    return -1.8315 + 7.820 *(1-gamma) - 14.231 *(1-gamma)**2 + 14.908* (1-gamma)**3 - 9.383* (1-gamma)**4 + 2.717* (1-gamma)**5




def eta_inf_div_eta0_SPHS(phi, gamma):
    return 1 + 2.5*gamma**3*phi*(1 + Saito_fcn_SPHS(phi, gamma))*(1 - gamma**3 * phi * (1 + Saito_fcn_SPHS(phi, gamma)))


# Here, the Saito function take up to the third order
def Saito_fcn_HS_S3(phi):
    """ Saito-type function: Third-order polynomial on phi (S3)
    The expression is already reported by Riest et al. Soft Matter (2015) although their actual use with solvent-permeable hard sphere is based on the linear fit with phi (see Saito_fcn_SPHS function).
    """
    return phi*(1. + 0.95*phi - 2.15*phi**2.0)
    # return phi*(1. + 0.95*phi - 2.15*phi**2.0)*(lambda_V_SPHS(gamma)/(2.5*gamma**3) - gamma**3)
    # return phi*(lambda_V_SPHS(gamma)/(2.5*gamma**3) - gamma**3)


def Saito_fcn_SPHS(phi, gamma):
    """ Saito-type function: First-order polynomial on phi (S3)
    The main reference is Riest et al. Soft Matter (2015).
    It is based on the "first-order" polynomial fit for permeable hard spheres based on the simulation result reported by Abade et al. JCP (2012).
    If we consider solvent-impermeable hard sphere case (i.e., gamma=1), the better polynomial fit is avaliable.
    For such a normal hard sphere case, see Saito_fcn_HS_S3 function.
    
    Remark: In the case of a typical (solvent-impermeable) hard sphere dispersions, we would recommend the function "Saito_fcn_HS_S3".
    """
    return phi*(lambda_V_SPHS(gamma)/(2.5*gamma**3) - gamma**3)


def lambda_V_SPHS(gamma):
    return 5.0021 - 39.279* (1-gamma) + 143.179 *(1-gamma)**2 - 288.202* (1-gamma)**3 + 254.581 *(1- gamma)**4


def Del_eta_noHI_div_eta0_SPHS(phi):
    # (Note: phi_RCP ~ 0.64 is already applied)    
    return (12/5.)*phi**2*(1 - 7.085*phi + 20.182*phi**2)/(1 - phi/0.64)
    

def Gamma_S_SPHS(phi, gamma):
    """ Generalized Stokes-Einstein function for self-diffusion coefficient of solvent-permeable hard spheres
    """
    return Ds_div_D0_SPHS(phi, gamma)*eta_inf_div_eta0_SPHS(phi, gamma)



def K_SPHS(phi, gamma):
    """ Sedimentation coefficient of solvent-permeable hard spheres
    """
    gp = gamma*phi
    return 1 + lambda_K_SPHS(gamma)*phi*(1 - 3.348*(gp) + 7.426*(gp)**2 - 10.034*(gp)**3 + 5.882*(gp)**4)


def lambda_K_SPHS(gamma):
    return -6.5464 + 8.592*(1-gamma) - 3.901*(1-gamma)**2 + 2.011*(1-gamma)**3 - 0.142*(1-gamma)**4


def S0_CS(phi):
    """ Compressibility factor using Carnahan-Starling equation
    """
    return (1-phi)**4/((1+2*phi)**2 + phi**3*(phi - 4))

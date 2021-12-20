#############################################################################
#   Auxililary functions related to the membrane geometry for mBLA_UF code  #
#                                                                           #
#   Used in the paper:                                                      #
#   [1] Park and N{\"a}gele, JCP, 2020                                      #
#   doi: 10.1063/5.0020986                                                  #
#                                                                           #
#   [2] Park and N{\"a}gele, Membranes, 2021                                #
#   doi: https://doi.org/10.3390/membranes11120960                          #
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
#   1. Coordination system "y" is now from bottom to top (FMM/FMS)          #
#   The corresponding "y" in CM is the same as "r" in the previous paper    #
#   The new updte is based on the coordination y (in the new manuscript)    #
#   This is exactly the same treatment with r in the code                   #
#                                                                           #
#   2. The definition of lam1 and lam2 is different from paper [2]          #
#   This is because manuscript used hydraulic radius Rh together with R     #
#   uses only R. Either ways are fine, just do not confuse lambda-parameters#
#############################################################################


from numpy import *

def get_add_term_cal_Fz(uZ_nodim, membrane_geometry):
    """ This is a proper adoption of particle-flux conservation law on mBLA method.
    The actual use of it is cal_Fz, which is related to the under-relaxed fixed-point iteration.
    """
    if membrane_geometry=='FMS':
        return (2/3.)*(1. - uZ_nodim)
    return 0.

def get_DLP_from_uin(u_ast, lam1, eta_s, R_channel, L_channel):
    """ this will calculate DLP_ast based on the linear approximation for P
    """
    DLP_ast = u_ast * lam1 * eta_s * L_channel/R_channel**2.0;
    return DLP_ast

def get_Lp_from_kappa_Darcy(membrane_geometry, kappa_Darcy, h_membrane, R_channel, eta0):
    """ Calculate hydraulic membrane permeability from given mean Darcy permeability kappa using Eq. (14) in [2].
    The default is the hollow fiber case ( membrane_geometry = 'HF' )
    """
    
    Lp = kappa_Darcy / (eta0 * R_channel * log(1. + h_membrane/R_channel))
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        Lp = kappa_Darcy / (eta0 * h_membrane)
    return Lp

def get_kappa_Darcy_from_Lp(membrane_geometry, Lp, h_membrane, R_channel, eta0):
    """ Calculating expected mean Darcy permeability kappa using the provided hydraulic membrane permeability using Eq. (14) in [2].
    """
    kappa_Darcy = Lp * eta0 * R_channel * log(1. + h_membrane/R_channel)
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        kappa_Darcy = Lp * eta0 * h_membrane
    return kappa_Darcy

def get_effective_permeability_parameter_K(lam1, lam2, R_channel, L_channel, Lp, eta0):
    """ Calculating the effective permeability parameter (K) defined in Eq. (35) in [2].
    Note that for the hollow fiber case, this expression is the same as k in Eq. (26) in [1].
    """
    return sqrt(lam1*lam2)*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)

def get_lam1(membrane_geometry):
    """ dimensionless geometry coefficient lambda1
    The corresponding definition (of different separability factor) is provided in Table 1 in [2].
    Note that the definition in [2] is slightly different due to the use of hydraulic radius of channel.
    This means that the lam1 in this codde is 4, 2, and 2 respectively for CM, FMM, and FMS.
    However, the lambda1 in reference [2] is 1, 2, and 2 respectively for CM, FMM, and FMS (divided by (RH/R)^2 from lam1 here)
    """
    # the default value will be the case of 'HF'
    lam1 = 4.
    if (membrane_geometry=='FMM'):
        lam1 = 2.
    elif (membrane_geometry=='FMS'):
        lam1 = 2.
    return lam1

def get_lam2(membrane_geometry):
    """ dimensionless geometry coefficient lambda2
    The corresponding definition (of different separability factor) is provided in Table 1 in [2].
    Note that the definition in [2] is slightly different due to the use of hydraulic radius of channel.
    This means that the lam2 in this code is 4, 3/2, and 3/4 respectively for CM, FMM, and FMS.
    However, the lambda1 in reference [2] is 2, 3/2, and 3/4 respectively for CM, FMM, and FMS (divided by RH/R from lam2 here)
    """
    
    # the default value will be the case of 'HF'
    lam2 = 4.
    if (membrane_geometry=='FMM'):
        lam2 = 3/2.
    elif (membrane_geometry=='FMS'):
        lam2 = 3/4.
    return lam2

def get_Uout_mean(membrane_geometry):
    """ dimensionless cross-sectional averaged Uout = 1 - (y/R)^2 (following Hagen-Poiseiulle-type flow profile)
    The cross-sectional average is taking as Eq. (22) in [2].
    """
    Uout_mean = 1/2.
    if (membrane_geometry=='FMM'):
        Uout_mean = 2/3.
    elif (membrane_geometry=='FMS'):
        Uout_mean = 2/3.
    
    return Uout_mean


def J_int_yt(yt, membrane_geometry):
    """ Jacobian using yt = 1. - rt coordination
    Note that the coordination definition for y is not consist between [1] and [2].
    """
    J = 1. - yt
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        J = 1.
    return J

def J_int_rt(rt, membrane_geometry):
    """ Jacobian using rt coordination
    Note that the coordination definition for y is not consist between [1] and [2].
    """
    J = rt
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        J = 1.
    return J


def fcn_VR_FMM(r_div_R):
    """ Transversal velocity factor of FMM in Eq. (31) in [2]
    """
    return (1/2.)*(3.*r_div_R - r_div_R**3.0)

def fcn_VR_FMS(r_div_R):
    """ Transversal velocity factor of FMS in Eq. (31) in [2]
    """
    return (1/4.)*(3.*r_div_R - r_div_R**3.0 + 2.0)

def fcn_VR_HF(r_div_R):
    """ Transversal velocity factor of CM in Eq. (31) in [2]
    """
    return 2.*r_div_R - r_div_R**3.0

def fcn_VR(r_div_R, membrane_geometry):
    """ Transversal velocity factor depends on membrane_geometry using expressions in Eq. (31) in [2]
    """
    if (membrane_geometry=='FMM'):
        return fcn_VR_FMM(r_div_R)
    elif (membrane_geometry=='FMS'):
        return fcn_VR_FMS(r_div_R)
    # when 'HF'
    return fcn_VR_HF(r_div_R)


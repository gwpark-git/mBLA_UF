#############################################################################
#   Auxililary functions related to the membrane geometry for mBLA_UF code  #
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

def get_add_term_cal_Fz(uZ_nodim, membrane_geometry):
    if membrane_geometry=='FMS':
        return (2/3.)*(1. - uZ_nodim)
    return 0.

def get_DLP_from_uin(u_ast, lam1, eta_s, R_channel, L_channel):
    # this will calculate DLP_ast based on the linear approximation for P
    # i.e., a constant approximation for u.
    
    DLP_ast = u_ast * lam1 * eta_s * L_channel/R_channel**2.0;
    return DLP_ast

def get_Lp_from_kappa_Darcy(membrane_geometry, kappa_Darcy, h_membrane, R_channel, eta0):
    # the default is when membrane_geometry = 'HF'
    Lp = kappa_Darcy / (eta0 * R_channel * log(1. + h_membrane/R_channel))
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        Lp = kappa_Darcy / (eta0 * h_membrane)
    return Lp

def get_kappa_Darcy_from_Lp(membrane_geometry, Lp, h_membrane, R_channel, eta0):
    # the default is when membrane_geometry = 'HF'
    kappa_Darcy = Lp * eta0 * R_channel * log(1. + h_membrane/R_channel)
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        kappa_Darcy = Lp * eta0 * h_membrane
    return kappa_Darcy

def get_effective_permeability_parameter_K(lam1, lam2, R_channel, L_channel, Lp, eta0):
    return sqrt(lam1*lam2)*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)

def get_lam1(membrane_geometry):
    # the default value will be the case of 'HF'
    lam1 = 4.
    if (membrane_geometry=='FMM'):
        lam1 = 2.
    elif (membrane_geometry=='FMS'):
        lam1 = 2.
    return lam1

def get_lam2(membrane_geometry):
    # the default value will be the case of 'HF'
    lam2 = 4.
    if (membrane_geometry=='FMM'):
        lam2 = 3/2.
    elif (membrane_geometry=='FMS'):
        lam2 = 3/4.
    return lam2


# def get_K(lam1, lam2, R_channel, L_channel, Lp, eta0):
#     # the default value will be the case of 'HF'
#     k = lam1*lam2*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)        
#     return k

def J_int_yt(yt, membrane_geometry):
    # this is integration Jacobian using yt = 1. - rt coordination
    # note that the coordination definition is not consist with the new manuscript.
    J = 1. - yt
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        J = 1.
    return J

def J_int_rt(rt, membrane_geometry):
    # this is integration Jacobian using rt coordination
    # note that the coordination definition is not consist with the new manuscript.
    J = rt
    if (membrane_geometry=='FMM' or membrane_geometry=='FMS'):
        J = 1.
    return J


def fcn_VR_FMM(r_div_R):
    return (1/2.)*(3.*r_div_R - r_div_R**3.0)

def fcn_VR_FMS(r_div_R):
    return (1/4.)*(3.*r_div_R - r_div_R**3.0 + 2.0)

def fcn_VR_HF(r_div_R):
    return 2.*r_div_R - r_div_R**3.0

def fcn_VR(r_div_R, membrane_geometry):
    if (membrane_geometry=='FMM'):
        return fcn_VR_FMM(r_div_R)
    elif (membrane_geometry=='FMS'):
        return fcn_VR_FMS(r_div_R)
    # when 'HF'
    return fcn_VR_HF(r_div_R)

from numpy import *

def get_lam1(membrane_geometry):
    # the default value will be the case of 'HF'
    lam1 = 2.
    if (membrane_geometry=='FMM'):
        lam1 = 2.
    elif (membrane_geometry=='FMS'):
        lam1 = 2.
    return lam1

def get_lam2(membrane_geometry):
    # the default value will be the case of 'HF'
    lam2 = 2.
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

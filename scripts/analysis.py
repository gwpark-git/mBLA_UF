#############################################################################
#   Analysis codes for cal_phiw_from_input.py                               #
#                                                                           #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################

# from sol_CT import *
# from sol_GT import *
# from sol_GT_parallel import *

import sol_solvent as PS
import sol_CT as CT
import sol_GT as GT


# from osmotic_pressure_CS import *
# from transport_properties_SPHS import *
import osmotic_pressure_CS as CS
import transport_properties_SPHS as PHS

import sys
from numpy import *
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.linalg import norm
from copy import deepcopy

def get_psi(s_bar, phi_b, phi_w): 
    """Return psi 
        psi = particle distribution function along y in Eq. (55)
            : get_psi(s_bar(y_bar, z), phi_b, phi_w(z)) = psi(y_bar, z)

    Parameters:
        s_bar = important exponent described in Eq. (43)
        phi_b = particle volume fraction in bulk/feed 
        phi_w = particle wall volume fraction 

    """
    return exp(-s_bar) - (phi_b/(phi_w - phi_b))*(s_bar*exp(-s_bar))

# def get_delta_z(yt_arr, vw_div_vw0, epsilon_d, INT_Dc, phi_b, phi_w): # not fully supported yet
#     Ny = size(yt_arr)
#     psi_2 = 1.0   # it must be unity at the wall
#     numer_psi = 0.    
#     denom_psi = 0.
#     for i in range(1, Ny):
#         dy = yt_arr[i] - yt_arr[i-1]
#         s_bar = (vw_div_vw0/epsilon_d)*INT_Dc[i]
#         psi_1 = psi_2
#         psi_2 = get_psi(s_bar, phi_b, phi_w)

#         f1 = psi_1 * yt_arr[i-1] * (1. - yt_arr[i-1])
#         f2 = psi_2 * yt_arr[i] * (1. - yt_arr[i])
#         numer_psi += 0.5 * dy * (f1 + f2)
        
#         f1_denom = psi_1 * (1. - yt_arr[i-1])
#         f2_denom = psi_2 * (1. - yt_arr[i])
#         denom_psi += 0.5 * dy * (f1_denom + f2_denom)
#     return numer_psi / denom_psi
    

def gen_analysis(z_arr, yt_arr, phiw_arr, cond_GT, fcn_Pi, fcn_Dc_given, fcn_eta_given, fn_out): 
    """ Return 0
    
    Save analysis data into file with name of fn_out    
    Data first stored in array "re":
        re[0] = z                in the unit of m
        re[1] = phi_w(z)         in the dimensionless unit
        re[2] = P(z)             in the unit of Pa
        re[3] = v_w(z)           in the unit of m/sec
        re[4] = u(r=0, z)        in the unit of m/sec
        re[5] = Pi(phi_w(z))     in the unit of Pa
        re[6] = P(z) - P_perm    in the unit of Pa
        re[7] = v_w(z)/v^\ast    in the dimensionless unit
        re[8] = u(r=0, z)/u^\ast in the dimensionless unit
        re[9] = Phi(z)           in the unit of m^3/sec
    
    Parameters:
        z_arr              = arrays for discretized z (m)
        yt_arr             = arrays for discretized y (dimensionless).
                             works as auxiliary function to calculate some functions <- check
        phiw_arr(z)        = arrays for particle volume fraction at the wall
        cond_GT            = conditions for general transport properties
        fcn_Pi(phi)        = given function for the osmotic pressure
        fcn_Dc_given(phi)  = given function for gradient diffusion coefficient
        fcn_eta_given(phi) = given function for suspension viscosity
        fn_out             = filename for data output

    """
    Nz = size(z_arr); Ny = size(yt_arr)
    dz = z_arr[1] - z_arr[0]
    re = zeros([Nz, 11])
    re[:,0] = z_arr
    re[:,1] = phiw_arr
    Pi_arr = fcn_Pi(re[:,1], cond_GT)
    gp_arr = zeros(Nz)
    gm_arr = zeros(Nz)
    for i in range(Nz):
        # gp_arr[i] = get_gpm(z_arr[i], dz, +1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
        # gm_arr[i] = get_gpm(z_arr[i], dz, -1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
        CT.gen_gpm_arr(+1.0, z_div_L_arr, dz_div_L, Pi_div_DLP_arr, k, gp_arr)
        CT.gen_gpm_arr(-1.0, z_div_L_arr, dz_div_L, Pi_div_DLP_arr, k, gm_arr)
        
        phi_arr = zeros(Ny)
        Ieta_arr = zeros(Ny)
        ID_arr = zeros(Ny)
        u0 = cond_GT['R']**2.0 * (cond_GT['Pin'] - cond_GT['Pout'])/(4. * cond_GT['eta0'] * cond_GT['L'])
    for i in range(Nz):
        z_tmp = z_arr[i]
        re[i, 2] = get_P_CT_boost(0., z_tmp, cond_GT, cond_GT['dz'], Pi_arr, gp_arr[i], gm_arr[i])
        re[i, 3] = get_vw_GT_boost(z_tmp, cond_GT, Pi_arr, gp_arr[i], gm_arr[i])
        vw_div_vw0 = re[i, 3]/cond_GT['vw0']
        get_phi_with_fixed_z_GT(z_tmp, cond_GT, re[i,1], Pi_arr, fcn_Dc_given, vw_div_vw0, yt_arr, phi_arr)
        get_int_eta_phi(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, yt_arr, phi_arr, Ieta_arr)
        get_int_D_phi(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, yt_arr, phi_arr, ID_arr)
        INT_Ieta = interp1d(yt_arr, Ieta_arr)
        re[i, 4] = cond_GT['u_HP']*get_u_center_GT_boost(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, Ieta_arr[-1], gp_arr[i], gm_arr[i])
        re[i, 5] = Pi_arr[i]

        re[i, 6] = re[i, 2] - cond_GT['Pper']
        re[i, 7] = re[i, 3]/cond_GT['vw0']
        re[i, 8] = re[i, 4]/cond_GT['u_HP']

        Phi_z = 0.
        u1 = 0; u2 = 0;
        for j in range(1, Ny): # cal: cross-sectional averaged particle flux
            dy = yt_arr[j] - yt_arr[j-1]
            u1 = u2
            u2 = get_u_GT_boost(cond_GT['R']*(1. - yt_arr[j]), z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, INT_Ieta, gp_arr[i], gm_arr[i])/u0
            Phi_z += 0.5 * dy * (phi_arr[j] * u2 * (1. - yt_arr[j]) + phi_arr[j-1]*u1*(1. - yt_arr[j-1]))
            Phi_z *= 2. * pi * u0 * cond_GT['R']**2.0
            re[i, 9] = Phi_z
            # if i != 0:
            #     re[i, 10] = get_delta_z(yt_arr, vw_div_vw0, cond_GT['epsilon_d'], ID_arr, cond_GT['phi_bulk'], re[i, 1])
    savetxt(fn_out, re) # write the result into the filename described in fn_out
    return 0


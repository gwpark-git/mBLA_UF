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

from aux_functions import *

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

from datetime import datetime

def print_preface(fn_inp, fn_out, fn_out_log, f_log):
    now_str = datetime.now().strftime("%H:%M (%d/%m/%Y)")

    print(fn_out_log)
    print ('##############################################################')
    print ('# Semi-analytic solution of mBLA UF using CP layer model     #')        
    print ('# git-repository: https://github.com/gwpark-git/mBLA_UF.git  #')
    print ('# Developer: Gunwoo Park (IBI-4, Forschungszentrum Juelich)  #')
    print ('# Reference: Park and Naegele, JCP (accepted, 2020)          #')
    print ('##############################################################')
    print ('')
    print ('Executed time (date): %s'%(now_str))
    print ('with Arguments: ', fn_inp, fn_out)
    print ('     Log will be stored in ', fn_out_log)

    f_log.write('\n##############################################################\n')
    f_log.write('# Semi-analytic solution of mBLA UF using CP layer model     #\n')        
    f_log.write('# git-repository: https://github.com/gwpark-git/mBLA_UF.git  #\n')
    f_log.write('# Developer: Gunwoo Park (IBI-4, Forschungszentrum Juelich)  #\n')
    f_log.write('# Reference: Park and Naegele, JCP (accepted, 2020)          #\n')
    f_log.write('##############################################################\n\n')
    f_log.write('Executed time (date): %s\n'%(now_str))
    f_log.write('with Arguments: %s, %s\n'%(fn_inp, fn_out))
    f_log.write('     Log will be stored in %s\n\n'%(fn_out_log))
    return now_str
    

def print_summary(cond_GT, f_log=None):
    print ('\nSystem and operating conditions:' )
    print ('  - Summary of dimensional quantities (in SI units):')
    print ('\tPin=%9d,   Pper=%9d, ref_Pout=%9d '%(int(cond_GT['Pin']), int(cond_GT['Pper']), int(cond_GT['Pout'])))
    print ('\tDLP=%9d, DTP_PS=%9d,   DTP_HP=%9d '%(int(cond_GT['DLP']), int(cond_GT['DTP_PS']), int(cond_GT['DTP_HP'])))
    print ('\tLp =%4.3e,   eta0=%4.3e,        R=%4.3e,        L=%4.3e'%(cond_GT['Lp'], cond_GT['eta0'], cond_GT['R'], cond_GT['L']))
    print ('\ta  =%4.3e,    a_H=%4.3e,       D0=%4.3e,  Phi_ast=%4.3e'%(cond_GT['a'], cond_GT['a_H'], cond_GT['D0'], cond_GT['Phi_ast']))

    print ('  - Corresponding dimensionless quantities:')
    print ('\tk=%.4f, alpha_ast=%.4f, beta_ast=%.4f, gamma=a_H/a=%4.3e'%(cond_GT['k'], cond_GT['alpha_ast'], cond_GT['beta_ast'], cond_GT['gamma']))
    print ('\tepsilon=%4.3e, epsilon_d=%4.3e (Pe_R=%.1f)'%(cond_GT['R']/cond_GT['L'], cond_GT['epsilon_d'], 1./cond_GT['epsilon_d']))

    if not f_log.closed:
        f_log.write('\nSystem and operating conditions:\n' )
        f_log.write('  - Summary of dimensional quantities (in SI units):\n')
        f_log.write('\tPin=%9d,   Pper=%9d, ref_Pout=%9d \n'%(int(cond_GT['Pin']), int(cond_GT['Pper']), int(cond_GT['Pout'])))
        f_log.write('\tDLP=%9d, DTP_PS=%9d,   DTP_HP=%9d \n'%(int(cond_GT['DLP']), int(cond_GT['DTP_PS']), int(cond_GT['DTP_HP'])))
        f_log.write('\tLp =%4.3e,   eta0=%4.3e,        R=%4.3e,  L=%4.3e\n'%(cond_GT['Lp'], cond_GT['eta0'], cond_GT['R'], cond_GT['L']))
        f_log.write('\ta  =%4.3e,    a_H=%4.3e,       D0=%4.3e\n'%(cond_GT['a'], cond_GT['a_H'], cond_GT['D0']))
        f_log.write('  - Corresponding dimensionless quantities:\n')
        f_log.write('\tk=%.4f, alpha_ast=%.4f, beta_ast=%.4f\n'%(cond_GT['k'], cond_GT['alpha_ast'], cond_GT['beta_ast']))
        f_log.write('\tepsilon=%4.3e, epsilon_d=%4.3e (Pe_R=%.1f)\n\n'%(cond_GT['R']/cond_GT['L'], cond_GT['epsilon_d'], 1./cond_GT['epsilon_d']))
        
    return 0

def print_iteration_info(n, z_div_L_arr, phiw_set_1, phiw_set_2, cond_GT, Pi_div_DLP_arr, gp_arr, gm_arr, f_log=None):
    # this part is for recording the analysis part
    Nz = size(z_div_L_arr)
    
    report_step = zeros(12)
    report_step[0] = n


    ind_max_z = argmax(phiw_set_2)
    chi_A = norm(phiw_set_1 - phiw_set_2)/float(Nz)                                           # estimated deviations
    report_step[1] = z_div_L_arr[ind_max_z]*cond_GT['L']
    report_step[2] = phiw_set_2[ind_max_z]*cond_GT['phi_bulk']
    report_step[3] = phiw_set_2[-1]*cond_GT['phi_bulk']

    r0_div_R = 0.; L_div_L = 1.
    dz_div_L = cond_GT['dz']/cond_GT['L']

    report_step[4] = length_average_f(z_div_L_arr, Pi_div_DLP_arr, L_div_L, dz_div_L)*cond_GT['DLP']/cond_GT['DTP_HP']

    report_P_div_DLP_arr = zeros(Nz)
    for i in range(Nz):
        report_P_div_DLP_arr[i] = GT.get_P_conv(r0_div_R, z_div_L_arr[i], cond_GT, gp_arr[i], gm_arr[i])

    report_step[5] = length_average_f(z_div_L_arr, report_P_div_DLP_arr - cond_GT['Pper_div_DLP'], L_div_L, dz_div_L)*cond_GT['DLP']/cond_GT['DTP_HP']
    print('iter=%d, chi_A=%4.3e'%(report_step[0], chi_A))
    print('\tz_max=%4.3f, phiw(z_max)=%.4f, phiw(L)=%.4f\n\t<Pi>/DTP_HP=%4.3e, DTP/DTP_HP=%4.3e'%(report_step[1], report_step[2], report_step[3], report_step[4], report_step[5]))
    print()
    if not f_log.closed:
        if(n==0):
            f_log.write('Columns of output files are \n')
            f_log.write('\t[0] n            : number of iteration\n')
            f_log.write('\t[1] z_max        : peak-position (z-coordinate) of phi_w\n')
            f_log.write('\t[2] phi_w(z_max) : peak-value of phi_w\n')
            f_log.write('\t[3] phi_w(L)     : phi_w at the outlet z=L\n')
            f_log.write('\t[4] <Pi>/DTP_HP  : (dimensionless) length-averaged osmotic pressure\n')
            f_log.write('\t[5] DTP/DTP_HP   : (dimensionless) length-averaged transmembrane pressure\n')
            f_log.write('\t[6] chi_A        : Absolute deviation using norm(phiw_1/phi_b - phiw_2/phi_b)/Nz\n\n')
            f_log.write('Calculating... \n')
        f_log.write('%d\t%e\t%e\t%e\t%e\t%e\t%e\n'%(report_step[0], report_step[1], report_step[2], report_step[3], report_step[4], report_step[5], chi_A))
    return chi_A



def gen_analysis(z_arr, y_div_R_arr, phiw_arr, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out): 
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
        re[10] = 0               (empty at this moment)
    
    Parameters:
        z_arr              = arrays for discretized z (m)
        y_div_R_arr        = arrays for discretized y (dimensionless).
                             works as auxiliary function to calculate some functions <- check
        phiw_arr(z)        = arrays for particle volume fraction at the wall
        cond_GT            = conditions for general transport properties
        fcn_Pi(phi)        = given function for the osmotic pressure
        fcn_Dc_given(phi)  = given function for gradient diffusion coefficient
        fcn_eta_given(phi) = given function for suspension viscosity
        fn_out             = filename for data output

    """

    Nz = size(z_arr); Ny = size(y_div_R_arr)
    dz = z_arr[1] - z_arr[0]
    dz_div_L = dz/cond_GT['L']
    z_div_L_arr = z_arr/cond_GT['L']
    
    sign_plus = +1.
    sign_minus = -1.

    re = zeros([Nz, 11])

    re[:, 0] = z_arr
    re[:, 1] = phiw_arr

    Pi_arr = fcn_Pi_given(phiw_arr, cond_GT)                              
    Pi_div_DLP_arr = deepcopy(Pi_arr)/cond_GT['DLP']
    gp_arr = zeros(Nz)                                                 # constructing array for g+(z) function
    gm_arr = zeros(Nz)                                                 # constructing array for g-(z) function
    CT.gen_gpm_arr(sign_plus,  z_div_L_arr, Pi_div_DLP_arr, cond_GT['k'], gp_arr)
    CT.gen_gpm_arr(sign_minus, z_div_L_arr, Pi_div_DLP_arr, cond_GT['k'], gm_arr)
    cond_GT['Gk'] = CT.get_Gk_boost(cond_GT['k'], dz_div_L, gp_arr[-1], gm_arr[-1])
    cond_GT['Bp'] = CT.get_Bpm_conv(sign_plus, cond_GT)
    cond_GT['Bm'] = CT.get_Bpm_conv(sign_minus, cond_GT)


    ind_z0 = 0 #z-index at inlet
    
    z0_div_L = 0. #z-coord at inlet
    
    r0_div_R = 0. #r-coord at the centerline of pipe
    rw_div_R = 1. #r-coord at the membrane wall
    
    for i in range(Nz):
        # when iteration is done
        phi_arr_zi = zeros(Ny)
        Ieta_arr_zi = zeros(Ny)
        ID_arr_zi = zeros(Ny)


        zi_div_L = z_div_L_arr[i]
        re[i, 2] = cond_GT['DLP']*GT.get_P_conv(r0_div_R, zi_div_L, cond_GT, gp_arr[i], gm_arr[i])
        vw_div_vw0_zi = GT.get_v_conv(rw_div_R, zi_div_L, Pi_div_DLP_arr[i], cond_GT, gp_arr[i], gm_arr[i])
        re[i, 3] = cond_GT['vw0']*vw_div_vw0_zi

        GT.gen_phi_wrt_yt(z_div_L_arr[i], phiw_arr[i], fcn_Dc_given, vw_div_vw0_zi, y_div_R_arr, phi_arr_zi, cond_GT)
        GT.gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_zi, Ieta_arr_zi, fcn_eta_given, cond_GT)
        GT.gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_zi, ID_arr_zi, fcn_Dc_given, cond_GT)

        re[i, 4] = cond_GT['u_HP']*GT.get_u_conv(r0_div_R, zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
        re[i, 5] = Pi_arr[i]
        re[i, 6] = re[i, 2] - cond_GT['Pper']
        re[i, 7] = re[i, 3]/cond_GT['vw0']
        re[i, 8] = re[i, 4]/cond_GT['u_HP']

        Phi_z = 0. # Using Eq. (50)
        u1 = 0; u2 = 0;
        for j in range(1, Ny):
            dy = y_div_R_arr[j] - y_div_R_arr[j-1]
            u1 = u2
            u2 = GT.get_u_conv(1. - y_div_R_arr[j], zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[j])

            j1 = phi_arr_zi[j-1]*u1; r1 = 1. - y_div_R_arr[j-1]
            j2 = phi_arr_zi[j]*u2; r2 = 1. - y_div_R_arr[j]

            Phi_z += 0.5 * dy * (j1*r1 + j2*r2)
            re[i, 9] = Phi_z * 2. * pi * cond_GT['u_HP']*cond_GT['R']**2.0
    
    # Nz = size(z_arr); Ny = size(yt_arr)
    # dz = z_arr[1] - z_arr[0]
    # re = zeros([Nz, 11])
    # re[:,0] = z_arr
    # re[:,1] = phiw_arr
    # Pi_arr = fcn_Pi(re[:,1], cond_GT)
    # gp_arr = zeros(Nz)
    # gm_arr = zeros(Nz)
    # for i in range(Nz):
    #     # gp_arr[i] = get_gpm(z_arr[i], dz, +1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
    #     # gm_arr[i] = get_gpm(z_arr[i], dz, -1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
    #     CT.gen_gpm_arr(+1.0, z_div_L_arr, Pi_div_DLP_arr, k, gp_arr)
    #     CT.gen_gpm_arr(-1.0, z_div_L_arr, Pi_div_DLP_arr, k, gm_arr)
        
    #     phi_arr = zeros(Ny)
    #     Ieta_arr = zeros(Ny)
    #     ID_arr = zeros(Ny)
    #     u0 = cond_GT['R']**2.0 * (cond_GT['Pin'] - cond_GT['Pout'])/(4. * cond_GT['eta0'] * cond_GT['L'])
    # for i in range(Nz):
    #     z_tmp = z_arr[i]
    #     re[i, 2] = get_P_CT_boost(0., z_tmp, cond_GT, cond_GT['dz'], Pi_arr, gp_arr[i], gm_arr[i])
    #     re[i, 3] = get_vw_GT_boost(z_tmp, cond_GT, Pi_arr, gp_arr[i], gm_arr[i])
    #     vw_div_vw0 = re[i, 3]/cond_GT['vw0']
    #     get_phi_with_fixed_z_GT(z_tmp, cond_GT, re[i,1], Pi_arr, fcn_Dc_given, vw_div_vw0, yt_arr, phi_arr)
    #     get_int_eta_phi(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, yt_arr, phi_arr, Ieta_arr)
    #     get_int_D_phi(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, yt_arr, phi_arr, ID_arr)
    #     INT_Ieta = interp1d(yt_arr, Ieta_arr)
    #     re[i, 4] = get_u_center_GT_boost(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, Ieta_arr[-1], gp_arr[i], gm_arr[i])
    #     re[i, 5] = Pi_arr[i]

    #     re[i, 6] = re[i, 2] - cond_GT['Pper']
    #     re[i, 7] = re[i, 3]/cond_GT['vw0']
    #     re[i, 8] = re[i, 4]/u0

    #     Phi_z = 0.
    #     u1 = 0; u2 = 0;
    #     for j in range(1, Ny): # cal: cross-sectional averaged particle flux
    #         dy = yt_arr[j] - yt_arr[j-1]
    #         u1 = u2
    #         u2 = get_u_GT_boost(cond_GT['R']*(1. - yt_arr[j]), z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, INT_Ieta, gp_arr[i], gm_arr[i])/u0
    #         Phi_z += 0.5 * dy * (phi_arr[j] * u2 * (1. - yt_arr[j]) + phi_arr[j-1]*u1*(1. - yt_arr[j-1]))
    #         Phi_z *= 2. * pi * u0 * cond_GT['R']**2.0
    #         re[i, 9] = Phi_z
    #         # if i != 0:
    #         #     re[i, 10] = get_delta_z(yt_arr, vw_div_vw0, cond_GT['epsilon_d'], ID_arr, cond_GT['phi_bulk'], re[i, 1])
    savetxt(fn_out, re) # write the result into the filename described in fn_out
    return 0


#############################################################################
#   Analysis codes for mBLA_UF code                                         #
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
#   1. The front factor for Phi_z, Phi_ex_z, and Phi_b_z come from the CM   #
#   For FMM/FMS geometries, it need to convert the proper front factors     #
#   as described in the paper [2].                                          #
#############################################################################



from aux_functions import *
from membrane_geometry_functions import *

import sol_solvent as PS
import sol_CT as CT
import sol_GT as GT


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
    print ('# Reference [1]: Park and Naegele, JCP (2020)                #')
    print ('# Reference [2]: Park and Naegele, Membranes (2021)          #')    
    print ('##############################################################')
    print ('')
    print ('WARNING: Phi_ast definition has not been updated properly.')
    print ('')
    print ('Executed time (date): %s'%(now_str))
    print ('with Arguments: ', fn_inp, fn_out)
    print ('     Log will be stored in ', fn_out_log)

    f_log.write('\n##############################################################\n')
    f_log.write('# Semi-analytic solution of mBLA UF using CP layer model     #\n')        
    f_log.write('# git-repository: https://github.com/gwpark-git/mBLA_UF.git  #\n')
    f_log.write('# Developer: Gunwoo Park (IBI-4, Forschungszentrum Juelich)  #\n')
    f_log.write('# Reference [1]: Park and Naegele, JCP (2020)                #\n')
    f_log.write('# Reference [2]: Park and Naegele, Membranes (2021)          #\n')    
    f_log.write('##############################################################\n\n')
    f_log.write('WARNING: Phi_ast definition has not been updated properly.\n\n')
    f_log.write('Executed time (date): %s\n'%(now_str))
    f_log.write('with Arguments: %s, %s\n'%(fn_inp, fn_out))
    f_log.write('     Log will be stored in %s\n\n'%(fn_out_log))
    return now_str
    

def print_summary(cond_GT, f_log=None):
    print ('\nSystem and operating conditions ( geometry = ', cond_GT['membrane_geometry'], '/ BC_inlet = ', cond_GT['BC_inlet'], ' ):')
    print ('  - Summary of dimensional quantities (in SI units):')
    print ('\tPin_ast=%9d,   Pper=%9d, ref_Pout=%9d '%(int(cond_GT['Pin_ast']), int(cond_GT['Pper']), int(cond_GT['Pout'])))
    print ('\tu_ast=%lf'%(cond_GT['u_ast']))
    print ('\tDLP=%9d, DTP_PS=%9d,   DTP_HP=%9d '%(int(cond_GT['DLP']), int(cond_GT['DTP_PS']), int(cond_GT['DTP_HP'])))
    print ('\tLp =%4.3e,   eta0=%4.3e,        R=%4.3e,        L=%4.3e'%(cond_GT['Lp'], cond_GT['eta0'], cond_GT['R'], cond_GT['L']))
    print ('\ta  =%4.3e,    a_H=%4.3e,       D0=%4.3e,  Phi_ast=%4.3e'%(cond_GT['a'], cond_GT['a_H'], cond_GT['D0'], cond_GT['Phi_ast']))

    print ('\n  - Permeability of system (kappa-Darcy and Lp):')
    if (cond_GT['define_permeability'].lower()=='darcy'):
        print('\tDarcy-permeability (kappa-Darcy = %.6e) is used instead of membrane permeability (Lp)'%(cond_GT['kappa_Darcy']))
        print('\th/R=%.4f, which leads Lp=%4.3e'%(cond_GT['h']/cond_GT['R'], cond_GT['Lp']))
        # print('\tCalculating Lp requires h, R, and eta0: Lp = %4.3e'%cond_GT['Lp'])
    else:
        print('\tMembrane-permeability (Lp = %4.3e) is used directly'%(cond_GT['Lp']))
        print('\tIt does not require h for all the calculation. AS AN EXAMPLE, if h=R/2, the re-calculated kappa-Darcy value will be %.6e'%(get_kappa_Darcy_from_Lp(cond_GT['membrane_geometry'], cond_GT['Lp'], cond_GT['R']/2., cond_GT['R'], cond_GT['eta0'])))

    print ('\n  - Corresponding dimensionless quantities: (WARNING: lam1 and lam2 have different definitions from [2])')
    print ('\tlam1=%.4f, lam2=%.4f'%(cond_GT['lam1'], cond_GT['lam2']))
    print ('\tk=%.4f, alpha_ast=%.4f, beta_ast=%.4f, gamma=a_H/a=%4.3e'%(cond_GT['k'], cond_GT['alpha_ast'], cond_GT['beta_ast'], cond_GT['gamma']))
    print ('\tepsilon=%4.3e, epsilon_d=%4.3e (Pe_R=%.1f)'%(cond_GT['R']/cond_GT['L'], cond_GT['epsilon_d'], 1./cond_GT['epsilon_d']))

    if not f_log.closed:
        f_log.write('\nSystem and operating conditions:\n' )
        f_log.write('  - Summary of dimensional quantities (in SI units):\n')        
        f_log.write('\tPin_ast=%9d,   Pper=%9d, ref_Pout=%9d \n'%(int(cond_GT['Pin_ast']), int(cond_GT['Pper']), int(cond_GT['Pout'])))
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
    print('iter=%d, chi_A=%4.3e (weight = %4.3e)'%(report_step[0], chi_A, cond_GT['weight']))
    print('\tz_max=%4.3f, phiw(z_max)=%.4f, phiw(L)=%.4f\n\t<Pi>/DTP_HP=%4.4f, DTP/DTP_HP=%4.4f'%(report_step[1], report_step[2], report_step[3], report_step[4], report_step[5]))
    print('\tP(0)/Pin_ast=%4.3f, u(0,0)/u_ast=%4.3f\n'%(report_P_div_DLP_arr[0]*cond_GT['DLP']/cond_GT['Pin_ast'], (report_P_div_DLP_arr[0] - report_P_div_DLP_arr[1])/(z_div_L_arr[1] - z_div_L_arr[0])))
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


def aux_gen_analysis(z_arr, y_div_R_arr, phiw_arr, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given):
    """ OUTPUT DATA FORMAT of mBLA-UF 
    
    Save analysis data into file with name of fn_out    
    Data first stored in array "re":
        re[ 0] = z                          in the unit of m
        re[ 1] = phi_w(z)                   in the dimensionless unit
        re[ 2] = P(z)                       in the unit of Pa
        re[ 3] = v_w(z)                     in the unit of m/sec
        re[ 4] = u(r=0, z)                  in the unit of m/sec
        re[ 5] = Pi(phi_w(z))               in the unit of Pa
        re[ 6] = P(z) - P_perm              in the unit of Pa
        re[ 7] = v_w(z)/v^ast               in the dimensionless unit
        re[ 8] = u(r=0, z)/u^ast            in the dimensionless unit
        re[ 9] = Phi(z)/Phi(0)              dimensionless
        re[10] = j_mean(z)/j0_mean          dimensionless (the same as before: re[9])
        re[11] = excess part of re[10]      dimensionless
        re[12] = bulk part of re[12]        dimensionless
    
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

    re = zeros([Nz, 15])

    re[:, 0] = z_arr
    re[:, 1] = phiw_arr

    Pi_arr = fcn_Pi_given(phiw_arr, cond_GT)                              
    Pi_div_DLP_arr = deepcopy(Pi_arr)/cond_GT['DLP']
    gp_arr = zeros(Nz)                                                 # constructing array for g+(z) function
    gm_arr = zeros(Nz)                                                 # constructing array for g-(z) function
    CT.gen_gpm_arr(sign_plus,  z_div_L_arr, Pi_div_DLP_arr, cond_GT['k'], gp_arr)
    CT.gen_gpm_arr(sign_minus, z_div_L_arr, Pi_div_DLP_arr, cond_GT['k'], gm_arr)
    cond_GT['Gk'] = CT.get_Gk_boost(cond_GT['k'], dz_div_L, gp_arr[-1], gm_arr[-1], cond_GT['denom_Gk_BC_specific'])

    cond_GT['Bp'] = CT.get_Bpm_conv(sign_plus, cond_GT)
    cond_GT['Bm'] = CT.get_Bpm_conv(sign_minus, cond_GT)


    ind_z0 = 0 #z-index at inlet
    
    z0_div_L = 0. #z-coord at inlet
    
    r0_div_R = 0. #r-coord at the centerline of pipe
    rw_div_R = 1. #r-coord at the membrane wall

    u_inlet = cond_GT['u_ast']*GT.get_u_conv(r0_div_R, 0., cond_GT, gp_arr[0], gm_arr[0], 1) # Ieta_arr_zi[-1]=1. because inlet flow field is assumed to be hydrodynamically fully developed flow, but the concentration is not yet build up (see the development length mentioned in [1])
    u_inlet_mean = get_Uout_mean(cond_GT['membrane_geometry'])*u_inlet 

    phi_b = cond_GT['phi_bulk']    
    j0_mean = phi_b*u_inlet_mean # This is the base unit used for the cross-sectional averaged particle flux j. Note that j_mean(z)=j_mean(0) which is j0_mean
    
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
        Ieta_arr_zi /= Ieta_arr_zi[-1]
        GT.gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_zi, ID_arr_zi, fcn_Dc_given, cond_GT)

        re[i, 4] = cond_GT['u_ast']*GT.get_u_conv(r0_div_R, zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
        re[i, 5] = Pi_arr[i]
        re[i, 6] = re[i, 2] - cond_GT['Pper']
        re[i, 7] = re[i, 3]/cond_GT['vw0']
        re[i, 8] = re[i, 4]/cond_GT['u_ast']

        # Phi, Phi_ex, Phi_b in the original code uses the Eqs. (50), (60) and (60) respectively in reference [1].
        # This is based on the "total" cross-sectional particle flux.
        # With the other geometry considered in reference [2], however, this is no more case and we only consider cross-sectional "averaged" particle flux.
        # This indicate that we reduce the calculated Phi by reducing factor Phi0, so Phi/Phi0.
        # This is equivalent to the j_mean/j0_mean where j_mean = Phi/A (where A is the cross-sectional area)
        
        Phi_z = 0. 
        Phi_ex_z = 0. 
        Phi_b_z = 0. 
        u1 = 0; u2 = 0;

        for j in range(1, Ny): # The cross-sectional average step
            dy = y_div_R_arr[j] - y_div_R_arr[j-1]
            u1 = u2
            u2 = GT.get_u_conv(1. - y_div_R_arr[j], zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[j])

            # Note: u1 and u2 are dimensionless axial velocities: u(z1)/u^0 and u(z2)/u^0.
            # Therefore the division with uinlet_div_u0 for the proper non-dimensionalization
            # Note that for the pressure-inlet BC, uinlet_div_u0 is not equal to 1 (usually slightly higher than 1)
            # In the case of the velocity-inlet BC, uinlet_div_u0 must be 1.
            
            j_ex_1 = (phi_arr_zi[j-1] - phi_b)*u1*cond_GT['u_ast']/j0_mean
            j_ex_2 = (phi_arr_zi[j] - phi_b)*u2*cond_GT['u_ast']/j0_mean

            j_b_1 = phi_b*u1*cond_GT['u_ast']/j0_mean
            j_b_2 = phi_b*u2*cond_GT['u_ast']/j0_mean

            j1 = j_ex_1 + j_b_1
            j2 = j_ex_2 + j_b_2


            J_r1 = J_int_yt(y_div_R_arr[j-1], cond_GT['membrane_geometry'])
            J_r2 = J_int_yt(y_div_R_arr[j], cond_GT['membrane_geometry'])

            # Numerical integration using Trapezoidal rule
            Phi_z += 0.5 * dy * (j1*J_r1 + j2*J_r2) 
            Phi_ex_z += 0.5 * dy * (j_ex_1*J_r1 + j_ex_2*J_r2)
            Phi_b_z += 0.5 * dy * (j_b_1*J_r1 + j_b_2*J_r2)

        # re[i, 9] = Phi_z * 2. * pi * cond_GT['u_ast']*cond_GT['R']**2.0
        re[i, 9] = Phi_z # New definition is cross-sectional averaged particle flux (j_mean) divided by j0_mean
        # For this reason, at this moment re[:,9] has the same definition of re[:,10]
        # This duplication is left as unsolved for the interface compatibility of my post-processing codes (which is not yet updated on GitHub repository)

        # The following are for the reduced cross-sectional averaged particle flux
        if (cond_GT['membrane_geometry']=='FMS'):
            add_re = (2/3.)*phi_b*GT.get_u_conv(0., zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
            re[i, 10] = (Phi_z + add_re)/2.0
            re[i, 11] = Phi_ex_z/2.0
            re[i, 12] = (Phi_b_z + add_re)/2.0
        elif (cond_GT['membrane_geometry']=='FMM'):
            re[i, 10] = Phi_z 
            re[i, 11] = Phi_ex_z
            re[i, 12] = Phi_b_z
        else:
            # In the case of CM geometry, the front factor 2 appears because (2\pi \int_0^R (1-(y/R)^2)ydy)/(\pi R^2) = 2 \int_0^1 (1-yt^2)yt dyt)
            re[i, 10] = 2.0*Phi_z 
            re[i, 11] = 2.0*Phi_ex_z
            re[i, 12] = 2.0*Phi_b_z
            
    # savetxt(fn_out, re) # write the result into the filename described in fn_out
    return re


def gen_analysis(z_arr, y_div_R_arr, phiw_arr, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out):
    re = aux_gen_analysis(z_arr, y_div_R_arr, phiw_arr, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given)
    savetxt(fn_out, re) # write the result into the filename described in fn_out
    return 0
    

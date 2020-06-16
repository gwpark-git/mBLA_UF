#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


import sys
from sol_CT import *
from sol_GT import *
from sol_GT_parallel import *

from osmotic_pressure_CS import *
from transport_properties_SPHS import *

from numpy import *
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.linalg import norm
from copy import deepcopy

def get_psi(s_bar, phi_b, phi_w):
    return exp(-s_bar) - (phi_b/(phi_w - phi_b))*(s_bar*exp(-s_bar))

def get_delta_z(yt_arr, vw_div_vw0, epsilon_d, INT_Dc, phi_b, phi_w):
    Ny = size(yt_arr)
    psi_2 = 1.0 # it must be unity at the wall
    numer_psi = 0.    
    denom_psi = 0.
    for i in range(1, Ny):
        dy = yt_arr[i] - yt_arr[i-1]
        s_bar = (vw_div_vw0/epsilon_d)*INT_Dc[i]
        psi_1 = psi_2
        psi_2 = get_psi(s_bar, phi_b, phi_w)

        f1 = psi_1 * yt_arr[i-1] * (1. - yt_arr[i-1])
        f2 = psi_2 * yt_arr[i] * (1. - yt_arr[i])
        numer_psi += 0.5 * dy * (f1 + f2)
        
        f1_denom = psi_1 * (1. - yt_arr[i-1])
        f2_denom = psi_2 * (1. - yt_arr[i])
        denom_psi += 0.5 * dy * (f1_denom + f2_denom)
    return numer_psi / denom_psi
    

def gen_analysis(z_arr, yt_arr, phiw_arr, cond_GT, fcn_Pi, fcn_Dc, fcn_eta, fn_out):
    Nz = size(z_arr); Ny = size(yt_arr)
    re = zeros([Nz, 11])
    re[:,0] = z_arr
    re[:,1] = phiw_arr
    Pi_arr = fcn_Pi(re[:,1], cond_GT)
    gp_arr = zeros(Nz)
    gm_arr = zeros(Nz)
    for i in range(Nz):
        gp_arr[i] = get_gpm(z_arr[i], dz, +1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
        gm_arr[i] = get_gpm(z_arr[i], dz, -1.0, Pi_arr, cond_GT['k'], cond_GT['L'])
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
        re[i, 4] = get_u_center_GT_boost(z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, Ieta_arr[-1], gp_arr[i], gm_arr[i])
        re[i, 5] = Pi_arr[i]

        re[i, 6] = re[i, 2] - cond_GT['Pper']
        re[i, 7] = re[i, 3]/cond_GT['vw0']
        re[i, 8] = re[i, 4]/u0

        Phi_z = 0.
        u1 = 0; u2 = 0;
        for j in range(1, Ny):
            dy = yt_arr[j] - yt_arr[j-1]
            u1 = u2
            u2 = get_u_GT_boost(cond_GT['R']*(1. - yt_arr[j]), z_tmp, cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, phi_arr, INT_Ieta, gp_arr[i], gm_arr[i])/u0
            Phi_z += 0.5 * dy * (phi_arr[j] * u2 * (1. - yt_arr[j]) + phi_arr[j-1]*u1*(1. - yt_arr[j-1]))
        Phi_z *= 2. * pi * u0 * cond_GT['R']**2.0
        re[i, 9] = Phi_z
        if i <> 0:
            re[i, 10] = get_delta_z(yt_arr, vw_div_vw0, cond_GT['epsilon_d'], ID_arr, cond_GT['phi_bulk'], re[i, 1])
    savetxt(fn_out, re)
    return 0
            

if len(sys.argv) < 3:
    print 'Usage: '
    print 'argv[1] == input file as Python script'
    print 'argv[2] == output file name'
else:

    
    fn_inp = str(sys.argv[1])
    fn_out = str(sys.argv[2])
    print 'Arguments: ', fn_inp, fn_out

    # system properties
    
    execfile(fn_inp)
    z_arr = linspace(0, L_channel, Nz)
    dz = z_arr[1] - z_arr[0]
    dr = (1/float(Nr))*R_channel

    k_B = const.k 
    kT = k_B*T

    a_H = a_particle*gamma # hydrodynamic radius
    D0 = kT/(6.*pi*eta0*a_H)
    Va = (4./3.)*pi*a_particle**3.0 # volume measure is still using particle exclusion-size

    # for solvent flow and CT 
    k = 4.*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)
    prefactor_U = sqrt(Lp*R_channel/eta0)

    # pre condition generation
    print 'k, prefactor_U :', k, prefactor_U

    Pin = get_Pin(DLP, ref_Pout)
    Pper = get_Pper(DLP, ref_DTP, k, ref_Pout)

    print '\nSummary:' 
    print 'Pin, Pper, ref_Pout :', Pin, Pper, ref_Pout

    # pre-conditioning
    pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0, 'preU':prefactor_U}
    print pre_cond

    cond_BT = get_cond(pre_cond, Pin, ref_Pout, Pper)
    # print cond_BT

    DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper
    #vw_dim = cond_BT['Lp']*DLP
    vw0 = cond_BT['Lp']*DTP_HP
    epsilon_d = D0/(cond_BT['R']*vw0)
    print epsilon_d, vw0

    # parameters related with sol_GT
    epsilon_d = D0/(R_channel*vw0)

    # generation data for osmotic pressure
    if IDENT_parallel:
        if IDENT_modification:
            phiw_update = get_new_phiw_div_phib_modi_arr_parallel
        else:
            phiw_update = get_new_phiw_div_phib_arr_parallel
    else:
        if IDENT_modification:
            phiw_update = get_new_phiw_div_phib_modi_arr
        else:
            phiw_update = get_new_phiw_div_phib_arr


    Pi_arr = zeros(size(phiw_arr)) 
            
    cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, Pi_arr)
    cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma)

    
    phi_b= cond_GT['phi_bulk']
    phiw_div_phib_arr = phiw_arr/phi_b

    phiw_set_1 = phiw_div_phib_arr # normalized initial condition
    
    phiw_set_2 = deepcopy(phiw_set_1)
    gp_arr = zeros(Nz)
    gm_arr = zeros(Nz)

    yt_arr = gen_yt_arr(cond_GT)
    Ny = size(yt_arr)
    if IDENT_verbose:
        print '  IDENT_verbose is turned on, which will record the analysis of result for every FPI steps'
        print '                The current IDENT_verbose option is not parallelized, which takes longer time for computation'
        fn_ver = fn_out + '.%05d'%(0)
        gen_analysis(z_arr, yt_arr, phiw_set_1*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
        
    for n in range(N_iter):
        phiw_set_1 = deepcopy(phiw_set_2)
        Pi_arr = fcn_Pi_given(phiw_set_1*phi_b, cond_GT)
        for i in range(Nz):
            gp_arr[i] = get_gpm(z_arr[i], dz, +1.0, Pi_arr, k, cond_BT['L'])
            gm_arr[i] = get_gpm(z_arr[i], dz, -1.0, Pi_arr, k, cond_BT['L'])
        cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, Pi_arr)
        cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, cond_GT['gamma'])
        cond_GT['k'] = cond_GT['k'] * eta_div_eta0_SPHS(phi_b, cond_GT)

        phiw_set_2= phiw_update(cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, z_arr, phiw_set_1, weight, gp_arr, gm_arr, yt_arr)
        if IDENT_verbose:
            fn_ver = fn_out + '.%05d'%(n + 1)
            gen_analysis(z_arr, yt_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
            
        ind_max = argmax(phiw_set_2)
        print 'n=%d, phiw/b(0)=%4.0f, phiw/b(L)=%4.0f, max:(phiw(%4.3f)/b)=%4.0f'%(n, phiw_set_2[0], phiw_set_2[-1], z_arr[ind_max], phiw_set_2[ind_max])

        err = norm(phiw_set_1 - phiw_set_2)
        print 'norm(p1-p2) : %4.3e, weight : %4.3f\n'%(err, weight)

    # Constructing the results and its analysis
    gen_analysis(z_arr, yt_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out)

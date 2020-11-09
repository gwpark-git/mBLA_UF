#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


import sys
# from sol_solvent import *
# from sol_CT import *
# from sol_GT import *
# from sol_GT_parallel import *
import sol_solvent as PS
import sol_CT as CT
import sol_GT as GT

import osmotic_pressure_CS as CS
import transport_properties_SPHS as PHS
# from osmotic_pressure_CS import *
# from transport_properties_SPHS import *
# from analysis import *
from analysis import *

from numpy import *
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.linalg import norm
from copy import deepcopy


if len(sys.argv) == 1:
    print ('Usage: ')
    print ('    argv[1] == input file as Python script')
    print ('    argv[2] == output file name')
    print ('')
    print ('Output:')
    print ('    col 0: z                 in the unit of m')
    print ('    col 1: phi_w(z)          in the dimensionless unit')
    print ('    col 2: P(z)              in the unit of Pa')
    print ('    col 3: v_w(z)            in the unit of m/sec')
    print ('    col 4: u(r=0, z)         in the unit of m/sec')
    print ('    col 5: Pi(phi_w(z))      in the unit of Pa')
    print ('    col 6: P(z) - P_perm     in the unit of Pa')
    print ('    col 7: v_w(z)/v^\ast     in the dimensionless unit')
    print ('    col 8: u(r=0, z)/u^\ast  in the dimensionless unit')
    print ('    col 9: Phi(z)            in the unit of m^3/sec')
else:
    fn_inp = str(sys.argv[1]) # get filename for input script
    fn_out = str(sys.argv[2]) # get filename for output data
    print ('Arguments: ', fn_inp, fn_out)
    
    # execfile(fn_inp)
    exec(open(fn_inp).read())

    # Note :
    #      :Some of functions uses its own base unit, whereas others are not.
    #      :This is due to internal history of the code, so it must be careful to check the units.
    #      :The revised code soon to be published.
    
    z_arr = linspace(0, L_channel, Nz)                                 # discretized z
    z_div_L_arr = z_arr/L_channel
    
    dz = z_arr[1] - z_arr[0]                                           # equi-step size for z
    dz_div_L = dz/L_channel
    
    dr = (1/float(Nr))*R_channel                                       # test step size for r, which will be adjusted in accordance with BLA
    dr_div_R = dr/R_channel

    k_B = const.k                                                      # Boltzmann constant
    kT = k_B*T                                                         # thermal energy

    a_H = a_particle*gamma                                             # hydrodynamic radius
    D0 = kT/(6.*pi*eta0*a_H)                                           # Stokes-Einstein-Sutherland
    Va = (4./3.)*pi*a_particle**3.0                                    # volume measure is still using particle exclusion-size

    k = 4.*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)            # dimensionless parameter k
    prefactor_U = sqrt(Lp*R_channel/eta0)                              # this is related with unit conversion

    print ('k, prefactor_U :', k, prefactor_U)

    Pin = PS.get_Pin(DLP, ref_Pout)                                       # calculating Pin for the given DLP and Pout
    Pper = PS.get_Pper(DLP, ref_DTP, k, ref_Pout)                         # calculating Pper for the given DLP, DTP_linear, k, and P_out

    print ('\nSummary:' )
    print ('Pin, Pper, ref_Pout :', Pin, Pper, ref_Pout)

    pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0, 'preU':prefactor_U}
    print (pre_cond)

    cond_PS = PS.get_cond(pre_cond, Pin, ref_Pout, Pper)                  # allocating Blank Test (pure test) conditions

    DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper                            # length-averaged TMP with a linearly declined pressure approximation
    vw0 = cond_PS['Lp']*DTP_HP                                         # v^\ast
    epsilon_d = D0/(cond_PS['R']*vw0)                                  # 1/Pe_R
    print (epsilon_d, vw0)

    # if IDENT_parallel:                                                 # parallel computation
    #     if IDENT_modification:
    #         phiw_update = get_new_phiw_div_phib_modi_arr_parallel
    #     else:
    #         phiw_update = get_new_phiw_div_phib_arr_parallel
    # else:                                                              # single-process computation
    #     if IDENT_modification:
    #         phiw_update = get_new_phiw_div_phib_modi_arr
    #     else:
    #         phiw_update = get_new_phiw_div_phib_arr
    phiw_update = gen_new_phiw_div_phib_arr

    Pi_arr = zeros(size(phiw_arr))                                     # calculating osmotic pressure using initial conditions
    Pi_div_DLP_arr = Pi_arr/cond_PS['DLP']
    
    Gk_tmp = get_Gk(cond_PS['k'], dz/L_channel, Pi_div_DLP_arr)
    cond_CT = CT.get_cond(cond_PS, a_particle, Va, kT, dz, Pi_arr, Gk_tmp)     # allocating conditions for the constant transport properties
    cond_GT = GT.get_cond(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma, weight) # allocating conditions for the general transport properties

    
    phi_b= cond_GT['phi_bulk']                                         # set the feed/bulk concentration
    phiw_div_phib_arr = phiw_arr/phi_b                                 # reduced wall concentration
    phiw_set_1 = phiw_div_phib_arr                                     # reduced initial wall concentration
    phiw_set_2 = deepcopy(phiw_set_1)                                  # reduced initial wall concentration
    
    gp_arr = zeros(Nz)                                                 # constructing array for g+(z) function
    gm_arr = zeros(Nz)                                                 # constructing array for g-(z) function

    y_div_R_arr = GT.gen_y_div_R_arr(cond_GT)                                       # generating tilde_y with given conditions described in cond_GT
    Ny = size(y_div_R_arr) 
    # if IDENT_verbose:
    #     print ('  IDENT_verbose is turned on, which will record the analysis of result for every FPI steps')
    #     print ('                The current IDENT_verbose option is not parallelized, which takes longer time for computation')
    #     fn_ver = fn_out + '.%05d'%(0)
    #     gen_analysis(z_arr, y_div_R_arr, phiw_set_1*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
    re = zeros([Nz, 11])

    for n in range(N_iter):                                                           # main iterator with number n
        phiw_set_1 = deepcopy(phiw_set_2)                                             # reduced wall concentration inherited from the previous iteration
        Pi_arr = fcn_Pi_given(phiw_set_1*phi_b, cond_GT)                              # calculating osmotic pressure for the given phiw
        Pi_div_DLP_arr = deepcopy(Pi_arr)/cond_GT['DLP']
        CT.gen_gpm_arr(+1.0, z_div_L_arr, dz_div_L, Pi_div_DLP_arr, k, gp_arr)
        CT.gen_gpm_arr(-1.0, z_div_L_arr, dz_div_L, Pi_div_DLP_arr, k, gm_arr)
        Gk_tmp = get_Gk_boost(k, dz_div_L, Pi_div_DLP_arr, gp_arr[-1], gm_arr[-1])
        # for i in range(Nz):                                                           # generating g+(z) and g-(z) functions
        #     gp_arr[i] = CT.get_gpm(+1.0, z_div_L_arr[i], dz_div_L, Pi_div_DLP_arr, k)
        #     gm_arr[i] = CT.get_gpm(-1.0, z_div_L_arr[i], dz_div_L, Pi_div_DLP_arr, k)
        #     # gp_arr[i] = CT.get_gpm(z_arr[i], dz, +1.0, Pi_arr, k, cond_PS['L'])
        #     # gm_arr[i] = CT.get_gpm(z_arr[i], dz, -1.0, Pi_arr, k, cond_PS['L'])
        cond_CT = CT.get_cond(cond_PS, a_particle, Va, kT, dz, Pi_arr, Gk_tmp)                # update conditions for CT
        cond_GT = GT.get_cond(cond_CT, phi_bulk, epsilon_d, dr, dz, cond_GT['gamma'], weight) # update conditions for GT
        cond_GT['k'] = cond_GT['k'] * eta_div_eta0_SPHS(phi_b, cond_GT)               # update the dimensionless value k

        # phiw_set_2= phiw_update(cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given,\
        #                         z_arr, phiw_set_1, weight, gp_arr, gm_arr, y_div_R_arr)    # main FPI iterator
        phiw_update(phiw_set_2, cond_GT, fcn_Dc_given, fcn_eta_given, z_div_L_arr, phiw_set_1, Pi_div_DLP_arr, cond_GT['weight'], gp_arr, gm_arr, y_div_R_arr)
        # print(phiw_set_2)
        
        # if IDENT_verbose:                                                             # the case when each steps will be printed out
        #     fn_ver = fn_out + '.%05d'%(n + 1)
        #     gen_analysis(z_arr, y_div_R_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
            
        ind_max = argmax(phiw_set_2)                                                  # get index number for the maximum values of phiw(z)
        print ('n=%d, phiw/b(0)=%4.3f, phiw/b(L)=%4.3f, max:(phiw(%4.3f)/b)=%4.3f'%(n, phiw_set_2[0], phiw_set_2[-1], z_arr[ind_max], phiw_set_2[ind_max]))

        err = norm(phiw_set_1 - phiw_set_2)                                           # estimated deviations
        print ('norm(p1-p2) : %4.3e, weight : %4.3f\n'%(err, weight))

        if(n == N_iter-1):
            re[:, 0] = z_arr
            re[:, 1] = phiw_set_2*phi_b

            Pi_arr = fcn_Pi_given(phiw_set_2*phi_b, cond_GT)                              
            Pi_div_DLP_arr = deepcopy(Pi_arr)/cond_GT['DLP']

            for i in range(Nz):
                # when iteration is done
                phi_arr_zi = zeros(Ny)
                Ieta_arr_zi = zeros(Ny)
                ID_arr_zi = zeros(Ny)

                
                zi_div_L = z_div_L_arr[i]
                re[i, 2] = cond_GT['DLP']*GT.get_P_conv(0., zi_div_L, cond_GT, gp_arr[i], gm_arr[i])
                vw_div_vw0_zi = GT.get_v_conv(1., zi_div_L, Pi_div_DLP_arr[i], cond_GT['k'], cond_GT['alpha_ast'], cond_GT['Bp'], cond_GT['Bm'], gp_arr[i], gm_arr[i])
                re[i, 3] = cond_GT['vw0']*vw_div_vw0_zi

                GT.gen_phi_wrt_yt(z_div_L_arr[i], phiw_div_phib_arr[i]*phi_b, fcn_D, vw_div_vw0_zi, yt_arr, phi_arr_zi, cond_GT)
                GT.gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, Ieta_arr_zi, fcn_eta, cond_GT)
                GT.gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, ID_arr_zi, fcn_D, cond_GT)

                re[i, 4] = GT.get_u_conv(0., zi_div_L, cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
                re[i, 5] = Pi_arr[i]
                re[i, 6] = re[i, 2] - cond_GT['Pper']
                # u_ast is not defined yet. [WIP]
                
                # re[i, 7] = re[i
                # uZ_zi = get_u_conv(r0_div_R, z_div_L_arr[i], cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
                

    gen_analysis(z_arr, y_div_R_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out)

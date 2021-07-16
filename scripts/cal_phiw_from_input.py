#############################################################################
#   Main script for mBLA_UF code                                            #
#                                                                           #
#   Used in the paper:                                                      #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   doi: 10.1063/5.0020986                                                  #
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




import sys
import sol_solvent as PS
import sol_CT as CT
import sol_GT as GT

import osmotic_pressure_CS as CS
import transport_properties_SPHS as PHS
from membrane_geometry_functions import *
from analysis import *

from numpy import *
import scipy.constants as const
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.linalg import norm
from copy import deepcopy


if __name__ == '__main__' :
    if len(sys.argv) == 1:
        print ('Usage: ')
        print ('    argv[1] == input file as Python script')
        print ('    argv[2] == output file name')
        print ('               (note: iteration log will be stored in argv[2].log file)')
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
        print ('TEMP: col 10: Phi(z)/Q^\ast')
        print ('TEMP: col 10: Phi_ex(z)/Q^\ast')
        print ('TEMP: col 10: Phi_b(z)/Q^\ast')        
    else:
        fn_inp = str(sys.argv[1]) # get filename for input script
        fn_out = str(sys.argv[2]) # get filename for output data
        fn_out_log = str(sys.argv[2] + '.log') # get filename for log data
        f_log = open(fn_out_log, 'w')
        print_preface(fn_inp, fn_out, fn_out_log, f_log)

        # execfile(fn_inp)
        exec(open(fn_inp).read())

        # Note :
        #      :Some of functions uses its own base unit, whereas others are not.
        #      :This is due to internal history of the code, so it must be careful to check the units.
        #      :The revised code soon to be published.

        # if (membrane_geometry == 'HF'):
            
        
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

        if (define_permeability.lower()=='darcy'):
            # this will check whether permeability is given by kappa_Darcy or Lp
            # this also means h_membrane is given
            Lp = get_Lp_from_kappa_Darcy(membrane_geometry, kappa_Darcy, h_membrane, R_channel, eta0)

        else:
            # this is normal situation, and Lp was directly given from input file
            # even though h_membrane is not necessary to be defined in this case,
            # we will give as a reference value for it: h = R/2
            # this is due to the easier understand the pre_cond values below
            # this also allows us some guess of Darcy's permeability of the membrane
            h_membrane = R_channel/2.
            kappa_Darcy = get_kappa_Darcy_from_Lp(membrane_geometry, Lp, h_membrane, R_channel, eta0)

            
        lam1 = get_lam1(membrane_geometry)
        lam2 = get_lam2(membrane_geometry)

        
        k = get_effective_permeability_parameter_K(lam1, lam2, R_channel, L_channel, Lp, eta0)


        # k = lam1*lam2*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)
        # print('dimensionless quantities (lam1, lam2, k) = ', lam1, lam2, k)
        # k = get_K(lam1, lam2, membrane_geometry, R_channel, L_channel, Lp, eta0)
        # k = 4.*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)            # dimensionless parameter k
        if BC_inlet == 'velocity':
            u_ast = u_inlet
            DLP = get_DLP_from_uin(u_ast, lam1, eta0, R_channel, L_channel)
        elif BC_inlet == 'pressure':
            # u_ast = R_channel**2.0 * DLP/(4.*eta0*L_channel) # u_ast = u_HP when pressure inlet BC is used
            u_ast = R_channel**2.0 * DLP/(lam1*eta0*L_channel) # u_ast = u_HP when pressure inlet BC is used
            
        
        Pin_ast = PS.get_Pin_ast(DLP, ref_Pout)                                       # calculating Pin_ast for the given DLP and Pout
        if BC_perm == 'Pperm':
            print('BC_perm is set with Pperm: this will re-calculate ref_DTP using linear-approximated value.')
            ref_DTP = PS.cal_DTP_HP(Pin_ast, ref_Pout, ref_Pperm)
            Pper = ref_Pperm
        else:
            print('BC_perm is set with DTP (or not-specified): this will use ref_DTP to determine Pperm. This has advantage when we want to compare BCP and BCu cases.')
            Pper = PS.get_Pper(DLP, ref_DTP, k, ref_Pout)                         # calculating Pper for the given DLP, DTP_linear, k, and P_out

        pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0, 'membrane_geometry':membrane_geometry, 'lam1':lam1, 'lam2':lam2, 'define_permeability':define_permeability, 'h':h_membrane, 'kappa_Darcy':kappa_Darcy, 'BC_inlet':BC_inlet, 'DLP':DLP, 'phi_freeze':phi_freeze, 'Nz':Nz}
        cond_PS = PS.get_cond(pre_cond, Pin_ast, ref_Pout, Pper, u_ast)                  # allocating Blank Test (pure test) conditions

        DTP_HP = (1/2.)*(Pin_ast + ref_Pout) - Pper                            # length-averaged TMP with a linearly declined pressure approximation
        vw0 = cond_PS['Lp']*DTP_HP                                         # v^\ast
        epsilon_d = D0/(cond_PS['R']*vw0)                                  # 1/Pe_R


        Pi_arr = zeros(size(phiw_arr))                                     # Set zero osmotic pressure
        Pi_div_DLP_arr = Pi_arr/cond_PS['DLP']

        Gk_tmp = CT.get_Gk(cond_PS['k'], dz_div_L, Pi_div_DLP_arr, CT.get_denom_Gk_BC_specific(cond_PS['k'], cond_PS['BC_inlet']))
        cond_CT = CT.get_cond(cond_PS, phi_bulk, a_particle, a_H, Va, kT, dz, Gk_tmp)     # allocating conditions for the constant transport properties
        cond_GT = GT.get_cond(cond_CT, Nr, weight) # allocating conditions for the general transport properties

        phi_b= cond_GT['phi_bulk']                                         # set the feed/bulk concentration
        phiw_div_phib_arr = phiw_arr/phi_b                                 # reduced wall concentration
        phiw_set_1 = phiw_div_phib_arr                                     # reduced initial wall concentration
        phiw_set_2 = deepcopy(phiw_set_1)                                  # reduced initial wall concentration

        y_div_R_arr = GT.gen_y_div_R_arr(cond_GT)                          # generating tilde_y with given conditions described in cond_GT
        Ny = size(y_div_R_arr) 

        gp_arr = zeros(Nz)                                                 # constructing array for g+(z) function
        gm_arr = zeros(Nz)                                                 # constructing array for g-(z) function

        print_summary(cond_GT, f_log)
        print ('\nCalculating...(chi_A = norm(phiw_pre - phiw_new)/(Nz*phi_b))\n')
        sign_plus = +1.
        sign_minus = -1.

        # using aux arrays which will be of important for multiprocessing
        # this introduce memory overhead for single processor purpose
        # in the multiprocessing purpocse, this will be useful
        # hence, here the code used the list of arrays where index is relevant for the z-index
        phi_yt_arr = []
        ID_yt_arr = []
        Ieta_yt_arr = []
        for i in range(Nz):
            phi_yt_arr.append(zeros(Ny))
            ID_yt_arr.append(zeros(Ny))
            Ieta_yt_arr.append(zeros(Ny))


        for n in range(N_iter):                                                           # main iterator with number n
            phiw_set_1 = deepcopy(phiw_set_2)                                             # reduced wall concentration inherited from the previous iteration
            # if ((n+1)%10 == 0 and cond_GT['weight'] < 0.1):
            #     tmp_weight = 2.*cond_GT['weight']
            #     if (tmp_weight > 0.1):
            #         tmp_weight = 0.1
            #     print('the current weight = %4.3e will be updated to %4.3e\n'%(cond_GT['weight'], tmp_weight))
            #     cond_GT['weight'] = tmp_weight    
            #     # cond_GT['weight'] *= 2.
            #     # if (cond_GT['weight']>0.1):
            #     #     cond_GT['weight']==0.1
                    
            CT.gen_gpm_arr(sign_plus,  z_div_L_arr, Pi_div_DLP_arr, k, gp_arr)
            CT.gen_gpm_arr(sign_minus, z_div_L_arr, Pi_div_DLP_arr, k, gm_arr)
            cond_GT['Gk'] = CT.get_Gk_boost(k, dz_div_L, gp_arr[-1], gm_arr[-1], cond_GT['denom_Gk_BC_specific'])
            cond_GT['Bp'] = CT.get_Bpm_conv(sign_plus, cond_GT)
            cond_GT['Bm'] = CT.get_Bpm_conv(sign_minus, cond_GT)

            GT.gen_new_phiw_div_phib_arr(N_PROCESSES, phiw_set_2, cond_GT, fcn_Dc_given, fcn_eta_given, z_div_L_arr, phiw_set_1, Pi_div_DLP_arr, cond_GT['weight'], gp_arr, gm_arr, y_div_R_arr, phi_yt_arr, ID_yt_arr, Ieta_yt_arr)

            Pi_arr = fcn_Pi_given(phiw_set_2*phi_b, cond_GT)                              # calculating osmotic pressure for the given phiw
            Pi_div_DLP_arr = Pi_arr/cond_GT['DLP']

            chi_A = print_iteration_info(n, z_div_L_arr, phiw_set_1, phiw_set_2, cond_GT, Pi_div_DLP_arr, gp_arr, gm_arr, f_log)            
            if n == N_iter-1 or chi_A < TOL_chi_A:
                print('\n Iteration is ended with n=%d and chi_A=%4.3e (STOP criterion: n=%d OR chi_A=%4.3e)\n'%(n+1, chi_A, N_iter, TOL_chi_A))
                gen_analysis(z_arr, y_div_R_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out)
                break
        f_log.close()



        

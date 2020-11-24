#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################



import sys
import sol_solvent as PS
import sol_CT as CT
import sol_GT as GT

import osmotic_pressure_CS as CS
import transport_properties_SPHS as PHS
from analysis import *

from numpy import *
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

        Pin = PS.get_Pin(DLP, ref_Pout)                                       # calculating Pin for the given DLP and Pout
        Pper = PS.get_Pper(DLP, ref_DTP, k, ref_Pout)                         # calculating Pper for the given DLP, DTP_linear, k, and P_out

        pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0}
        cond_PS = PS.get_cond(pre_cond, Pin, ref_Pout, Pper)                  # allocating Blank Test (pure test) conditions

        DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper                            # length-averaged TMP with a linearly declined pressure approximation
        vw0 = cond_PS['Lp']*DTP_HP                                         # v^\ast
        epsilon_d = D0/(cond_PS['R']*vw0)                                  # 1/Pe_R


        Pi_arr = zeros(size(phiw_arr))                                     # Set zero osmotic pressure
        Pi_div_DLP_arr = Pi_arr/cond_PS['DLP']

        Gk_tmp = CT.get_Gk(cond_PS['k'], dz_div_L, Pi_div_DLP_arr)
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

            CT.gen_gpm_arr(sign_plus,  z_div_L_arr, Pi_div_DLP_arr, k, gp_arr)
            CT.gen_gpm_arr(sign_minus, z_div_L_arr, Pi_div_DLP_arr, k, gm_arr)
            cond_GT['Gk'] = CT.get_Gk_boost(k, dz_div_L, gp_arr[-1], gm_arr[-1])
            cond_GT['Bp'] = CT.get_Bpm_conv(sign_plus, cond_GT)
            cond_GT['Bm'] = CT.get_Bpm_conv(sign_minus, cond_GT)

            gen_new_phiw_div_phib_arr(N_PROCESSES, phiw_set_2, cond_GT, fcn_Dc_given, fcn_eta_given, z_div_L_arr, phiw_set_1, Pi_div_DLP_arr, cond_GT['weight'], gp_arr, gm_arr, y_div_R_arr, phi_yt_arr, ID_yt_arr, Ieta_yt_arr)

            Pi_arr = fcn_Pi_given(phiw_set_2*phi_b, cond_GT)                              # calculating osmotic pressure for the given phiw
            Pi_div_DLP_arr = Pi_arr/cond_GT['DLP']

            chi_A = print_iteration_info(n, z_div_L_arr, phiw_set_1, phiw_set_2, cond_GT, Pi_div_DLP_arr, gp_arr, gm_arr, f_log)
            if n == N_iter-1 or chi_A < TOL_chi_A:
                print('\n Iteration is ended with n=%d and chi_A=%4.3e (STOP criterion: n=%d OR chi_A=%4.3e)\n'%(n+1, chi_A, N_iter, TOL_chi_A))
                gen_analysis(z_arr, y_div_R_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out)
        f_log.close()




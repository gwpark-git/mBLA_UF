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
    dz = z_arr[1] - z_arr[0]                                           # equi-step size for z
    dr = (1/float(Nr))*R_channel                                       # test step size for r, which will be adjusted in accordance with BLA

    k_B = const.k                                                      # Boltzmann constant
    kT = k_B*T                                                         # thermal energy

    a_H = a_particle*gamma                                             # hydrodynamic radius
    D0 = kT/(6.*pi*eta0*a_H)                                           # Stokes-Einstein-Sutherland
    Va = (4./3.)*pi*a_particle**3.0                                    # volume measure is still using particle exclusion-size

    k = 4.*sqrt(L_channel**2.0 * Lp * eta0 /R_channel**3.0)            # dimensionless parameter k
    prefactor_U = sqrt(Lp*R_channel/eta0)                              # this is related with unit conversion

    print ('k, prefactor_U :', k, prefactor_U)

    Pin = get_Pin(DLP, ref_Pout)                                       # calculating Pin for the given DLP and Pout
    Pper = get_Pper(DLP, ref_DTP, k, ref_Pout)                         # calculating Pper for the given DLP, DTP_linear, k, and P_out

    print ('\nSummary:' )
    print ('Pin, Pper, ref_Pout :', Pin, Pper, ref_Pout)

    pre_cond = {'k':k, 'R':R_channel, 'L':L_channel, 'Lp':Lp, 'eta0':eta0, 'preU':prefactor_U}
    print (pre_cond)

    cond_BT = get_cond(pre_cond, Pin, ref_Pout, Pper)                  # allocating Blank Test (pure test) conditions

    DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper                            # length-averaged TMP with a linearly declined pressure approximation
    vw0 = cond_BT['Lp']*DTP_HP                                         # v^\ast
    epsilon_d = D0/(cond_BT['R']*vw0)                                  # 1/Pe_R
    print (epsilon_d, vw0)

    if IDENT_parallel:                                                 # parallel computation
        if IDENT_modification:
            phiw_update = get_new_phiw_div_phib_modi_arr_parallel
        else:
            phiw_update = get_new_phiw_div_phib_arr_parallel
    else:                                                              # single-process computation
        if IDENT_modification:
            phiw_update = get_new_phiw_div_phib_modi_arr
        else:
            phiw_update = get_new_phiw_div_phib_arr

    Pi_arr = zeros(size(phiw_arr))                                     # calculating osmotic pressure using initial conditions
            
    cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, Pi_arr)     # allocating conditions for the constant transport properties
    cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma) # allocating conditions for the general transport properties

    
    phi_b= cond_GT['phi_bulk']                                         # set the feed/bulk concentration
    phiw_div_phib_arr = phiw_arr/phi_b                                 # reduced wall concentration
    phiw_set_1 = phiw_div_phib_arr                                     # reduced initial wall concentration
    phiw_set_2 = deepcopy(phiw_set_1)                                  # reduced initial wall concentration
    
    gp_arr = zeros(Nz)                                                 # constructing array for g+(z) function
    gm_arr = zeros(Nz)                                                 # constructing array for g-(z) function

    yt_arr = gen_yt_arr(cond_GT)                                       # generating tilde_y with given conditions described in cond_GT
    Ny = size(yt_arr) 
    if IDENT_verbose:
        print ('  IDENT_verbose is turned on, which will record the analysis of result for every FPI steps')
        print ('                The current IDENT_verbose option is not parallelized, which takes longer time for computation')
        fn_ver = fn_out + '.%05d'%(0)
        gen_analysis(z_arr, yt_arr, phiw_set_1*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
        
    for n in range(N_iter):                                                           # main iterator with number n
        phiw_set_1 = deepcopy(phiw_set_2)                                             # reduced wall concentration inherited from the previous iteration
        Pi_arr = fcn_Pi_given(phiw_set_1*phi_b, cond_GT)                              # calculating osmotic pressure for the given phiw
        for i in range(Nz):                                                           # generating g+(z) and g-(z) functions
            gp_arr[i] = get_gpm(z_arr[i], dz, +1.0, Pi_arr, k, cond_BT['L'])
            gm_arr[i] = get_gpm(z_arr[i], dz, -1.0, Pi_arr, k, cond_BT['L'])
        cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, Pi_arr)                # update conditions for CT
        cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, cond_GT['gamma']) # update conditions for GT
        cond_GT['k'] = cond_GT['k'] * eta_div_eta0_SPHS(phi_b, cond_GT)               # update the dimensionless value k

        phiw_set_2= phiw_update(cond_GT, Pi_arr, fcn_Dc_given, fcn_eta_given, \       # main FPI iterator
                                z_arr, phiw_set_1, weight, gp_arr, gm_arr, yt_arr)
        if IDENT_verbose:                                                             # the case when each steps will be printed out
            fn_ver = fn_out + '.%05d'%(n + 1)
            gen_analysis(z_arr, yt_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_ver)
            
        ind_max = argmax(phiw_set_2)                                                  # get index number for the maximum values of phiw(z)
        print ('n=%d, phiw/b(0)=%4.0f, phiw/b(L)=%4.0f, max:(phiw(%4.3f)/b)=%4.0f'%(n, phiw_set_2[0], phiw_set_2[-1], z_arr[ind_max], phiw_set_2[ind_max]))

        err = norm(phiw_set_1 - phiw_set_2)                                           # estimated deviations
        print ('norm(p1-p2) : %4.3e, weight : %4.3f\n'%(err, weight))

    gen_analysis(z_arr, yt_arr, phiw_set_2*phi_b, cond_GT, fcn_Pi_given, fcn_Dc_given, fcn_eta_given, fn_out)

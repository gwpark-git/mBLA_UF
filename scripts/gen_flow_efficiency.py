#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################


from numpy import *
import sys

if len(sys.argv)<3:
    print 'Usage of generating Q_perm and Q_in in SPHS with given gamma'
    print 'argv[1] == gamma'
    print 'argv[2] == output filename'
    print 'argv[3] == ref.py'
else:

    gamma = float(sys.argv[1])
    fn_out = str(sys.argv[2])
    if len(sys.argv)==4:
        fn_ref = str(sys.argv[3])
    else:
        fn_ref = 'ref.py'
    print 'given arguments: ', gamma, fn_out, fn_ref
    
    from scipy.stats import linregress

    path_codes = 'analytic_solution/'

    import sys
    sys.path.append(path_codes)
    from sol_solvent import *


    # generation data for osmotic pressure
    from sol_CT import *
    from osmotic_pressure_CS import *
    from scipy.interpolate import interp1d

    from sol_GT_parallel import *
    from transport_properties_SPHS import *
    from transport_properties_Roa import *


    fcn_eta_given = eta_div_eta0_SPHS
    fcn_Dc_given = Dc_short_div_D0_SPHS

    def get_umax_arr(cond_GT, INT_Pi, INT_phiw, fcn_Dc_given, fcn_eta_given, z_arr):
        Nz = size(z_arr)
        umax_arr = zeros(Nz)

        yt_arr = get_yt_arr(cond_GT)
        Nyt = size(yt_arr)
        phi_test_arr = zeros(Nyt)
        Ieta_test_arr = zeros(Nyt)
    #    ID_test_arr = zeros(Nyt)
        for i in range(Nz):
            z = z_arr[i]
            vw_div_vw0_test = get_vw_GT(z, cond_GT, INT_Pi, INT_phiw)/cond_GT['vw0']
            get_phi_with_fixed_z_GT(z, cond_GT, INT_phiw, INT_Pi, fcn_Dc_given, vw_div_vw0_test, yt_arr, phi_test_arr)
            get_int_eta_phi(z, cond_GT, INT_phiw, INT_Pi, fcn_Dc_given, fcn_eta_given, yt_arr, phi_test_arr, Ieta_test_arr)
            umax_arr[i] = get_u_center_GT(z, cond_GT, INT_phiw, INT_Pi, fcn_Dc_given, fcn_eta_given, phi_test_arr[-1], Ieta_test_arr[-1])
        return umax_arr

    def get_vw_arr(cond_GT, INT_Pi, INT_phiw, z_arr):
        Nz = size(z_arr)
        vw_arr = zeros(Nz)

        for i in range(Nz):
            z = z_arr[i]
            vw_arr[i] = get_vw_GT(z, cond_GT, INT_Pi, INT_phiw)
        return vw_arr

    def get_P_arr(cond_GT, INT_Pi, INT_phiw, z_arr):
        Nz = size(z_arr)
        P_arr = zeros(Nz)

        for i in range(Nz):
            z = z_arr[i]
            P_arr[i] = get_P_GT(0, z, cond_GT, INT_Pi)
        return P_arr


    def get_DTP(cond_GT, INT_Pi, Nz):
        z_arr = linspace(0, cond_GT['L'], Nz)
        dz = cond_GT['L']/float(Nz)
    #    P_arr = zeros(Nz)
        re = 0.
        for i in range(1, Nz):
            f1 = get_P_CT(0, z_arr[i-1], cond_GT, dz, INT_Pi) - cond_GT['Pper']        
            f2 = get_P_CT(0, z_arr[i], cond_GT, dz, INT_Pi) - cond_GT['Pper']
            re += 0.5*dz*(f1 + f2)
        return re/cond_GT['L']

    def get_yt_arr(cond_GT):
        phi_b = cond_GT['phi_bulk']
        ed = cond_GT['epsilon_d']
        dr = cond_GT['dr']
        R = cond_GT['R']; L = cond_GT['L']
        dyt = dr/R
        dyp = ed*dyt
    #    print dyp, dyt
        tmp_yt = 0.
        yt_arr = [tmp_yt]
        while(tmp_yt < 1. -dyt):
            if tmp_yt < ed:
                tmp_dy = dyp
            elif tmp_yt < 2.*ed:
                tmp_dy = 2.*dyp
            elif tmp_yt < 10.*ed:
                tmp_dy = 10.*dyp
            else:
                tmp_dy = dyt
            tmp_yt += tmp_dy
            yt_arr.append(tmp_yt)
        yt_arr = asarray(yt_arr)
        return yt_arr

    # system properties
    # ref_DTP_arr = asarray([100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    ref_DTP_arr = asarray([200, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750])
    
    Q_in_arr = []
    Q_perm_arr = []

    for ref_DTP in ref_DTP_arr:
        try:
            execfile(path_codes+fn_ref)

            # pre condition generation
            print k, prefactor_U

            ref_Pout = 101325 #Pa

            DLP = 130 # Pa
            Pin = get_Pin(DLP, ref_Pout)
            Pper = get_Pper(DLP, ref_DTP, k, ref_Pout)
            #dz = L_channel/1000.

            print '\nSummary:' 
            print Pin
            print Pper
            print ref_Pout

            cond_BT = get_cond(pre_cond, Pin, ref_Pout, Pper)
            print cond_BT
            #cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, INT_Pi_BLA)

            # parameter related with sol_CT

            D0 = kT/(6.*pi*cond_BT['eta0']*a_particle)
            DTP_HP = (1/2.)*(Pin + ref_Pout) - Pper
            vw0 = cond_BT['Lp']*DTP_HP
            epsilon_d = D0/(cond_BT['R']*vw0)
            print epsilon_d, vw0

            # parameters related with sol_GT
            phi_bulk = 1e-3
            epsilon_d = D0/(R_channel*vw0)
            dr = (1/200.)*R_channel
            dz = (1/100.)*L_channel

            Nz = 100
            z_arr = linspace(0, L_channel, Nz)
            phiw_arr = ones(Nz)*phi_bulk # initial condition
            Pi_arr = get_Pi(phiw_arr, Va, kT)

            INT_phiw = interp1d(z_arr, phiw_arr)
            INT_Pi = interp1d(z_arr, Pi_arr)

            cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, INT_Pi)
            cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma)
            import copy
            phi_b= cond_GT['phi_bulk']
            phiw_set_1 = ones(Nz)*phi_b
            phiw_set_1[0] = phi_bulk
            phiw_set_2 = copy.deepcopy(phiw_set_1)
            phiw_rec = []
            weight = 0.1
            k_new = cond_GT['k']

            N_iter = 100
            for n in range(N_iter):
                phiw_set_1 = copy.deepcopy(phiw_set_2)
                print 'n=%d, phiw(n-1)(0)=%4.3f, phiw(n-1)(L)=%4.3f, k_new=%4.3f'%(n, phiw_set_1[0], phiw_set_1[-1], k_new)
                phiw_rec.append(1.*phiw_set_1)    
                INT_phiw = interp1d(z_arr, phiw_set_1, fill_value='extrapolate')
                INT_Pi = interp1d(z_arr, get_Pi(phiw_set_1, Va, kT), fill_value='extrapolate')
                cond_CT = get_cond_CT(cond_BT, a_particle, Va, kT, dz, INT_Pi)
                cond_GT = get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, cond_GT['gamma'])
                phiw_set_2= get_new_phiw_arr(cond_GT, INT_phiw, INT_Pi, fcn_Dc_given, fcn_eta_given, z_arr, phiw_set_1, weight)

                print
            vw_SPHS_arr = get_vw_arr(cond_GT, INT_Pi, INT_phiw, z_arr)
            umax_SPHS_arr = get_umax_arr(cond_GT, INT_Pi, INT_phiw, fcn_Dc_given, fcn_eta_given, z_arr)

            Q_in_SPHS = (pi/2.0)*R_channel**2.0 * umax_SPHS_arr[0]
            Q_perm_SPHS = 0.
            for i in range(1, size(z_arr)):
                dz = z_arr[i] - z_arr[i-1]
                Q_perm_SPHS += 0.5*dz*2.*pi*R_channel*(vw_SPHS_arr[i-1] + vw_SPHS_arr[i])
            Q_in_arr.append(Q_in_SPHS)
            Q_perm_arr.append(Q_perm_SPHS)
        except:
            break

    re_Q = zeros([size(ref_DTP_arr), 3])
    re_Q[:,0] = ref_DTP_arr
    re_Q[:,1] = asarray(Q_in_arr)
    re_Q[:,2] = asarray(Q_perm_arr)
    savetxt(fn_out, re_Q)    


# import sys
from analysis import *

# def gen_radial_J(phiw_z, vw_z, 

def gen_cond_GT(fn_inp, given_phiw):
    print(fn_inp)
    exec(open(fn_inp).read())
    print(L_channel)

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
    # phiw_set_1 = phiw_div_phib_arr                                     # reduced initial wall concentration
    # phiw_set_2 = deepcopy(phiw_set_1)                                  # reduced initial wall concentration
    phiw_set_1 = given_phiw/phi_b

    Pi_arr = fcn_Pi_given(phiw_set_1*phi_b, cond_GT)                              # calculating osmotic pressure for the given phiw
    Pi_div_DLP_arr = Pi_arr/cond_GT['DLP']


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



    cond_GT['Gk'] = CT.get_Gk_boost(k, dz_div_L, gp_arr[-1], gm_arr[-1], cond_GT['denom_Gk_BC_specific'])
    cond_GT['Bp'] = CT.get_Bpm_conv(sign_plus, cond_GT)
    cond_GT['Bm'] = CT.get_Bpm_conv(sign_minus, cond_GT)

    return cond_GT

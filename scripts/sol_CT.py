#############################################################################
#   Concentration and flow profile of suspension flow in CF UF              #
#   Using constant diffusivity D=D_0 and viscosity eta = eta_s              #
#                                                                           #
#   Note:                                                                   #
#   it is possible to give particle-contributed osmotic pressure            #
#   if Pi=0 regardless the concentration, this is CT0 case (see table II).  #
#   for non-zero Pi, Pi/DLP must be provided for calculating vw             #
#   in accordance with Darcy-Starling law                                   #
#                                                                           #
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
#############################################################################



from numpy import *
import sol_solvent as PS

def get_cond(cond_PS, phi_bulk, a_colloid, a_hydrodynamic, Va, kT, dz, Gk):
    """ Update condition dictionary inherited from cond_PS (defined by get_cond in sol_solvent.py)
        CT stands for the constant transport properties with particle-contributed osmotic pressure (See Tab. II)
        According to the paper, CT0 is when diffusivity and viscosity are constants, and Pi=0.
        This CT0 case gives the explicit solution of phi_w(z) in accordance with Eq. (52).
        Therefore, CT0 DO NOT require the FPI iterator.
        In the general CT, however, the particle-contributed osmotic pressure Pi(phi_w(z)) make Eq. (52) as the implicit equation.
        Therefore, CT DO require the FPI iterator.
        For these reason, this sol_CT.py package is designed to applicable for the general CT case for the given Pi,
        which, however, the iterator is not included. Such FPI iterator (Eq. (D4)) is defined on sol_GT.py.
        This is also the reason that the input parameters 'weight' is not necessary on here.

    """
    if (cond_PS['COND'] != 'PS'):
        print('Error: inherit non-PS type of dictionary in get_cond_CT is not supported.')
    
    COND_TYPE = 'CT'
    
    re         = cond_PS.copy()
    re['COND'] = COND_TYPE                                                             # update the given type to CT (constant transport properties)
    re['phi_bulk'] = phi_bulk
    re['a']    = a_colloid                                                             # hard-core radius of particles in the unit of m
    re['a_H']  = a_hydrodynamic
    re['gamma']= a_hydrodynamic/a_colloid
    re['Va']   = Va                                                                    # particle volume in the unit of m^3
    re['kT']   = kT                                                                    # thermal energy in the unit of J
    re['D0']   = kT/(6.*pi*re['eta0']*a_hydrodynamic) # Stokes-Einstein-Sutherland diffusion coefficient in Eq. (14)

    re['Phi_ast'] = pi*re['R']**2.0 * re['phi_bulk'] * re['u_HP']
    
    re['Pe_R'] = re['R']*re['vw0']/re['D0'] # radial Peclet number from Eq. (37)
    re['epsilon_d'] = 1./re['Pe_R'] # selected perturbation constant using Eq. (41)
    re['dz']   = dz                                                                    # step size for z in the unit of m
    re['Gk']   = Gk # correction factor for Bpm in the dimensionless unit

    sign_plus = +1.
    sign_minus = -1.
    
    re['Bp']   = get_Bpm_conv(sign_plus, re)
    re['Bm']   = get_Bpm_conv(sign_minus, re)

    return re


def get_gpm(pm, z_div_L, dz_div_L, Pi_div_DLP_arr, k):
    """ [Overhead version] Using expresion for gpm(zt=1) in Eq. (46)
    This is an overhead version for computing gp and gm since it require to integrate over z from 0 to given z value.
    This function should be used only for the purpose of the final analysis or other instances.
    The optimized version to calculate gp and gm for all discretized z is defined in gen_gpm_arr
    """
    re = 0.
    for i in range(1, int(round(z_div_L/dz_div_L))):
        zt_tmp_1 = (i-1)*dz_div_L
        zt_tmp_2 = zt_tmp_1 + dz_div_L
        y1 = exp(pm * k * zt_tmp_1)*Pi_div_DLP_arr[i-1]
        y2 = exp(pm * k * zt_tmp_2)*Pi_div_DLP_arr[i]
        re += 0.5 * dz_div_L * (y1 + y2)
    return pm * re * k/2.


def gen_gpm_arr(pm, z_div_L_arr, Pi_div_DLP_arr, k, gpm_arr):
    """ Generation gpm_arr based on Eq. (46)
    This is optimized version for get_gpm function.
    gen_gpm_arr require array for (dimensionless) z and Pi, then taking numerical integration.
    The step size is automatically identified by dz = z_i - z_{i-1}, so the adaptable step size could be appplied unlike get_gpm.
    Note, however, the other functions may not support the adaptive dz (unlike adaptive dr because of the boundary layer analysis).
    """
    re = 0.
    gpm_arr[0] = re
    for i in range(1, size(z_div_L_arr)):
        zt_tmp_1 = z_div_L_arr[i-1]
        zt_tmp_2 = z_div_L_arr[i]
        dz_div_L = zt_tmp_2 - zt_tmp_1
        
        y1 = exp(pm * k * zt_tmp_1)*Pi_div_DLP_arr[i-1]
        y2 = exp(pm * k * zt_tmp_2)*Pi_div_DLP_arr[i]
        re += 0.5 * dz_div_L * (y1 + y2)
        gpm_arr[i] = pm * re * k/2.
    return 0

def get_Gk(k, dz_div_L, Pi_div_DLP_arr):
    """ [Overhead version] Using expression "Gk = -(Bp - Ap) = Bm - Am" in Eq. (46)
    Note that Gk is sign independent (pm)

    This is an overhead version for computing Gk because it needs gp1 and gm1 (defined inside function)
    which takes the numerical integration from z=0 to z=given z.
    Therefore, it would be recommendable to use get_Gk_boost where it gives the values of gp1 and gm1 as parameters.
    """
    sign_plus = +1.
    sign_minus = -1.

    L_div_L = 1.
    
    gp1 = get_gpm(sign_plus, L_div_L, dz_div_L, Pi_div_DLP_arr, k)
    gm1 = get_gpm(sign_minus, L_div_L, dz_div_L, Pi_div_DLP_arr, k)

    return (gp1 * exp(-k) + gm1 * exp(k))/(2.*sinh(k))


def get_Gk_boost(k, dz_div_L, gp1, gm1):
    """ Using expression "Gk = -(Bp - Ap) = Bm - Am" in Eq. (46)
    Note that Gk is sign independent (pm).

    Because get_Gk_boost takes parameters of gp1 and gm1, this function do not require other parameters.
    """
    return (gp1 * exp(-k) + gm1 * exp(k))/(2.*sinh(k))


def get_Bpm(pm, k, alpha_ast, Gk):
    """ [Overhead version] Using expression for Bpm in Eq. (47)
    This function recalculate Apm.
    """
    return PS.get_Apm(pm, k, alpha_ast) - pm * Gk

def get_Bpm_conv(pm, cond_GT):
    """ Using expression for Bpm in Eq. (47)
    cond_GT must stored the proper values of 'Ap' and 'Am', and 'Gk'.
    Otherwise, this convenient function will not properly work.
    """
    if pm > 0: # for Bp
        return cond_GT['Ap'] - cond_GT['Gk']
    # for Bm
    return cond_GT['Am'] + cond_GT['Gk']


def get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm):
    """ Pout expression in Eq. (45)
    because P = Pout in Eq. (49)        
    """
    return Pper_div_DLP\
        + exp(k*z_div_L)*(Bp + gm)\
        + exp(-k*z_div_L)*(Bm + gp)

def get_P_conv(r_div_R, z_div_L, cond_CT, gp, gm):
    return get_P(r_div_R, z_div_L, cond_CT['Pper_div_DLP'], cond_CT['k'], cond_CT['Bp'], cond_CT['Bm'], gp, gm)

def get_u(r_div_R, z_div_L, k, Bp, Bm, gp, gm):
    """ Using u^out expression in Eq. (45)
    because matched asymptotic u in Eq. (49) with constant transport properties
    give the same expression as u^out in Eq. (45)
    """
    uR_HP = 1. - r_div_R**2.0
    uZ_out = -k*(exp( k*z_div_L)*(Bp + gm) \
                 -exp(-k*z_div_L)*(Bm + gp))
    return uZ_out*uR_HP

def get_u_conv(r_div_R, z_div_L, cond_CT, gp, gm):
    return get_u(r_div_R, z_div_L, cond_CT['k'], cond_CT['Bp'], cond_CT['Bm'], gp, gm)

def get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm):
    """ Using v^out expression in Eq. (45)
    One must careful about coordinate function r_div_R in comparison with y_div_R
    because the direction is reversed, which means the minus sign in Eq. (45) for v^out
    should be reversed to plus sign since we use r_div_R in this code.
    This will be clear if we compare v^out(y,z) in Eq. (45) with v(r,z) in Eq. 31 for pure solvent flow
    This aspect reflected in sign = +1 for the explicit.
    """
    sign = +1.
    vR = 2.*r_div_R - r_div_R**3.0
    vw =(exp( k*z_div_L)*(Bp + gm)\
         +exp(-k*z_div_L)*(Bm + gp)\
         -Pi_div_DLP)/alpha_ast
    return sign*vR*vw

def get_v_conv(r_div_R, z_div_L, Pi_div_DLP, cond_CT, gp, gm):
    return get_v(r_div_R, z_div_L, Pi_div_DLP, cond_CT['k'], cond_CT['alpha_ast'], cond_CT['Bp'], cond_CT['Bm'], gp, gm)


def get_phi(r_div_R, z_div_L, vw_div_vw0, epsilon_d, phiw, phi_bulk):
    """ matched asymptotic phi for constant transport properties (CT) using Eqs. (43) and (49) 
    With CT, s_bar is simplified by vw_div_vw0*y_bar.
    """
    s_bar = vw_div_vw0*(1. - r_div_R)/epsilon_d
    return (phiw - phi_bulk)*exp(-s_bar) + phi_b*(1. - s_bar*exp(-s_bar))

def get_phi_conv(r_div_R, z_div_L, vw_div_vw0, phiw, cond_CT):
    return get_phi(r_div_R, z_div_L, vw_div_vw0, cond_CT['epsilon_d'], phiw, cond_CT['phi_bulk'])

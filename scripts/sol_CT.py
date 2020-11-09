#############################################################################
#   Concentration and flow profile of suspension flow in CF UF              #
#   Using constant diffusivity D=D_0 and viscosity eta = eta_s              #
#                                                                           #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#                                                                           #
#   Note:                                                                   #
#   it is possible to give particle-contributed osmotic pressure            #
#   if Pi=0 regardless the concentration, this is CT0 case (see table II).  #
#   for non-zero Pi, one must given by the value Pi/DLP for calculating vw  #
#   in accordance with Darcy-Starling law                                   #
#   without particle-contributed osmotic pressure Pi=0                      #
#   This is the case of CT0 denoted in the paper (see Table II).            #
#############################################################################


from numpy import *
import sol_solvent as PS

def get_cond(cond_PS, a_colloid, Va, kT, dz, Pi_arr, Gk):
    if (cond_PS['COND'] != 'PS'):
        print('Error: inherit non-PS type of dictionary in get_cond_CT is not supported.')
    
    COND_TYPE = 'CT'
    
    re         = cond_PS.copy()
    re['COND'] = COND_TYPE                                                             # update the given type to CT (constant transport properties)
    re['a']    = a_colloid                                                             # hard-core radius of particles in the unit of m
    re['Va']   = Va                                                                    # particle volume in the unit of m^3
    re['kT']   = kT                                                                    # thermal energy in the unit of J
    re['dz']   = dz                                                                    # step size for z in the unit of m
    re['Gk']   = Gk # correction factor for Bpm in the dimensionless unit
    re['Bp']   = get_Bpm(+1.0, re['k'], re['alpha_ast'], re['Pper_div_DLP'], re['Gk']) # calculated Bp in the dimensionless unit
    re['Bm']   = get_Bpm(-1.0, re['k'], re['alpha_ast'], re['Pper_div_DLP'], re['Gk']) # calculate Bm in the dimensionless unit

    # re['Gk']   = get_Gk(re['k'], re['dz']/re['L'], Pi_arr/re['DLP'])                   # correction factor for Bpm in the dimensionless unit
    # re['Bp']   = get_Bpm(+1.0, re['k'], re['alpha_ast'], re['Pper_div_DLP'], re['Gk']) # calculated Bp in the dimensionless unit
    # re['Bm']   = get_Bpm(-1.0, re['k'], re['alpha_ast'], re['Pper_div_DLP'], re['Gk']) # calculate Bm in the dimensionless unit

    print ('Gk = ', re['Gk'])
    print ('Bp, Bm : ', re['Bp'], re['Bm'])
    return re


def get_gpm(pm, z_div_L, dz_div_L, Pi_div_DLP_arr, k):
    """ Using expresion for gpm(zt=1) in Eq. (46)
    Note: This contains a lot of overhead since it will take a numerical integration from z=0 to z=given value.
        Therefore, it is recommendable to use other function: gen_gpm_arr
    """
    re = 0.
    for i in range(0, int(round(z_div_L/dz_div_L)) - 1):
        zt_tmp_1 = i*dz_div_L
        zt_tmp_2 = zt_tmp_1 + dz_div_L
        y1 = exp(pm * k * zt_tmp_1)*Pi_div_DLP_arr[i]
        y2 = exp(pm * k * zt_tmp_2)*Pi_div_DLP_arr[i+1]
        re += 0.5 * dz_div_L * (y1 + y2)
    return pm*re*k/2.

def get_Gk(k, dz_div_L, Pi_div_DLP_arr):
    """ Using expression for part of Bpm (Bpm - Apm) in Eq. (46)
    """
    gp1 = get_gpm(+1., 1., dz_div_L, Pi_div_DLP_arr, k)
    gm1 = get_gpm(-1., 1., dz_div_L, Pi_div_DLP_arr, k)
    print ('Gpt(1), Gmt(1) = ', gp1, gm1)
    return (gm1 * exp(k) + gp1 * exp(-k))/(2.*sinh(k))

def gen_gpm_arr(pm, z_div_L_arr, dz_div_L, Pi_div_DLP_arr, k, gpm_arr):
    """ Generation gpm_arr based on Eq. (46)
    """
    re = 0.
    gpm_arr[0] = re
    # for i in range(0, int(round(z_div_L/dz_div_L)) - 1):
    for i in range(1, size(z_div_L_arr)):
        zt_tmp_1 = (i-1)*dz_div_L
        zt_tmp_2 = zt_tmp_1 + dz_div_L
        y1 = exp(pm * k * zt_tmp_1)*Pi_div_DLP_arr[i-1]
        y2 = exp(pm * k * zt_tmp_2)*Pi_div_DLP_arr[i]
        re += 0.5 * dz_div_L * (y1 + y2)
        gpm_arr[i] = pm * re * k/2.
    return 0
    # return pm*re*k/2.

def get_Gk_boost(k, dz_div_L, Pi_div_DLP_arr, gp1, gm1):
    """ Using expression for part of Bpm (Bpm - Apm) in Eq. (46)
    """
    # gp1 = get_gpm(+1., 1., dz_div_L, Pi_div_DLP_arr, k)
    # gm1 = get_gpm(-1., 1., dz_div_L, Pi_div_DLP_arr, k)
    print ('Gpt(1), Gmt(1) = ', gp1, gm1)
    return (gm1 * exp(k) + gp1 * exp(-k))/(2.*sinh(k))


def get_Bpm(pm, k, alpha_ast, Pper_div_DLP, Gk):
    """ Using expression for Bpm in Eq. (47)
    """
    return PS.get_Apm(pm, k, alpha_ast, Pper_div_DLP) - pm * Gk


def get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm):
    """ Pout expression in Eq. (45)
    because P = Pout in Eq. (49)        
    """
    return Pper_div_DLP\
        + exp(k*z_div_L)*(Bp + (k/2.)*gm)\
        + exp(-k*z_div_L)*(Bm + (k/2.)*gp)

def get_P_conv(r_div_R, z_div_L, cond_CT, gp, gm):
    return get_P(r_div_R, z_div_L, cond_CT['Pper_div_DLP'], cond_CT['k'], cond_CT['Bp'], cond_CT['Bm'], gp, gm)

def get_u(r_div_R, z_div_L, k, Bp, Bm, gp, gm):
    """ Using u^out expression in Eq. (45)
    because matched asymptotic u in Eq. (49) with constant transport properties
    give the same expression as u^out in Eq. (45)
    """
    uR_HP = 1. - r_div_R**2.0
    uZ_out = -k*(exp( k*z_div_L)*(Bp + (k/2.)*gm) \
                 -exp(-k*z_div_L)*(Bm + (k/2.)*gp))
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
    vw =(exp( k*z_div_L)*(Bp + (k/2.)*gm)\
         +exp(-k*z_div_L)*(Bm + (k/2.)*gp)\
         -Pi_div_DLP)/alpha_ast
    return sign*vR*vw

def get_v_conv(r_div_R, z_div_L, Pi_div_DLP, cond_CT, gp, gm):
    return get_v(r_div_R, z_div_L, Pi_div_DLP, cond_CT['k'], cond_CT['alpha_ast'], cond_CT['Bp'], cond_CT['Bm'], gp, gm)


# def get_P_CT(r_div_R, z_div_L, cond_CT, gp, gm):
#     """ Pout expression in Eq. (45)
#     because P = Pout in Eq. (49)        
#     """
#     return cond_CT['Pper_div_DLP'] + \
#         exp(cond_CT['k']*z_div_L)*(cond_CT['Bp'] + (cond_CT['k']/2.)*gm)\
#         + exp(-cond_CT['k']*z_div_L)*(cond_CT['Bm'] + (cond_CT['k']/2.)*gp)

# def get_u_CT(r_div_R, z_div_L, cond_CT, gp, gm):
#     """ Using u^out expression in Eq. (45)
#     because matched asymptotic u in Eq. (49) with constant transport properties
#     give the same expression as u^out in Eq. (45)
#     """
#     uR_HP = 1. - r_div_R**2.0
#     uZ_out = -k*(exp( cond_CT['k']*z_div_L)*(cond_CT['Bp'] + (cond_CT['k']/2.)*gm) \
#                  -exp(-cond_CT['k']*z_div_L)*(cond_CT['Bm'] + (cond_CT['k']/2.)*gp))
#     return uZ_out*uR_HP

# def get_v_CT(r_div_R, z_div_L, cond_CT, Pi_div_DLP, gp, gm):
#     """ Using v^out expression in Eq. (45)
#     One must careful about coordinate function r_div_R in comparison with y_div_R
#     because the direction is reversed, which means the minus sign in Eq. (45) for v^out
#     should be reversed to plus sign since we use r_div_R in this code.
#     This aspect reflected in sign = +1 for the explicit.
#     """
#     sign = +1.
#     vR = 2.*r_div_R - r_div_R**3.0
#     vw =( exp( cond_CT['k']*z_div_L)*(cond_CT['Bp'] + (cond_CT['k']/2.)*gm)\
#           +exp(-cond_CT['k']*z_div_L)*(cond_CT['Bm'] + (cond_CT['k']/2.)*gp)\
#           -Pi_div_DLP)/cond_CT['alpha_ast']
#     return sign*vR*vw

    
    
    # [WIP]
    # return -cond_CT['preU']*uR*uZ

# def get_gpm(z, dz, pm, Pi_arr, k, L):
#     re = 0.
#     for i in range(0, int(round(z/dz))-1):
#         z_tmp = i*dz
#         y1 = exp(pm * k * z_tmp/L)*Pi_arr[i]
#         y2 = exp(pm * k * (z_tmp + dz)/L)*Pi_arr[i+1]
#         re += 0.5*dz*(y1 + y2)
#     re *= pm/L # L must be divided since the integration take over a dimensional z rather than dimensionless z in the digital note.
#     return re

# def get_gpm_backward(z, dz, pm, INT_Pi, k, L):
#     re = 0.
#     # for z_tmp in arange(0, z , dz):
#     for z_tmp in arange(z, L - dz, dz):
#         y1 = exp(pm * k * z_tmp/L)*INT_Pi(z_tmp)
#         y2 = exp(pm * k * (z_tmp + dz)/L)*INT_Pi(z_tmp + dz)
#         re += 0.5*dz*(y1 + y2)
#     # if (z==L):
#     #     print 'get_gpm(L)=', pm*re, ' with ', dz, pm, k, L
#     re *= pm/L # L must be divided since the integration take over a dimensional z rather than dimensionless z in the digital note.
#     return -re

# def get_Gk_CT(k, dz, Pi_arr, L):
#     gp1 = get_gpm(L, dz, +1.0, Pi_arr, k, L)
#     gm1 = get_gpm(L, dz, -1.0, Pi_arr, k, L)

#     # gp1 = get_gpm(0, dz, +1.0, INT_Pi, k, L)
#     # gm1 = get_gpm(0, dz, -1.0, INT_Pi, k, L)
    
#     print ('Gp(L), Gm(L) = ', gp1, gm1)
#     # return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))
#     return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))

# def get_Gk_CT_backward(k, dz, INT_Pi, L):
#     gp1 = get_gpm_backward(0, dz, +1.0, INT_Pi, k, L)
#     gm1 = get_gpm_backward(0, dz, -1.0, INT_Pi, k, L)
#     # gp1 = 0
#     # gm1 = 0
#     # gp1 = get_gpm(0, dz, +1.0, INT_Pi, k, L)
#     # gm1 = get_gpm(0, dz, -1.0, INT_Pi, k, L)
    
#     print 'Gp(L), Gm(L) = ', gp1, gm1
#     # return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))
#     return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))

# def get_Bpm_backward(k, pm, P_in, P_out, P_per, dz, INT_Pi, L):
#     gp0 = get_gpm_backward(0, dz, +1.0, INT_Pi, k, L)
#     gm0 = get_gpm_backward(0, dz, -1.0, INT_Pi, k, L)
#     return get_Apm(k, pm, P_in, P_out, P_per) + pm * (k/sinh(k))*(gp0 + gm0)*(1. - (1. + pm)*sinh(k))

# def get_Bpm(k, pm, P_in, P_out, P_per, dz, Pi_arr, L, Gk):
#     return PS.get_Apm(k, pm, P_in, P_out, P_per) - pm * Gk


# def get_cond_CT_backward(cond_PS, a_colloid, Va, kT, dz, INT_Pi):
#     re = cond_PS.copy()
#     re['a']=a_colloid
#     re['Va']=Va
#     re['kT']=kT
#     re['dz'] = dz
# #    re['Gk'] = get_Gk_CT(re['k'], re['dz'], INT_Pi, re['L'])
# #    print 'Gk = ', re['Gk']
#     re['Bp'] = get_Bpm_backward(re['k'], +1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], INT_Pi, re['L'])
#     re['Bm'] = get_Bpm_backward(re['k'], -1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], INT_Pi, re['L'])
#     print 'Bp, Bm : ', re['Bp'], re['Bm']
#     return re


# def get_P_CT(r, z, cond_CT, dz, Pi_arr):
#     k=cond_CT['k']; Pper = cond_CT['Pper'];
#     Ap = cond_CT['Bp']; Am = cond_CT['Bm']; L = cond_CT['L']
    
#     gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
#     gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)

#     return get_P_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

# def get_u_CT(r, z, cond_CT, dz, Pi_arr):
#     k=cond_CT['k']; Pper = cond_CT['Pper']; Ap = cond_CT['Bp']; Am = cond_CT['Bm']
#     L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']
    
#     gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
#     gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
#     return get_u_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

# def get_v_CT(r, z, cond_CT, dz, Pi_arr):
#     k=cond_CT['k']; Pper = cond_CT['Pper']; Ap = cond_CT['Bp']; Am = cond_CT['Bm']
#     L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']; Lp=cond_CT['Lp']    
#     gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
#     gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
#     return get_v_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

# def get_P_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
#     k=cond_CT['k']; Pper = cond_CT['Pper'];
#     Ap = cond_CT['Bp']; Am = cond_CT['Bm']; L = cond_CT['L']
    
#     return Pper                         \
#         + exp( k*z/L)*(Ap + (k/2.)*gm)  \
#         + exp(-k*z/L)*(Am + (k/2.)*gp)


# def get_u_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
#     k=cond_CT['k']; Pper = cond_CT['Pper']; Ap = cond_CT['Bp']; Am = cond_CT['Bm']
#     L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']
    
#     uR = 1. - (r/R)**2.0
    
#     uZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
#          - exp(-k*z/L)*(Am + (k/2.)*gp)

#     return -preU * uR * uZ 


# def get_v_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
#     k=cond_CT['k']; Pper = cond_CT['Pper']; Ap = cond_CT['Bp']; Am = cond_CT['Bm']
#     L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']; Lp=cond_CT['Lp']    
#     i_tmp = int(round(z/dz))
#     vR = 2.*(r/R) - (r/R)**3.0
#     vZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
#          + exp(-k*z/L)*(Am + (k/2.)*gp) \
#          - Pi_arr[i_tmp]
#          # - INT_Pi(z)
#     return Lp * vR * vZ





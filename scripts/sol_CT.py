#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################

# this is functions related with outer solution

from numpy import *
from sol_solvent import *

def get_gpm(z, dz, pm, Pi_arr, k, L):
    re = 0.
    for i in range(0, int(round(z/dz))-1):
        z_tmp = i*dz
        y1 = exp(pm * k * z_tmp/L)*Pi_arr[i]
        y2 = exp(pm * k * (z_tmp + dz)/L)*Pi_arr[i+1]
        re += 0.5*dz*(y1 + y2)
    re *= pm/L # L must be divided since the integration take over a dimensional z rather than dimensionless z in the digital note.
    return re

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

def get_Gk_CT(k, dz, Pi_arr, L):
    gp1 = get_gpm(L, dz, +1.0, Pi_arr, k, L)
    gm1 = get_gpm(L, dz, -1.0, Pi_arr, k, L)

    # gp1 = get_gpm(0, dz, +1.0, INT_Pi, k, L)
    # gm1 = get_gpm(0, dz, -1.0, INT_Pi, k, L)
    
    print ('Gp(L), Gm(L) = ', gp1, gm1)
    # return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))
    return (gm1 * exp(k) + gp1 * exp(-k)) * k/(4.*sinh(k))

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

# def get_Cpm_CT_backward(k, pm, P_in, P_out, P_per, dz, INT_Pi, L):
#     gp0 = get_gpm_backward(0, dz, +1.0, INT_Pi, k, L)
#     gm0 = get_gpm_backward(0, dz, -1.0, INT_Pi, k, L)
#     return get_Cpm(k, pm, P_in, P_out, P_per) + pm * (k/sinh(k))*(gp0 + gm0)*(1. - (1. + pm)*sinh(k))

def get_Cpm_CT(k, pm, P_in, P_out, P_per, dz, Pi_arr, L, Gk):
    return get_Cpm(k, pm, P_in, P_out, P_per) - pm * Gk

def get_cond_CT(cond_BT, a_colloid, V_a, kT, dz, Pi_arr):
    re = cond_BT.copy()
    re['a']=a_colloid
    re['Va']=V_a
    re['kT']=kT
    re['dz'] = dz
    re['Gk'] = get_Gk_CT(re['k'], re['dz'], Pi_arr, re['L'])
    print ('Gk = ', re['Gk'])
    re['Cp_CT'] = get_Cpm_CT(re['k'], +1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], Pi_arr, re['L'], re['Gk'])
    re['Cm_CT'] = get_Cpm_CT(re['k'], -1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], Pi_arr, re['L'], re['Gk'])
    print ('Cp_CT, Cm_CT : ', re['Cp_CT'], re['Cm_CT'])
    return re

# def get_cond_CT_backward(cond_BT, a_colloid, V_a, kT, dz, INT_Pi):
#     re = cond_BT.copy()
#     re['a']=a_colloid
#     re['Va']=V_a
#     re['kT']=kT
#     re['dz'] = dz
# #    re['Gk'] = get_Gk_CT(re['k'], re['dz'], INT_Pi, re['L'])
# #    print 'Gk = ', re['Gk']
#     re['Cp_CT'] = get_Cpm_CT_backward(re['k'], +1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], INT_Pi, re['L'])
#     re['Cm_CT'] = get_Cpm_CT_backward(re['k'], -1.0, re['Pin'], re['Pout'], re['Pper'], re['dz'], INT_Pi, re['L'])
#     print 'Cp_CT, Cm_CT : ', re['Cp_CT'], re['Cm_CT']
#     return re


def get_P_CT(r, z, cond_CT, dz, Pi_arr):
    k=cond_CT['k']; Pper = cond_CT['Pper'];
    Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']; L = cond_CT['L']
    
    gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
    gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)

    return get_P_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

def get_u_CT(r, z, cond_CT, dz, Pi_arr):
    k=cond_CT['k']; Pper = cond_CT['Pper']; Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']
    L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']
    
    gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
    gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    return get_u_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

def get_v_CT(r, z, cond_CT, dz, Pi_arr):
    k=cond_CT['k']; Pper = cond_CT['Pper']; Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']
    L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']; Lp=cond_CT['Lp']    
    gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
    gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    return get_v_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm)

def get_P_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
    k=cond_CT['k']; Pper = cond_CT['Pper'];
    Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']; L = cond_CT['L']
    
    return Pper                         \
        + exp( k*z/L)*(Cp + (k/2.)*gm)  \
        + exp(-k*z/L)*(Cm + (k/2.)*gp)


def get_u_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
    k=cond_CT['k']; Pper = cond_CT['Pper']; Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']
    L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']
    
    uR = 1. - (r/R)**2.0
    
    uZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         - exp(-k*z/L)*(Cm + (k/2.)*gp)

    return -preU * uR * uZ 


def get_v_CT_boost(r, z, cond_CT, dz, Pi_arr, gp, gm):
    k=cond_CT['k']; Pper = cond_CT['Pper']; Cp = cond_CT['Cp_CT']; Cm = cond_CT['Cm_CT']
    L = cond_CT['L']; R = cond_CT['R']; preU=cond_CT['preU']; Lp=cond_CT['Lp']    
    i_tmp = int(round(z/dz))
    vR = 2.*(r/R) - (r/R)**3.0
    vZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         + exp(-k*z/L)*(Cm + (k/2.)*gp) \
         - Pi_arr[i_tmp]
         # - INT_Pi(z)
    return Lp * vR * vZ





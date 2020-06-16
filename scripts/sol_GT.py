#############################################################################
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################

# calculating semi-analytic matched asymtptoic solution

from numpy import *
from sol_CT import *
from scipy.interpolate import interp1d

def get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma):
    re = cond_CT.copy()
    re['phi_bulk']=phi_bulk
    re['epsilon_d']=epsilon_d
    re['dr']=dr
    re['dz']=dz
    re['gamma']=gamma
    return re

def get_v_GT(r, z, cond_GT, Pi_arr, INT_Ieta_yt_with_fixed_z):
    vZ = get_vw_GT(z, cond_GT, Pi_arr) # dimensional one
    rt = r/cond_GT['R']
    vR = 2.*rt - rt**3.0

    return vR * vZ


def get_vw_GT(z, cond_GT, Pi_arr):
    return get_v_CT(cond_GT['R'], z, cond_GT, cond_GT['dz'], Pi_arr)


def get_P_GT(r, z, cond_GT, Pi_arr):
    return get_P_CT(r, z, cond_GT, cond_GT['dz'], Pi_arr)

def get_u_GT(r, z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi_yt_with_fixed_z, INT_Ieta_yt_with_fixed_z):
    k=cond_GT['k']; Pper = cond_GT['Pper']; Cp = cond_GT['Cp_CT']; Cm = cond_GT['Cm_CT']
    L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
    dr = cond_GT['dr']; dz = cond_GT['dz']
    ed = cond_GT['epsilon_d']
    
    rt = r/R; zt = z/L;
    dyt = dr/R; dyp = ed*dyt
    
    gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
    gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    
    uZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         - exp(-k*z/L)*(Cm + (k/2.)*gp)

    # note that uR is not the only function of r since this is the matched asymptotic solution
    # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
    uR = 0.
    ed = cond_GT['epsilon_d']
    tmp_yt = 0.
    yt = 1. - rt

    int_Y = INT_Ieta_yt_with_fixed_z(yt)
    uY = (2. - yt)*int_Y
    return -preU * uY * uZ 


def get_u_center_GT(z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi, INT_eta):
    k=cond_GT['k']; Pper = cond_GT['Pper']; Cp = cond_GT['Cp_CT']; Cm = cond_GT['Cm_CT']
    L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
    dr = cond_GT['dr']; dz = cond_GT['dz']
    ed = cond_GT['epsilon_d']
    
    rt = 0./R; zt = z/L;
    dyt = dr/R; dyp = ed*dyt
    
    gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
    gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    
    uZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         - exp(-k*z/L)*(Cm + (k/2.)*gp)

    # note that uR is not the only function of r since this is the matched asymptotic solution
    # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
    uR = 0.
    ed = cond_GT['epsilon_d']
    tmp_yt = 0.
    yt = 1. - rt

    uY = (2. - yt)*INT_eta
    return -preU * uY * uZ 


def get_v_GT_boost(r, z, cond_GT, Pi_arr, INT_Ieta_yt_with_fixed_z, gp_z, gm_z):
    vZ = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z) # dimensional one
    rt = r/cond_GT['R']
    vR = 2.*rt - rt**3.0

    return vR * vZ

def get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z):
    return get_v_CT_boost(cond_GT['R'], z, cond_GT, cond_GT['dz'], Pi_arr, gp_z, gm_z)

def get_P_GT_boost(r, z, cond_GT, Pi_arr, gp_z, gm_z):
    return get_P_CT_boost(r, z, cond_GT, cond_GT['dz'], Pi_arr, gp_z, gm_z)

def get_u_GT_boost(r, z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi_yt_with_fixed_z, INT_Ieta_yt_with_fixed_z, gp_z, gm_z):
    k=cond_GT['k']; Pper = cond_GT['Pper']; Cp = cond_GT['Cp_CT']; Cm = cond_GT['Cm_CT']
    L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
    dr = cond_GT['dr']; dz = cond_GT['dz']
    ed = cond_GT['epsilon_d']
    
    rt = r/R; zt = z/L;
    dyt = dr/R; dyp = ed*dyt
    
    gp = gp_z
    gm = gm_z
    
    uZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         - exp(-k*z/L)*(Cm + (k/2.)*gp)

    # note that uR is not the only function of r since this is the matched asymptotic solution
    # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
    uR = 0.
    ed = cond_GT['epsilon_d']
    tmp_yt = 0.
    yt = 1. - rt

    int_Y = INT_Ieta_yt_with_fixed_z(yt)
    uY = (2. - yt)*int_Y
    return -preU * uY * uZ 


def get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi, INT_eta, gp_z, gm_z):
    k=cond_GT['k']; Pper = cond_GT['Pper']; Cp = cond_GT['Cp_CT']; Cm = cond_GT['Cm_CT']
    L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
    dr = cond_GT['dr']; dz = cond_GT['dz']
    ed = cond_GT['epsilon_d']
    
    rt = 0./R; zt = z/L;
    dyt = dr/R; dyp = ed*dyt
    
    gp = gp_z
    gm = gm_z
    
    uZ = exp(k*z/L)*(Cp + (k/2.)*gm)    \
         - exp(-k*z/L)*(Cm + (k/2.)*gp)

    # note that uR is not the only function of r since this is the matched asymptotic solution
    # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
    uR = 0.
    ed = cond_GT['epsilon_d']
    tmp_yt = 0.
    yt = 1. - rt

    uY = (2. - yt)*INT_eta
    return -preU * uY * uZ 


def gen_yt_arr(cond_GT):
    dy = cond_GT['dr']
    dyt = dy/cond_GT['R']
    dyp = cond_GT['epsilon_d'] * dyt
    
    tmp_yt = 0.
    yt_arr = [tmp_yt]
    while(tmp_yt < 1. - dyt):
        if tmp_yt < cond_GT['epsilon_d']:
            tmp_dy = dyp
        elif tmp_yt < 2. * cond_GT['epsilon_d']:
            tmp_dy = 2.*dyp
        elif tmp_yt < 10. * cond_GT['epsilon_d']:
            tmp_dy = 10.*dyp
        else:
            tmp_dy = dyt
        tmp_yt += tmp_dy
        yt_arr.append(tmp_yt)
    yt_arr = asarray(yt_arr)
    return yt_arr

# def RK4(f, h, y, phi):
#     k1 = h*f(y, phi)
#     k2 = h*f(y + h/2., phi + k1/2.)
#     k3 = h*f(y + h/2., phi + k2/2.)
#     k4 = h*f(y + h,    phi + k3)
#     return phi + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def get_f_RK(yt, dyt, f, df, vw_div_vw0, fcn_D, cond_GT):
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    y_new = yt + dyt
    f_new = f + df
    # int_INV_D = int_INV_D_pre
    # if df <> 0.: # it is related with half-step for RK4 method
    #     int_INV_D += (dyt/2.)*(1./fcn_D(f, cond_GT) + 1./fcn_D(f_new, cond_GT))

    return (-1./ed)*(vw_div_vw0/fcn_D(f_new, cond_GT))*(f_new - phi_b)
    # return (-1./ed)*(vw_div_vw0/fcn_D(f, cond_GT))*(f - phi_b*(1. - exp(-(vw_div_vw0/eb)*int_INV_D)))

def get_phi_with_fixed_z_GT(z, cond_GT, phiw_z, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr):
    # note that if we directly integrate dphi/dy = -(1/ed)*vw/D, the stiff decreasing when y is near 0 make inaccuracy
    # therefore, it must be account the proper steps with dy' = ed* dy.
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    R = cond_GT['R']; L = cond_GT['L']

    zt = z/L;
    
    dyt = dr/R
    dyp = ed*dyt

    phiw = phiw_z#fcn_phiw(z)
    rt = 0.
    yt = 1. - rt # coordinate transform
    tmp_yt = 0.
    phi = phiw

    phi_arr[0] = phi

    # # simple Euler method
    # for i in range(1, size(yt_arr)):
    #     dy = yt_arr[i] - yt_arr[i-1]
    #     # phi += dy*(1/ed)*(vw_div_vw0/fcn_D(phi))*(phi - phi_b)
    #     phi_arr[i] = phi_arr[i-1] - dy*(1/ed)*(vw_div_vw0/fcn_D(phi_arr[i-1]))*(phi_arr[i-1] - phi_b)

    # Runge-Kutta 4th order method
    for i in range(1, size(yt_arr)):
        y_2 = yt_arr[i]; y_1 = yt_arr[i-1]
        dy = y_2 - y_1
        y_h = y_1 + dy/2.
        phi_1 = phi_arr[i-1]
        k1 = dy * get_f_RK(y_1, 0., phi_1, 0., vw_div_vw0, fcn_D, cond_GT)
        k2 = dy * get_f_RK(y_1, dy/2., phi_1, k1/2., vw_div_vw0, fcn_D, cond_GT)
        k3 = dy * get_f_RK(y_1, dy/2., phi_1, k2/2., vw_div_vw0, fcn_D, cond_GT)
        k4 = dy * get_f_RK(y_1, dy, phi_1, k3, vw_div_vw0, fcn_D, cond_GT)

        phi_2 = phi_1 + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        phi_arr[i] = phi_2
            
    return 0



def get_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr):
    # this function will integrate 1/fcn(phi) with respect to y
    # Note that this function cannot be used inside get_phi_GT since int_fcn_phi and get_phi_GT will be interconnected in this case
    # In this code, get_phi_GT will be independent from int_fcn_phi while it will be used for other functions

    R = cond_GT['R']; L = cond_GT['L']

    ed = cond_GT['epsilon_d']
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    re = 0.
    Ieta_arr[0] = re
    tmp_f1 = 1./fcn_eta(phi_arr[0], cond_GT)
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i]  - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = 1./fcn_eta(phi_arr[i], cond_GT)
        Ieta_arr[i] = Ieta_arr[i-1] + 0.5*dy*(tmp_f1 + tmp_f2)
    return 0

def get_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr):
    # this function will integrate 1/fcn(phi) with respect to y
    # Note that this function cannot be used inside get_phi_GT since int_fcn_phi and get_phi_GT will be interconnected in this case
    # In this code, get_phi_GT will be independent from int_fcn_phi while it will be used for other functions
    R = cond_GT['R']; L = cond_GT['L']

    ed = cond_GT['epsilon_d']
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    re = 0.
    ID_arr[0] = re
    tmp_f1 = 1./fcn_D(phi_arr[0], cond_GT)
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i]  - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = 1./fcn_D(phi_arr[i], cond_GT)
        ID_arr[i] = ID_arr[i-1] + 0.5*dy*(tmp_f1 + tmp_f2)

    return 0

def get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr):
    # for a convenience, we replace the dimensionless coordination from r to y = 1-r

    R = cond_GT['R']; L = cond_GT['L']

    ed = cond_GT['epsilon_d']
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    re = 0.
    tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])
        re += 0.5 * dy * (tmp_f1 + tmp_f2)

    return re*uZz


def get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr):
    # for a convenience, we replace the dimensionless coordination from r to y = 1-r
    R = cond_GT['R']; L = cond_GT['L']

    ed = cond_GT['epsilon_d']
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    re = 0.
    tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]
        re += 0.5 * dy * (tmp_f1 + tmp_f2)
    
    return re*uZz


def process_at_z(index_i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, weight, gp_arr, gm_arr):
    # print index_i
    index_i = int(index_i)
    phi_b = cond_GT['phi_bulk']
    z = z_arr[index_i]
    phi_arr = zeros(size(yt_arr))
    Ieta_arr = zeros(size(yt_arr))
    ID_arr = zeros(size(yt_arr))

    gp_z = gp_arr[index_i]
    gm_z = gm_arr[index_i]
    
    vw = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z)
    vw_div_vw0 = vw/cond_GT['vw0']        
    get_phi_with_fixed_z_GT(z, cond_GT, phiw_div_phib_arr[index_i]*phi_b, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr)
    get_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr)
    get_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr)
    # bar_eta_with_fixed_z = cross_sectional_average_eta(cond_GT, INT_phi_yt_with_fixed_z, fcn_eta)
    uZz = get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr[-1], Ieta_arr[-1], gp_z, gm_z)
    F1 = get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
    F2_Z = get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr)

    return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_Z)/F1)


def get_new_phiw_div_phib_arr(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
    phiw_new_arr = ones(size(phiw_div_phib_arr))
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']
    R = cond_GT['R']; L = cond_GT['L']
    dyt = dr/R
    dyp = ed*dyt

    Ny = size(yt_arr)
    phi_arr_z0 = zeros(Ny)
    Ieta_arr_z0 = zeros(Ny)
    ID_arr_z0 = zeros(Ny)

    vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        

    get_phi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
    get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
    get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
    uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
    F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)

    for i in range(size(z_arr)):
        phiw_new_arr[i] = process_at_z(i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, weight, gp_arr, gm_arr)
    
    return phiw_new_arr

def get_f_modi_RK(yt, dyt, f, df, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT):
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    y_new = yt + dyt
    f_new = f + df
    int_INV_D = int_INV_D_pre
    if df <> 0.: # it is related with half-step for RK4 method
        int_INV_D += (dyt/2.)*(1./fcn_D(f, cond_GT) + 1./fcn_D(f_new, cond_GT))
    return (-1./ed)*(vw_div_vw0/fcn_D(f_new, cond_GT))*(f_new - phi_b*(1. - exp(-(vw_div_vw0/ed)*int_INV_D)))
    # return (-1./ed)*(vw_div_vw0/fcn_D(f, cond_GT))*(f - phi_b*(1. - exp(-(vw_div_vw0/eb)*int_INV_D)))

def get_phi_modi_with_fixed_z_GT(z, cond_GT, phiw_z, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr):
    # note that if we directly integrate dphi/dy = -(1/ed)*vw/D, the stiff decreasing when y is near 0 make inaccuracy
    # therefore, it must be account the proper steps with dy' = ed* dy.
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    R = cond_GT['R']; L = cond_GT['L']

    zt = z/L;
    
    dyt = dr/R
    dyp = ed*dyt

    phiw = phiw_z#fcn_phiw(z)
    rt = 0.
    yt = 1. - rt # coordinate transform
    tmp_yt = 0.
    phi = phiw

    phi_arr[0] = phi

    int_INV_D_pre = 0.
    for i in range(1, size(yt_arr)):
        y_2 = yt_arr[i]; y_1 = yt_arr[i-1]
        dy = y_2 - y_1
        y_h = y_1 + dy/2.
        phi_1 = phi_arr[i-1]
        k1 = dy * get_f_modi_RK(y_1, 0., phi_1, 0., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k2 = dy * get_f_modi_RK(y_1, dy/2., phi_1, k1/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k3 = dy * get_f_modi_RK(y_1, dy/2., phi_1, k2/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k4 = dy * get_f_modi_RK(y_1, dy, phi_1, k3, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        phi_2 = phi_1 + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        phi_arr[i] = phi_2

        int_INV_D_pre += (dy/2.) * (1./fcn_D(phi_2, cond_GT) + 1./fcn_D(phi_1, cond_GT))
    return 0


def get_F_modi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr):
    # for a convenience, we replace the dimensionless coordination from r to y = 1-r

    R = cond_GT['R']; L = cond_GT['L']

    ed = cond_GT['epsilon_d']
    phi_b = cond_GT['phi_bulk']

    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']

    re = 0.
    tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])*((vw_div_vw0/ed)*ID_arr[0])
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])*((vw_div_vw0/ed)*ID_arr[i])
        re += 0.5 * dy * (tmp_f1 + tmp_f2)
    return re*uZz

def process_at_z_modi(index_i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, F2_0_modi, weight, gp_arr, gm_arr):
    index_i = int(index_i)
    phi_b = cond_GT['phi_bulk']
    z = z_arr[index_i]
    phi_arr = zeros(size(yt_arr))
    Ieta_arr = zeros(size(yt_arr))
    ID_arr = zeros(size(yt_arr))

    gp_z = gp_arr[index_i]
    gm_z = gm_arr[index_i]

    vw = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z)
    vw_div_vw0 = vw/cond_GT['vw0']
    get_phi_modi_with_fixed_z_GT(z, cond_GT, phiw_div_phib_arr[index_i]*phi_b, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr)
    get_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr)
    get_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr)

    uZz = get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr[-1], Ieta_arr[-1], gp_z, gm_z)
    F1 = get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
    F2_Z = get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr)
    F2_Z_modi = get_F_modi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
    return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_0_modi - (F2_Z - F2_Z_modi))/F1)
    # return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_0_modi - (F2_Z - F2_Z_modi))/F1)

    # return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - (F2_Z - F2_Z_modi))/F1)
# no ITER at z=0 introduce additional discontinuity

def get_new_phiw_div_phib_modi_arr(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
    import time
    phiw_new_arr = ones(size(phiw_div_phib_arr))
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    dr = cond_GT['dr']
    R = cond_GT['R']; L = cond_GT['L']
    dyt = dr/R
    dyp = ed*dyt

    Ny = size(yt_arr)
    phi_arr_z0 = zeros(Ny)
    Ieta_arr_z0 = zeros(Ny)
    ID_arr_z0 = zeros(Ny)
    
    vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        

    get_phi_modi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
    get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
    get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
    uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
    F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)
    F2_0_modi = get_F_modi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, vw_div_vw0_z0, yt_arr, phi_arr_z0, Ieta_arr_z0, ID_arr_z0)
    # F2_0 = uZ0/4.
    # the difference between uZ0/4 and get_F2 is less than 1%.
    # F2_0_modi = 0.
    
    for i in range(size(z_arr)):
        phiw_new_arr[i] = process_at_z_modi(i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, F2_0_modi, weight, gp_arr, gm_arr)
    
    return phiw_new_arr



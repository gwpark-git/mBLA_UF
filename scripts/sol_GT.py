#############################################################################
#   Concentration and flow profile of suspension flow in CF UF              #
#   Using a general concentration-dependent diffusivity and viscosity       #
#   One must give the experssion for D and eta on top of the Pi             #
#                                                                           #
#   Modeling cross-flow ultrafiltration of permeable particles dispersions  #
#   Paper authors: Park, Gun Woo and Naegele, Gerhard                       #
#   Developer: Park, Gun Woo                                                #
#   email: g.park@fz-juelich.de                                             #
#############################################################################

# calculating semi-analytic matched asymtptoic solution

from numpy import *
import sol_CT as CT
from scipy.interpolate import interp1d

def get_cond_GT(cond_CT, phi_bulk, epsilon_d, dr, dz, gamma):
    if (cond_CT['COND'] <> 'CT'):
        print('Error: inherit non-CT type of dictionary in get_cond_GT is not supported.')
        
    COND_TYPE       = 'GT'
    re              = cond_CT.copy()
    re['COND']      = COND_TYPE # update the given type to GT (general transport properties)
    re['phi_bulk']  = phi_bulk
    re['epsilon_d'] = epsilon_d
    re['dr']        = dr
    re['dz']        = dz
    re['gamma']     = gamma
    return re


# convinient wrapper functions using dictionary of conditions

def get_P_conv(r_div_R, z_div_L, cond_GT, gp, gm):
    return get_P(r_div_R, z_div_L, cond_GT['Pper_div_DLP'], cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp, gm)

def get_u_conv(r_div_R, z_div_L, cond_GT, gp, gm, INT_Ieta_yt_with_fixed_z):
    return get_u(r_div_R, z_div_L, cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp, gm, INT_Ieta_yt_with_fixed_z)

def get_v_conv(r_div_R, z_div_L, Pi_div_DLP, cond_GT, gp, gm):
    return get_v(r_div_R, z_div_L, Pi_div_DLP, cond_GT['k'], cond_GT['alpha_ast'], cond_GT['Bp'], cond_GT['Bm'], gp, gm)

# flow profiles 

def get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm):
    """ Using expression in P=P^out in Eqs. (45) and (49)
    There is no difference from CT.get_P
    """
    return CT.get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm)


def get_u(r_div_R, z_div_L, k, Bp, Bm, gp, gm, INT_Ieta_yt_with_fixed_z):
    """ Using expressions u in Eq. (45) 
    and integrate 1/eta from r to 1 is reversed from 0 to y (sign change is already applied)
    u_Z^out is given in following Eq. (45)
    """
    
    y_div_R = 1. - r_div_R
    int_Y = INT_Ieta_yt_with_fixed_z(y_div_R)
    
    uR = (1. + r_div_R)*int_Y
    uZ_out = exp(k*z_div_L)*(Bp + (k/2.)*gm) \
        - exp(-k*z_div_L)*(Bm + (k/2.)*gp)

    return uZ_out * uR


def get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm):
    """ Using expression v=v^out in Eqs. (45) and (49)
    As described in CT.get_v, sign is positive because we are using coordinate function r here
    """
    sign = +1.
    rw_div_R = 1.
    vw = CT.get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm)
    vR = 2.*r_div_R - r_div_R**3.0
    return sign * vR * vw


# for y-directional information related with outer solution

def gen_yt_arr(cond_GT):
    """ This is generating discretized dimensionless y-coordinate
    The selected way for the adaptive step size is only for the temporary
    There are revised version, which will be applied for the near future.
    """
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


def cal_f_RK(yt, dyt, f, df, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT):
    """ Auxiliary function for Runge-Kutta method to generate matched asymptotic phi
    The complication of input arguments and result are due to the implicit expression of phi,
    which is described in help-doc in gen_phi_wrt_yt function.
    """
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    
    y_new = yt + dyt
    f_new = f + df
    int_INV_D = int_INV_D_pre
    if df != 0.: # it is related with half-step for RK4 method
        int_INV_D += (dyt/2.)*(1./fcn_D(f, cond_GT) + 1./fcn_D(f_new, cond_GT))
    return (-1./ed)*(vw_div_vw0/fcn_D(f_new, cond_GT))*(f_new - phi_b*(1. - exp(-(vw_div_vw0/ed)*int_INV_D)))


def gen_phi_wrt_yt(z_div_L, phiw, fcn_D, vw_div_vw0, y_div_R_arr, phi_arr, phi_b, ed):
    """ Generating phi with respect to y_div_R using expression of phi Eq. (49).

    Note that the matched asymptotic phi is the implicit equation, which make difficult
    to use Runge-Kutta function in scipy directly.

    Instead, we define our own y_div_R evolution based on Runge-Kutta 4th order method.
    This is supported by cal_f_RK function defined above.
    """
    
    phi = phiw
    int_INV_D_pre = 0.
    for i in range(1, size(y_div_R_arr)):
        y2 = y_div_R_arr[i]; y1 = y_div_R_arr[i-1]
        yh = y_1 + dy/2. # for RK4 method        
        dy = y2 - y1
        phi_1 = phi_arr[i-1]
        k1 = dy * cal_f_RK(y_1, 0., phi_1, 0., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k2 = dy * cal_f_RK(y_1, dy/2., phi_1, k1/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k3 = dy * cal_f_RK(y_1, dy/2., phi_1, k2/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k4 = dy * cal_f_RK(y_1, dy, phi_1, k3, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        phi_2 = phi_1 + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
        phi_arr[i] = phi_2

        int_INV_D_pre += (dy/2.) * (1./fcn_D(phi_2, cond_GT) + 1./fcn_D(phi_1, cond_GT))

    return 0


def gen_INT_inv_f_wrt_yt(yt_arr, phi_arr, INT_inv_f_arr, f_given, cond_GT):
    """ Taking integration over 1/f_given

    Parameters:
        yt_arr        = arrays for discretized y (unit: dimensionless)
        phi_arr       = arrays for discretized phi with respect to y (unit: dimensionless)
        INT_inv_f_arr = arrays for int_0^y 1/f (unit: inverse of function f)
        f_given       = given function f. Usually the function or interpolated function will be provided
        cond_GT       = dictionary for conditions

    Output:
        INT_inv_f_arr = The each result integrated from 0 to yt will be stored here

    Return: 0: no specific return type

    """
    re = 0.
    INT_inv_f_arr[0] = re
    tmp_f1 = 1./f_given(phi_arr[0], cond_GT)
    tmp_f2 = tmp_f1
    for i in range(1, size(yt_arr)):
        dy = yt_arr[i]  - yt_arr[i-1]
        tmp_f1 = tmp_f2
        tmp_f2 = 1./f_given(phi_arr[i], cond_GT)
        INT_inv_f_arr[i] = INT_inv_f_arr[i-1] + 0.5*dy*(tmp_f1 + tmp_f2)
    return 0


def cal_F2_0(vw_div_vw0_z0, ed, yt_arr, Ieta_arr_z0, ID_arr_z0):
    """ Calculate F2_0 for cal_int_Fz
    This is decoupled from cal_int_Fz for z>0 because the value F2_0 will be used for calculating F2_Z for all z>0.
    Therefore, this additional help function will reduce the overhead of calculation, even though the real function is just a part of cal_int_Fz.

    """
    re_F2_0 = 0.
    tmp_F2_0_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*(1. - Ieta_arr_z0[0]*exp(-(vw_div_vw0_z0/ed)*ID_arr_z0[0])*((vw_div_vw0_z0/ed)*ID_arr_z0[0]))
    tmp_F2_0_2 = tmp_F2_0_1

    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]

        tmp_F2_0_1 = tmp_F2_0_2
        tmp_F2_0_2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr_z0[i]*(1. - exp(-(vw_div_vw0_z0/ed)*ID_arr_z0[i])*((vw_div_vw0_z0/ed)*ID_arr_z0[i]))
        re_F2_0 += 0.5 * dy * (tmp_F2_0_1 + tmp_F2_0_2)
    return re_F2_0

def cal_int_Fz(given_re_F2_0, vw_div_vw0, ed, yt_arr, Ieta_arr, ID_arr):
    # ref: process_at_z_modi
    """ Calculate Fz[phi_w] using Eq. (D3)
    For readability, a new notation is used here:
        Fz[phi_w] := 1 + (F2_0 - F2_Z)/F1_Z
        where F2_Z is T_z[1-s_bar*e^{s_bar}], and F2_0 is F2_Z at z=0,
        and   F1_Z is T_z[e^{-s_bar}].
    The calculation of each integral operator T_z is merged into single function for performance.
    The F2_0 is the given value because it does not necessary to be re-calculate for each z.
    """
    
    re_F1_Z = 0.
    tmp_F1_Z_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])
    tmp_F1_Z_2 = tmp_F1_Z_1
    
    re_F2_Z = 0.
    tmp_F2_Z_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*(1. - Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])*((vw_div_vw0/ed)*ID_arr[0]))
    tmp_F2_Z_2 = tmp_F2_Z_1

    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]

        tmp_F1_Z_1 = tmp_F1_Z_2
        tmp_F1_Z_2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])
        re_F1_Z += 0.5 * dy * (tmp_F1_Z_1 + tmp_F1_Z_2)
        
        tmp_F2_Z_1 = tmp_F2_Z_2
        tmp_F2_Z_2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*(1. - exp(-(vw_div_vw0/ed)*ID_arr[i])*((vw_div_vw0/ed)*ID_arr[i]))
        re_F2_Z += 0.5 * dy * (tmp_F2_Z_1 + tmp_F2_Z_2)
    return 1. - (given_re_F2_0 - re_F2_Z)/re_F1_Z

def FPI_operator(weight, val_pre, val_new):
    """ val_new = (1. - weight)*val_pre + weight*val_new
    Applying operator for the fixed-point iteration
    Here the notation is simply by
    val_new = (1 - weight)*val_pre + weight*val_new
    where weight = 1 is the normal FPI, and
          weight < 1 is the under-relaxed FPI.

    Parameter:
        weight: scalar
        val_pre: scalar or vector
        val_new: scalar or vector

    Output: overwrite val_new (without return value)
    """
    val_new = (1. - weight)*val_pre + weight*val_new 
    return 0
    # return (1. - weight)*val_pre + weight*val_new

def gen_new_phiw_div_phib_arr(phiw_div_phib_arr_new, cond_GT, fcn_D, fcn_eta, z_div_L_arr, phiw_div_phib_arr, Pi_div_DLP_arr, weight, gp_arr, gm_arr, y_div_R_arr):
    """ Calculation phi_w/phi_b at the given z using Eq. (D3)
    The detailed terms in Eq. (D3) is explained in the function cal_int_Fz which calculate the integration.
    """
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    
    Ny = size(y_div_R_arr)
    phi_arr_z0 = zeros(Ny)
    Ieta_arr_z0 = zeros(Ny)
    ID_arr_z0 = zeros(Ny)

    phi_arr_zi = zeros(Ny)
    Ieta_arr_zi = zeros(Ny)
    ID_arr_zi = zeros(Ny)

    ind_z0 = 0
    
    z0_div_L = 0.
    
    r0_div_R = 0.
    rw_div_R = 1.
    
    vw_div_vw0_z0 = get_v_conv(r_div_R=rw_div_R, z_div_L=z0_div_L, Pi_div_DLP_arr[ind_z0], cond_GT, gp_arr[ind_z0], gm_arr[ind_z0])
    gen_phi_wrt_yt(z_div_L=z0_div_L, phiw_div_phib_arr[ind_z0]*phi_b, fcn_D, vw_div_vw0_z0, y_div_R_arr, phi_arr_z0, phi_b, ed)
    gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_z0, Ieta_arr_z0, fcn_eta, cond_GT)
    gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_z0, ID_arr_z0, fcn_D, cond_GT)

    uZ0 = get_u_conv(r_div_R=r0_div_R, z_div_L=z0_div_L, cond_GT, gp_arr[ind_z0], gm_arr[ind_z0], Ieta_arr_z0)
    F2_0 = cal_F2_0(vw_div_vw0_z0, ed, y_div_R_arr, Ieta_arr_z0, ID_arr_z0)

    for i in range(1, size(z_arr)):
        vw_div_vw0_zi = get_v_conv(r_div_R=rw_div_R, z_div_L=z_div_L_arr[i], Pi_div_DLP_arr[i], cond_GT, gp_arr[i], gm_arr[i])
        gen_phi_wrt_yt(z_div_L=z_div_L_arr[i], phiw_div_phib_arr[i]*phi_b, fcn_D, vw_div_vw0_zi, y_div_R_arr, phi_arr_zi, phi_b, ed)
        gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_zi, Ieta_arr_zi, fcn_eta, cond_GT)
        gen_INT_inv_f_wrt_yt(y_div_R_arr, phi_arr_zi, ID_arr_zi, fcn_D, cond_GT)
        
        phiw_div_phib_arr_new[i] = cal_int_Fz(given_re_F2_0, vw_div_vw0_zi, ed, yt_arr, Ieta_arr_zi, ID_arr_zi)

    FPI_operator(cond_GT['weight'], phiw_div_phib_arr, phiw_div_phib_arr_new)
    
    return 0


# # def get_phi_modi_with_fixed_z_GT(z, cond_GT, phiw_z, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr):
# #     # note that if we directly integrate dphi/dy = -(1/ed)*vw/D, the stiff decreasing when y is near 0 make inaccuracy
# #     # therefore, it must be account the proper steps with dy' = ed* dy.
# #     phi_b = cond_GT['phi_bulk']

# #     ed = cond_GT['epsilon_d']
# #     dr = cond_GT['dr']

# #     R = cond_GT['R']; L = cond_GT['L']

# #     zt = z/L;
    
# #     dyt = dr/R
# #     dyp = ed*dyt

# #     phiw = phiw_z#fcn_phiw(z)
# #     rt = 0.
# #     yt = 1. - rt # coordinate transform
# #     tmp_yt = 0.
# #     phi = phiw

# #     phi_arr[0] = phi

# #     int_INV_D_pre = 0.
# #     for i in range(1, size(yt_arr)):
# #         y_2 = yt_arr[i]; y_1 = yt_arr[i-1]
# #         dy = y_2 - y_1
# #         y_h = y_1 + dy/2.
# #         phi_1 = phi_arr[i-1]
# #         k1 = dy * get_f_modi_RK(y_1, 0., phi_1, 0., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
# #         k2 = dy * get_f_modi_RK(y_1, dy/2., phi_1, k1/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
# #         k3 = dy * get_f_modi_RK(y_1, dy/2., phi_1, k2/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
# #         k4 = dy * get_f_modi_RK(y_1, dy, phi_1, k3, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
# #         phi_2 = phi_1 + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
# #         phi_arr[i] = phi_2

# #         int_INV_D_pre += (dy/2.) * (1./fcn_D(phi_2, cond_GT) + 1./fcn_D(phi_1, cond_GT))
# #     return 0



# # # def get_v(r_div_R, z_div_L):
# # #     vw = get_vw_GT(z

# # def get_v_GT(r, z, cond_GT, Pi_arr, INT_Ieta_yt_with_fixed_z):
# #     vZ = get_vw_GT(z, cond_GT, Pi_arr) # dimensional one
# #     rt = r/cond_GT['R']
# #     vR = 2.*rt - rt**3.0

# #     return vR * vZ

# # def get_vw_GT(z, cond_GT, Pi_arr):
# #     return CT.get_v(cond_GT['R'], z, cond_GT, cond_GT['dz'], Pi_arr)

# # def get_P_GT(r, z, cond_GT, Pi_arr):
# #     return CT.get_P(r, z, cond_GT, cond_GT['dz'], Pi_arr)

# # def get_u_GT(r, z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi_yt_with_fixed_z, INT_Ieta_yt_with_fixed_z):
# #     k=cond_GT['k']; Pper = cond_GT['Pper']; Ap = cond_GT['Bp']; Am = cond_GT['Bm']
# #     L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
# #     dr = cond_GT['dr']; dz = cond_GT['dz']
# #     ed = cond_GT['epsilon_d']
    
# #     rt = r/R; zt = z/L;
# #     dyt = dr/R; dyp = ed*dyt
    
# #     gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
# #     gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    
# #     uZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
# #          - exp(-k*z/L)*(Am + (k/2.)*gp)

# #     # note that uR is not the only function of r since this is the matched asymptotic solution
# #     # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
# #     uR = 0.
# #     ed = cond_GT['epsilon_d']
# #     tmp_yt = 0.
# #     yt = 1. - rt

# #     int_Y = INT_Ieta_yt_with_fixed_z(yt)
# #     uY = (2. - yt)*int_Y
# #     return -preU * uY * uZ 


# # def get_u_center_GT(z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi, INT_eta):
# #     k=cond_GT['k']; Pper = cond_GT['Pper']; Ap = cond_GT['Bp']; Am = cond_GT['Bm']
# #     L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
# #     dr = cond_GT['dr']; dz = cond_GT['dz']
# #     ed = cond_GT['epsilon_d']
    
# #     rt = 0./R; zt = z/L;
# #     dyt = dr/R; dyp = ed*dyt
    
# #     gp = get_gpm(z, dz, +1.0, Pi_arr, k, L)
# #     gm = get_gpm(z, dz, -1.0, Pi_arr, k, L)
    
# #     uZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
# #          - exp(-k*z/L)*(Am + (k/2.)*gp)

# #     # note that uR is not the only function of r since this is the matched asymptotic solution
# #     # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
# #     uR = 0.
# #     ed = cond_GT['epsilon_d']
# #     tmp_yt = 0.
# #     yt = 1. - rt

# #     uY = (2. - yt)*INT_eta
# #     return -preU * uY * uZ 


# # def get_v_GT_boost(r, z, cond_GT, Pi_arr, INT_Ieta_yt_with_fixed_z, gp_z, gm_z):
# #     vZ = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z) # dimensional one
# #     rt = r/cond_GT['R']
# #     vR = 2.*rt - rt**3.0

# #     return vR * vZ

# # def get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z):
# #     return CT.get_v_conv(cond_GT['R'], z, cond_GT, cond_GT['dz'], Pi_arr, gp_z, gm_z)

# # def get_P_GT_boost(r, z, cond_GT, Pi_arr, gp_z, gm_z):
# #     return CT.get_P_conv(r, z, cond_GT, cond_GT['dz'], Pi_arr, gp_z, gm_z)

# # def get_u_GT_boost(r, z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi_yt_with_fixed_z, INT_Ieta_yt_with_fixed_z, gp_z, gm_z):
# #     k=cond_GT['k']; Pper = cond_GT['Pper']; Ap = cond_GT['Bp']; Am = cond_GT['Bm']
# #     L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
# #     dr = cond_GT['dr']; dz = cond_GT['dz']
# #     ed = cond_GT['epsilon_d']
    
# #     rt = r/R; zt = z/L;
# #     dyt = dr/R; dyp = ed*dyt
    
# #     gp = gp_z
# #     gm = gm_z
    
# #     uZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
# #          - exp(-k*z/L)*(Am + (k/2.)*gp)

# #     # note that uR is not the only function of r since this is the matched asymptotic solution
# #     # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
# #     uR = 0.
# #     ed = cond_GT['epsilon_d']
# #     tmp_yt = 0.
# #     yt = 1. - rt

# #     int_Y = INT_Ieta_yt_with_fixed_z(yt)
# #     uY = (2. - yt)*int_Y
# #     return -preU * uY * uZ 


# # def get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta, INT_phi, INT_eta, gp_z, gm_z):
# #     k=cond_GT['k']; Pper = cond_GT['Pper']; Ap = cond_GT['Bp']; Am = cond_GT['Bm']
# #     L = cond_GT['L']; R = cond_GT['R']; preU=cond_GT['preU']
# #     dr = cond_GT['dr']; dz = cond_GT['dz']
# #     ed = cond_GT['epsilon_d']
    
# #     rt = 0./R; zt = z/L;
# #     dyt = dr/R; dyp = ed*dyt
    
# #     gp = gp_z
# #     gm = gm_z
    
# #     uZ = exp(k*z/L)*(Ap + (k/2.)*gm)    \
# #          - exp(-k*z/L)*(Am + (k/2.)*gp)

# #     # note that uR is not the only function of r since this is the matched asymptotic solution
# #     # u = uZ * eta_w * (1+r) * int_r^1 1/eta(phi) dr'
# #     uR = 0.
# #     ed = cond_GT['epsilon_d']
# #     tmp_yt = 0.
# #     yt = 1. - rt

# #     uY = (2. - yt)*INT_eta
# #     return -preU * uY * uZ 



# # def RK4(f, h, y, phi):
# #     k1 = h*f(y, phi)
# #     k2 = h*f(y + h/2., phi + k1/2.)
# #     k3 = h*f(y + h/2., phi + k2/2.)
# #     k4 = h*f(y + h,    phi + k3)
# #     return phi + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

# # def get_f_RK(yt, dyt, f, df, vw_div_vw0, fcn_D, cond_GT):
# #     phi_b = cond_GT['phi_bulk']
# #     ed = cond_GT['epsilon_d']
# #     y_new = yt + dyt
# #     f_new = f + df
# #     # int_INV_D = int_INV_D_pre
# #     # if df <> 0.: # it is related with half-step for RK4 method
# #     #     int_INV_D += (dyt/2.)*(1./fcn_D(f, cond_GT) + 1./fcn_D(f_new, cond_GT))

# #     return (-1./ed)*(vw_div_vw0/fcn_D(f_new, cond_GT))*(f_new - phi_b)
# #     # return (-1./ed)*(vw_div_vw0/fcn_D(f, cond_GT))*(f - phi_b*(1. - exp(-(vw_div_vw0/eb)*int_INV_D)))

# # def get_phi_with_fixed_z_GT(z, cond_GT, phiw_z, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr):
# #     # note that if we directly integrate dphi/dy = -(1/ed)*vw/D, the stiff decreasing when y is near 0 make inaccuracy
# #     # therefore, it must be account the proper steps with dy' = ed* dy.
# #     phi_b = cond_GT['phi_bulk']

# #     ed = cond_GT['epsilon_d']
# #     dr = cond_GT['dr']

# #     R = cond_GT['R']; L = cond_GT['L']

# #     zt = z/L;
    
# #     dyt = dr/R
# #     dyp = ed*dyt

# #     phiw = phiw_z#fcn_phiw(z)
# #     rt = 0.
# #     yt = 1. - rt # coordinate transform
# #     tmp_yt = 0.
# #     phi = phiw

# #     phi_arr[0] = phi

# #     # # simple Euler method
# #     # for i in range(1, size(yt_arr)):
# #     #     dy = yt_arr[i] - yt_arr[i-1]
# #     #     # phi += dy*(1/ed)*(vw_div_vw0/fcn_D(phi))*(phi - phi_b)
# #     #     phi_arr[i] = phi_arr[i-1] - dy*(1/ed)*(vw_div_vw0/fcn_D(phi_arr[i-1]))*(phi_arr[i-1] - phi_b)

# #     # Runge-Kutta 4th order method
# #     for i in range(1, size(yt_arr)):
# #         y_2 = yt_arr[i]; y_1 = yt_arr[i-1]
# #         dy = y_2 - y_1
# #         y_h = y_1 + dy/2.
# #         phi_1 = phi_arr[i-1]
# #         k1 = dy * get_f_RK(y_1, 0., phi_1, 0., vw_div_vw0, fcn_D, cond_GT)
# #         k2 = dy * get_f_RK(y_1, dy/2., phi_1, k1/2., vw_div_vw0, fcn_D, cond_GT)
# #         k3 = dy * get_f_RK(y_1, dy/2., phi_1, k2/2., vw_div_vw0, fcn_D, cond_GT)
# #         k4 = dy * get_f_RK(y_1, dy, phi_1, k3, vw_div_vw0, fcn_D, cond_GT)

# #         phi_2 = phi_1 + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)
# #         phi_arr[i] = phi_2
            
# #     return 0



# def cal_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr):
#     # this function will integrate 1/fcn(phi) with respect to y
#     # Note that this function cannot be used inside get_phi_GT since int_fcn_phi and get_phi_GT will be interconnected in this case
#     # In this code, get_phi_GT will be independent from int_fcn_phi while it will be used for other functions

#     R = cond_GT['R']; L = cond_GT['L']

#     ed = cond_GT['epsilon_d']
#     phi_b = cond_GT['phi_bulk']

#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']

#     re = 0.
#     Ieta_arr[0] = re
#     tmp_f1 = 1./fcn_eta(phi_arr[0], cond_GT)
#     tmp_f2 = tmp_f1
#     for i in range(1, size(yt_arr)):
#         dy = yt_arr[i]  - yt_arr[i-1]
#         tmp_f1 = tmp_f2
#         tmp_f2 = 1./fcn_eta(phi_arr[i], cond_GT)
#         Ieta_arr[i] = Ieta_arr[i-1] + 0.5*dy*(tmp_f1 + tmp_f2)
#     return 0

# def cal_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr):
#     # this function will integrate 1/fcn(phi) with respect to y
#     # Note that this function cannot be used inside get_phi_GT since int_fcn_phi and get_phi_GT will be interconnected in this case
#     # In this code, get_phi_GT will be independent from int_fcn_phi while it will be used for other functions
#     R = cond_GT['R']; L = cond_GT['L']

#     ed = cond_GT['epsilon_d']
#     phi_b = cond_GT['phi_bulk']

#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']

#     re = 0.
#     ID_arr[0] = re
#     tmp_f1 = 1./fcn_D(phi_arr[0], cond_GT)
#     tmp_f2 = tmp_f1
#     for i in range(1, size(yt_arr)):
#         dy = yt_arr[i]  - yt_arr[i-1]
#         tmp_f1 = tmp_f2
#         tmp_f2 = 1./fcn_D(phi_arr[i], cond_GT)
#         ID_arr[i] = ID_arr[i-1] + 0.5*dy*(tmp_f1 + tmp_f2)

#     return 0

# def get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr):
#     # for a convenience, we replace the dimensionless coordination from r to y = 1-r
    
#     R = cond_GT['R']; L = cond_GT['L']

#     ed = cond_GT['epsilon_d']
#     phi_b = cond_GT['phi_bulk']

#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']

#     re = 0.
#     tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])
#     tmp_f2 = tmp_f1
#     for i in range(1, size(yt_arr)):
#         dy = yt_arr[i] - yt_arr[i-1]
#         tmp_f1 = tmp_f2
#         tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])
#         re += 0.5 * dy * (tmp_f1 + tmp_f2)

#     return re*uZz


# def get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr):
#     # for a convenience, we replace the dimensionless coordination from r to y = 1-r
#     R = cond_GT['R']; L = cond_GT['L']

#     ed = cond_GT['epsilon_d']
#     phi_b = cond_GT['phi_bulk']

#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']

#     re = 0.
#     tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]
#     tmp_f2 = tmp_f1
#     for i in range(1, size(yt_arr)):
#         dy = yt_arr[i] - yt_arr[i-1]
#         tmp_f1 = tmp_f2
#         tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]
#         re += 0.5 * dy * (tmp_f1 + tmp_f2)
    
#     return re*uZz


# def process_at_z(index_i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, weight, gp_arr, gm_arr):
#     # print index_i
#     index_i = int(index_i)
#     phi_b = cond_GT['phi_bulk']
#     z = z_arr[index_i]
#     phi_arr = zeros(size(yt_arr))
#     Ieta_arr = zeros(size(yt_arr))
#     ID_arr = zeros(size(yt_arr))

#     gp_z = gp_arr[index_i]
#     gm_z = gm_arr[index_i]
    
#     vw = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z)
#     vw_div_vw0 = vw/cond_GT['vw0']        
#     get_phi_with_fixed_z_GT(z, cond_GT, phiw_div_phib_arr[index_i]*phi_b, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr)
#     get_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr)
#     get_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr)
#     # bar_eta_with_fixed_z = cross_sectional_average_eta(cond_GT, INT_phi_yt_with_fixed_z, fcn_eta)
#     uZz = get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr[-1], Ieta_arr[-1], gp_z, gm_z)
#     F1 = get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
#     F2_Z = get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr)

#     return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_Z)/F1)


# def get_new_phiw_div_phib_arr(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
#     phiw_new_arr = ones(size(phiw_div_phib_arr))
#     phi_b = cond_GT['phi_bulk']
#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']
#     R = cond_GT['R']; L = cond_GT['L']
#     dyt = dr/R
#     dyp = ed*dyt

#     Ny = size(yt_arr)
#     phi_arr_z0 = zeros(Ny)
#     Ieta_arr_z0 = zeros(Ny)
#     ID_arr_z0 = zeros(Ny)

#     vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        

#     get_phi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
#     get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
#     get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
#     uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
#     F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)

#     for i in range(size(z_arr)):
#         phiw_new_arr[i] = process_at_z(i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, weight, gp_arr, gm_arr)
    
#     return phiw_new_arr

# def get_f_modi_RK(yt, dyt, f, df, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT):
#     phi_b = cond_GT['phi_bulk']
#     ed = cond_GT['epsilon_d']
#     y_new = yt + dyt
#     f_new = f + df
#     int_INV_D = int_INV_D_pre
#     if df != 0.: # it is related with half-step for RK4 method
#         int_INV_D += (dyt/2.)*(1./fcn_D(f, cond_GT) + 1./fcn_D(f_new, cond_GT))
#     return (-1./ed)*(vw_div_vw0/fcn_D(f_new, cond_GT))*(f_new - phi_b*(1. - exp(-(vw_div_vw0/ed)*int_INV_D)))
#     # return (-1./ed)*(vw_div_vw0/fcn_D(f, cond_GT))*(f - phi_b*(1. - exp(-(vw_div_vw0/eb)*int_INV_D)))



# def get_F_modi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr):
#     # for a convenience, we replace the dimensionless coordination from r to y = 1-r

#     R = cond_GT['R']; L = cond_GT['L']

#     ed = cond_GT['epsilon_d']
#     phi_b = cond_GT['phi_bulk']

#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']

#     re = 0.
#     tmp_f1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])*((vw_div_vw0/ed)*ID_arr[0])
#     tmp_f2 = tmp_f1
#     for i in range(1, size(yt_arr)):
#         dy = yt_arr[i] - yt_arr[i-1]
#         tmp_f1 = tmp_f2
#         tmp_f2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])*((vw_div_vw0/ed)*ID_arr[i])
#         re += 0.5 * dy * (tmp_f1 + tmp_f2)
#     return re*uZz

# def process_at_z_modi(index_i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, F2_0_modi, weight, gp_arr, gm_arr):
#     index_i = int(index_i)
#     phi_b = cond_GT['phi_bulk']
#     z = z_arr[index_i]
#     phi_arr = zeros(size(yt_arr))
#     Ieta_arr = zeros(size(yt_arr))
#     ID_arr = zeros(size(yt_arr))

#     gp_z = gp_arr[index_i]
#     gm_z = gm_arr[index_i]

#     vw = get_vw_GT_boost(z, cond_GT, Pi_arr, gp_z, gm_z)
#     vw_div_vw0 = vw/cond_GT['vw0']
#     get_phi_modi_with_fixed_z_GT(z, cond_GT, phiw_div_phib_arr[index_i]*phi_b, Pi_arr, fcn_D, vw_div_vw0, yt_arr, phi_arr)
#     get_int_eta_phi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr, Ieta_arr)
#     get_int_D_phi(z, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr, ID_arr)

#     uZz = get_u_center_GT_boost(z, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr[-1], Ieta_arr[-1], gp_z, gm_z)
#     F1 = get_F1(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
#     F2_Z = get_F2(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, yt_arr, phi_arr, Ieta_arr)
#     F2_Z_modi = get_F_modi(z, cond_GT, Pi_arr, fcn_D, fcn_eta, uZz, vw_div_vw0, yt_arr, phi_arr, Ieta_arr, ID_arr)
#     return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_0_modi - (F2_Z - F2_Z_modi))/F1)
#     # return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - F2_0_modi - (F2_Z - F2_Z_modi))/F1)

#     # return (1.-weight)*phiw_div_phib_arr[index_i] + weight*(1. + (F2_0 - (F2_Z - F2_Z_modi))/F1)
# # no ITER at z=0 introduce additional discontinuity

# def get_new_phiw_div_phib_modi_arr(cond_GT, Pi_arr, fcn_D, fcn_eta, z_arr, phiw_div_phib_arr, weight, gp_arr, gm_arr, yt_arr):
#     import time
#     phiw_new_arr = ones(size(phiw_div_phib_arr))
#     phi_b = cond_GT['phi_bulk']
#     ed = cond_GT['epsilon_d']
#     dr = cond_GT['dr']
#     R = cond_GT['R']; L = cond_GT['L']
#     dyt = dr/R
#     dyp = ed*dyt

#     Ny = size(yt_arr)
#     phi_arr_z0 = zeros(Ny)
#     Ieta_arr_z0 = zeros(Ny)
#     ID_arr_z0 = zeros(Ny)
    
#     vw_div_vw0_z0 = get_vw_GT_boost(0, cond_GT, Pi_arr, gp_arr[0], gm_arr[0])/cond_GT['vw0']        
    
#     get_phi_modi_with_fixed_z_GT(0, cond_GT, phiw_div_phib_arr[0]*phi_b, Pi_arr, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0)
#     get_int_eta_phi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, yt_arr, phi_arr_z0, Ieta_arr_z0)
#     get_int_D_phi(0, cond_GT, Pi_arr, fcn_D, yt_arr, phi_arr_z0, ID_arr_z0)
#     uZ0 = get_u_center_GT_boost(0, cond_GT, Pi_arr, fcn_D, fcn_eta,  phi_arr_z0[-1], Ieta_arr_z0[-1], gp_arr[0], gm_arr[0])
#     F2_0 = get_F2(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, yt_arr, phi_arr_z0, Ieta_arr_z0)
#     F2_0_modi = get_F_modi(0, cond_GT, Pi_arr, fcn_D, fcn_eta, uZ0, vw_div_vw0_z0, yt_arr, phi_arr_z0, Ieta_arr_z0, ID_arr_z0)
#     # F2_0 = uZ0/4.
#     # the difference between uZ0/4 and get_F2 is less than 1%.
#     # F2_0_modi = 0.
    
#     for i in range(size(z_arr)):
#         phiw_new_arr[i] = process_at_z_modi(i, cond_GT, z_arr, yt_arr, Pi_arr, fcn_D, fcn_eta, phiw_div_phib_arr, phiw_new_arr, F2_0, F2_0_modi, weight, gp_arr, gm_arr)
    
#     return phiw_new_arr



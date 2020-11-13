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

def get_cond(cond_CT, dr, weight):
    if (cond_CT['COND'] != 'CT'):
        print('Error: inherit non-CT type of dictionary in get_cond is not supported.')
        
    COND_TYPE       = 'GT'
    re              = cond_CT.copy()
    re['COND']      = COND_TYPE # update the given type to GT (general transport properties)
    # re['phi_bulk']  = phi_bulk
    re['dr']        = dr        # reference dr. note that the actual dr will be adjusted in accordance with boundary layer analysis.
    re['weight']    = weight    # weight for under-relaxed FPI operator
    return re


# convinient wrapper functions using dictionary of conditions

def get_P_conv(r_div_R, z_div_L, cond_GT, gp, gm):
    return get_P(r_div_R, z_div_L, cond_GT['Pper_div_DLP'], cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp, gm)

def get_u_conv(r_div_R, z_div_L, cond_GT, gp, gm, int_Y):
    return get_u(r_div_R, z_div_L, cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp, gm, int_Y)

def get_v_conv(r_div_R, z_div_L, Pi_div_DLP, cond_GT, gp, gm):
    return get_v(r_div_R, z_div_L, Pi_div_DLP, cond_GT['k'], cond_GT['alpha_ast'], cond_GT['Bp'], cond_GT['Bm'], gp, gm)

# flow profiles 

def get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm):
    """ Using expression in P=P^out in Eqs. (45) and (49)
    There is no difference from CT.get_P
    """
    return CT.get_P(r_div_R, z_div_L, Pper_div_DLP, k, Bp, Bm, gp, gm)


def get_u(r_div_R, z_div_L, k, Bp, Bm, gp, gm, int_Y):
    """ Using expressions u in Eq. (45) 
    and integrate 1/eta from r to 1 is reversed from 0 to y (sign change is already applied)
    u_Z^out is given in following Eq. (45)
    """
    
    y_div_R = 1. - r_div_R
    # int_Y = INT_Ieta_yt_with_fixed_z(y_div_R) # for variable y, we should use the interpolation function
    
    uR = (1. + r_div_R)*int_Y
    uZ_out = -k*(exp(k*z_div_L)*(Bp + gm) \
                 - exp(-k*z_div_L)*(Bm + gp))

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

def gen_y_div_R_arr(cond_GT):
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


def gen_phi_wrt_yt(z_div_L, phiw, fcn_D, vw_div_vw0, y_div_R_arr, phi_arr, cond_GT):
    """ Generating phi with respect to y_div_R using expression of phi Eq. (49).

    Note that the matched asymptotic phi is the implicit equation, which make difficult
    to use Runge-Kutta function in scipy directly.

    Instead, we define our own y_div_R evolution based on Runge-Kutta 4th order method.
    This is supported by cal_f_RK function defined above.
    """
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    
    phi = phiw
    int_INV_D_pre = 0.
    for i in range(1, size(y_div_R_arr)):
        y2 = y_div_R_arr[i]; y1 = y_div_R_arr[i-1]
        dy = y2 - y1        
        yh = y1 + dy/2. # for RK4 method        

        phi_1 = phi_arr[i-1]
        k1 = dy * cal_f_RK(y1, 0., phi_1, 0., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k2 = dy * cal_f_RK(y1, dy/2., phi_1, k1/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k3 = dy * cal_f_RK(y1, dy/2., phi_1, k2/2., int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
        k4 = dy * cal_f_RK(y1, dy, phi_1, k3, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT)
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


def cal_F2_0(vw_div_vw0_z0, ed, yt_arr, Ieta_arr_z0, ID_arr_z0, uZ_z0):
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

    re_F2_0 *= uZ_z0
    return re_F2_0

def cal_int_Fz(given_re_F2_0, vw_div_vw0, ed, yt_arr, Ieta_arr, ID_arr, uZ_zi):
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
        
    re_F1_Z *= uZ_zi
    re_F2_Z *= uZ_zi
    return 1. + (given_re_F2_0 - re_F2_Z)/re_F1_Z

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
    for i in range(size(val_pre)):
        val_new[i] = (1. - weight)*val_pre[i] + weight*val_new[i]
    return 0

def gen_new_phiw_div_phib_arr(phiw_div_phib_arr_new, cond_GT, fcn_D, fcn_eta, z_div_L_arr, phiw_div_phib_arr, Pi_div_DLP_arr, weight, gp_arr, gm_arr, yt_arr):
    """ Calculation phi_w/phi_b at the given z using Eq. (D3)
    The detailed terms in Eq. (D3) is explained in the function cal_int_Fz which calculate the integration.
    """
    phi_b = cond_GT['phi_bulk']
    ed = cond_GT['epsilon_d']
    
    Ny = size(yt_arr)
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
    
    vw_div_vw0_z0 = get_v_conv(rw_div_R, z0_div_L, Pi_div_DLP_arr[ind_z0], cond_GT, gp_arr[ind_z0], gm_arr[ind_z0])
    gen_phi_wrt_yt(z0_div_L, phiw_div_phib_arr[ind_z0]*phi_b, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0, cond_GT)
    gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_z0, Ieta_arr_z0, fcn_eta, cond_GT)
    gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_z0, ID_arr_z0, fcn_D, cond_GT)

    uZ_z0 = get_u_conv(r0_div_R, z0_div_L, cond_GT, gp_arr[ind_z0], gm_arr[ind_z0], Ieta_arr_z0[-1])
    F2_0 = cal_F2_0(vw_div_vw0_z0, ed, yt_arr, Ieta_arr_z0, ID_arr_z0, uZ_z0)

    for i in range(1, size(z_div_L_arr)):
        vw_div_vw0_zi = get_v_conv(rw_div_R, z_div_L_arr[i], Pi_div_DLP_arr[i], cond_GT, gp_arr[i], gm_arr[i])
        gen_phi_wrt_yt(z_div_L_arr[i], phiw_div_phib_arr[i]*phi_b, fcn_D, vw_div_vw0_zi, yt_arr, phi_arr_zi, cond_GT)
        gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, Ieta_arr_zi, fcn_eta, cond_GT)
        gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, ID_arr_zi, fcn_D, cond_GT)
        uZ_zi = get_u_conv(r0_div_R, z_div_L_arr[i], cond_GT, gp_arr[i], gm_arr[i], Ieta_arr_zi[-1])
        
        phiw_div_phib_arr_new[i] = cal_int_Fz(F2_0, vw_div_vw0_zi, ed, yt_arr, Ieta_arr_zi, ID_arr_zi, uZ_zi)

    FPI_operator(cond_GT['weight'], phiw_div_phib_arr, phiw_div_phib_arr_new)

    # # this part is for recording the analysis part
    # ind_max_z = argmax(phiw_div_phib_arr_new)
    # report_n_iter[1] = z_div_L_arr[ind_max_z]*cond_GT['L']
    # report_n_iter[2] = phiw[ind_max_z]
    # report_n_iter[3] = phiw[-1]
    
    return 0



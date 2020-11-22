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
from aux_functions import *
import sol_CT as CT
from scipy.interpolate import interp1d
from copy import deepcopy

def get_cond(cond_CT, Nr, weight):
    if (cond_CT['COND'] != 'CT'):
        print('Error: inherit non-CT type of dictionary in get_cond is not supported.')
        
    COND_TYPE       = 'GT'
    re              = cond_CT.copy()
    re['COND']      = COND_TYPE # update the given type to GT (general transport properties)
    # re['phi_bulk']  = phi_bulk
    # re['dr']        = dr        # reference dr. note that the actual dr will be adjusted in accordance with boundary layer analysis.
    re['Nr']        = Nr
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


def get_uZ_out(z_div_L, k, Bp, Bm, gp, gm):
    """ Using expression below Eq. (45):
    uZ_out(z) = -dP_out(z)/dz
    """
    uZ_out = -k*(exp(k*z_div_L)*(Bp + gm) \
                 - exp(-k*z_div_L)*(Bm + gp))
    return uZ_out

def get_u(r_div_R, z_div_L, k, Bp, Bm, gp, gm, int_Y):
    """ Using expressions u in Eq. (45) 
    and integrate 1/eta from r to 1 is reversed from 0 to y (sign change is already applied)
    u_Z^out is given in following Eq. (45)
    """
    
    # y_div_R = 1. - r_div_R
    # int_Y = INT_Ieta_yt_with_fixed_z(y_div_R) # for variable y, we should use the interpolation function
    
    uR = (1. + r_div_R)*int_Y
    uZ_out = -k*(exp(k*z_div_L)*(Bp + gm) \
                 - exp(-k*z_div_L)*(Bm + gp))

    return uZ_out * uR


def get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm):
    """ Using expression v=v^out in Eqs. (45) and (49)
    As described in CT.get_v, sign is positive because we are using coordinate function r here
    """
    # sign = +1.
    # rw_div_R = 1.
    # vw = CT.get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm)
    # vR = 2.*r_div_R - r_div_R**3.0
    return CT.get_v(r_div_R, z_div_L, Pi_div_DLP, k, alpha_ast, Bp, Bm, gp, gm)


# for y-directional information related with outer solution


def gen_y_div_R_arr(cond_GT):
    """ Generating discretized dimensionless y-coordinate with adaptive step size
    The selected way for the adaptive step size is only for the temporary
    There are revised version, which will be applied for the near future.
    """

    # generating temporal yt_arr using constant step size on the log-scale of yt
    # Ny = int(cond_GT['R']/cond_GT['dr'])
    Ny = int(cond_GT['Nr'])
    yt_arr = zeros(Ny)

    Ny_tmp = Ny*10
    yt_tmp_arr = linspace(-6, 0, Ny_tmp)
    yt_tmp_arr = 10.**yt_tmp_arr
    yt_tmp_arr[0] = 0.

    # generating arc_length of (phi^CT - phi_b)/(phi_w - phi_b) using vw_div_vw0=1 and CT assumption
    # this is only approximately interprete the arc length, which, however,
    # CT could be the maximum stiffness compared with the general concentration-dependent transport properties.
    # In addition, vw_div_vw0=1 is not too bad to interprete the about the measure of arc-length
    larc_yt_tmp_arr = zeros(Ny_tmp)
    ed = cond_GT['epsilon_d']
    for i in range(1, Ny_tmp):
        y2_tmp = yt_tmp_arr[i]
        y1_tmp = yt_tmp_arr[i-1]
        dy_tmp = y2_tmp - y1_tmp
        
        l_arc_2 = sqrt(1. + (exp(-y2_tmp/ed)/ed)**2.0)
        l_arc_1 = sqrt(1. + (exp(-y1_tmp/ed)/ed)**2.0)

        larc_yt_tmp_arr[i] = larc_yt_tmp_arr[i-1] + 0.5 * dy_tmp * (l_arc_1 + l_arc_2)

    total_arc_length = larc_yt_tmp_arr[-1]
    dl = total_arc_length/float(Ny)
    int_larc_vs_yt_arr = interp1d(larc_yt_tmp_arr, yt_tmp_arr, kind='cubic')


    l_i = 0.
    larc_arr = zeros(Ny)
    for i in range(1, Ny):
        l_tmp = dl * i + l_i
        yt_arr[i] = int_larc_vs_yt_arr(l_tmp)
        larc_arr[i] = l_tmp        

    # re_test = zeros([Ny, 2])
    # re_test[:,0] = yt_arr
    # re_test[:,1] = larc_arr
    # savetxt('tset_yt_arr.dat', re_test)
        
    
    # dy = cond_GT['dr']
    # dyt = dy/cond_GT['R']
    # dyp = cond_GT['epsilon_d'] * dyt
    
    # tmp_yt = 0.
    # yt_arr = [tmp_yt]
    # while(tmp_yt < 1. - dyt):
    #     if tmp_yt < cond_GT['epsilon_d']:
    #         tmp_dy = dyp
    #     elif tmp_yt < 2. * cond_GT['epsilon_d']:
    #         tmp_dy = 2.*dyp
    #     elif tmp_yt < 10. * cond_GT['epsilon_d']:
    #         tmp_dy = 10.*dyp
    #     else:
    #         tmp_dy = dyt
    #     tmp_yt += tmp_dy
    #     yt_arr.append(tmp_yt)
    # yt_arr = asarray(yt_arr)
    print (yt_arr)
    print ("size = ", size(yt_arr))
    
    return yt_arr


def cal_f_RK(yt, dyt, f, df, int_INV_D_pre, vw_div_vw0, fcn_D, cond_GT):
    """ Auxiliary function for Runge-Kutta method to generate matched asymptotic phi
    The complication of input arguments and result are due to the implicit expression of phi,
    which is described in help-doc in gen_phi_wrt_yt function.

    Note that the return value is related with d_phi/d_y_tilde = (1/epsilon_d)*d_phi/d_y_bar = (1/epsilon_d)*vw/D*(phi - phi_b*(1-exp(-s_bar)))

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
    
    phi_arr[0] = phiw
    int_INV_D_pre = 0.
    Ny = size(y_div_R_arr)
    for i in range(1, Ny):
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

        # if (i>=Ny-1):
        #     print (i, y2, dy, yh, phi_arr[0], phi_2, int_INV_D_pre)
        #     print (k1, k2, k3, k4)

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
        re += 0.5 * dy * (tmp_f1 + tmp_f2)
        INT_inv_f_arr[i] = re
    return 0

def cal_F2_Z(vw_div_vw0, ed, yt_arr, Ieta_arr, ID_arr, uZ_zi):
    """ [Overhead version] Calculate F2_Z defined in cal_int_Fz
    This function will calculate T_z[1-s_bar * e^{s_bar}] independently from cal_int_Fz.
    This might be useful for the analysis of the data.
    """
    re_F2_Z = 0.
    tmp_F2_Z_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*(1. - exp(-(vw_div_vw0/ed)*ID_arr[0])*((vw_div_vw0/ed)*ID_arr[0]))
    tmp_F2_Z_2 = tmp_F2_Z_1

    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]

        tmp_F2_Z_1 = tmp_F2_Z_2
        tmp_F2_Z_2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*(1. - exp(-(vw_div_vw0/ed)*ID_arr[i])*((vw_div_vw0/ed)*ID_arr[i]))
        re_F2_Z += 0.5 * dy * (tmp_F2_Z_1 + tmp_F2_Z_2)
        
    return re_F2_Z * uZ_zi

def cal_F1_Z(vw_div_vw0, ed, yt_arr, Ieta_arr, ID_arr, uZ_zi):
    """ [Overhead version] Calculate F1_Z defined in cal_int_Fz
    This function will calculate T_z[e^{s_bar}] independently from cal_int_Fz.
    This might be useful for the analysis of the data.
    """
    re_F1_Z = 0.
    tmp_F1_Z_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*exp(-(vw_div_vw0/ed)*ID_arr[0])
    tmp_F1_Z_2 = tmp_F1_Z_1

    for i in range(1, size(yt_arr)):
        dy = yt_arr[i] - yt_arr[i-1]
        tmp_F1_Z_1 = tmp_F1_Z_2
        tmp_F1_Z_2 = (1. - yt_arr[i])*(2. - yt_arr[i])*Ieta_arr[i]*exp(-(vw_div_vw0/ed)*ID_arr[i])
        re_F1_Z += 0.5 * dy * (tmp_F1_Z_1 + tmp_F1_Z_2)
        
    return re_F1_Z * uZ_zi

def cal_Phi_div_Phiast_conv(phiw, phi_bulk, F1_Z, F2_Z):
    """ [Auxiliary function] Calculate Phi(z)/Phi_ast using definition of F1_Z and F2_Z in cal_int_Fz, and Eqs. (49), (50), (D1).
    The original definition of Phi(z) in Eq. (50) is divided by Phi_ast=pi*R^2*phi_bulk*u_HP in accordance with caption of Fig. 11.
    By definition of T_z[phi] in Eq. (D1), Phi(z)/Phi_ast = (2/phi_bulk)*T_z[phi].
    By definition of matched asymptotic phi in Eq. (49), phi = (phiw - phi_bulk)*exp(-s_bar) + phi_bulk*(1 - s_bar*exp(-s_bar)).
    Therefore, we have Phi(z)/Phi_ast = (2/phi_bulk)*((phiw - phi_bulk)*F1_Z + phi_bulk*F2_Z),
    where F1_Z and F2_Z are defined on cal_int_Fz function.
    """
    return (2./phi_bulk)*((phiw-phi_bulk)*F1_Z + phi_bulk*F2_Z)



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
    tmp_F2_Z_1 = (1. - yt_arr[0])*(2. - yt_arr[0])*Ieta_arr[0]*(1. - exp(-(vw_div_vw0/ed)*ID_arr[0])*((vw_div_vw0/ed)*ID_arr[0]))
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

def FPI_operator(weight, val_pre, val_new, N_skip=0):
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
    for i in range(N_skip, size(val_pre)):
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

    ind_z0 = 0 #z-index at inlet
    
    z0_div_L = 0. #z-coord at inlet
    
    r0_div_R = 0. #r-coord at the centerline of pipe
    rw_div_R = 1. #r-coord at the membrane wall
    
    vw_div_vw0_z0 = get_v_conv(rw_div_R, z0_div_L, Pi_div_DLP_arr[ind_z0], cond_GT, gp_arr[ind_z0], gm_arr[ind_z0])
    # print (0)
    gen_phi_wrt_yt(z0_div_L, phiw_div_phib_arr[ind_z0]*phi_b, fcn_D, vw_div_vw0_z0, yt_arr, phi_arr_z0, cond_GT)
    # print (1)
    gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_z0, Ieta_arr_z0, fcn_eta, cond_GT)
    gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_z0, ID_arr_z0, fcn_D, cond_GT)

    uZ_z0 = get_uZ_out(z0_div_L, cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp_arr[ind_z0], gm_arr[ind_z0])
    F2_0 = cal_F2_Z(vw_div_vw0_z0, ed, yt_arr, Ieta_arr_z0, ID_arr_z0, uZ_z0)

    Nz = size(z_div_L_arr)
    for i in range(1, Nz):
        # print (i, 1)
        vw_div_vw0_zi = get_v_conv(rw_div_R, z_div_L_arr[i], Pi_div_DLP_arr[i], cond_GT, gp_arr[i], gm_arr[i])
        # print (i, 2)        
        gen_phi_wrt_yt(z_div_L_arr[i], phiw_div_phib_arr[i]*phi_b, fcn_D, vw_div_vw0_zi, yt_arr, phi_arr_zi, cond_GT)
        # print (i, 3)
        gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, Ieta_arr_zi, fcn_eta, cond_GT)
        # print (i, 4)
        gen_INT_inv_f_wrt_yt(yt_arr, phi_arr_zi, ID_arr_zi, fcn_D, cond_GT)
        # print (i, 5)
        uZ_zi = get_uZ_out(z_div_L_arr[i], cond_GT['k'], cond_GT['Bp'], cond_GT['Bm'], gp_arr[i], gm_arr[i])
        
        phiw_div_phib_arr_new[i] = cal_int_Fz(F2_0, vw_div_vw0_zi, ed, yt_arr, Ieta_arr_zi, ID_arr_zi, uZ_zi)

    FPI_operator(cond_GT['weight'], phiw_div_phib_arr, phiw_div_phib_arr_new, N_skip=1) # phiw(0) must be phib.

    return 0


# def process_at_z():
    

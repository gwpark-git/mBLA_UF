#############################################################################
#   Flow profile of pure solvent flow in CF UF                              #
#   This is the solution of pure solvent flow described in Sec. IV A        #
#   of the paper described below.                                           #
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
from membrane_geometry_functions import *

def cal_DTP_HP(Pin, Pout, Pper):
    """ Calculate Delta_T P for Hagen-Poiseuille (HP) flow using Eq. (8)
    No particle-contributed osmotic pressure
    """
    return (1/2.)*(Pin + Pout) - Pper

def cal_DTP_PS(Pin, Pout, Pper, k):
    """ Calculate Delta_T P for Pure Solvent (PS) flow using Eqs. (8) and (34)
    No particle-contributed osmotic pressure
    """
    return cal_DTP_HP(Pin, Pout, Pper) * 2. * tanh(k/2.)/k

def get_Pin(DLP, Pout):
    """ Calculate Pin for given DLP and Pout
    """
    return DLP + Pout

def get_Pper(DLP, DTP_HP, k, Pout):
    """ Calculate Pper for given DLP, DTP_HP, k, and Pout
    It is noteworthy that the given DTP is DTP_HP in Eq. (8).
    """
    return (1./2.)*DLP - DTP_HP/(2.*tanh(k/2.)/k) + Pout

def get_cond(pre_cond, Pin, Pout, Pper): # conditions for pure solvent flow
    """ return dictionary "cond" of conditions for pure solvent flow (Sec. IV A)
    dictionary "cond":
    inherite from dictionary "pre_cond" parameter described below.
    
    Parameters:
        pre_cond = {'k', 'R', 'L', 'Lp', 'eta0',               // the default configuration
                    'membrane_geometry', 'lam1', 'lam2',       // geometrical aspect
                    'define_permeability', 'h', 'kappa_Darcy', // additional permeability options
                    'BC_inlet'}                                // boundary conditions
            'k'    : system parameter k in Eq. (26)   in the dimensionless unit
            'R'    : radius of membrane channel       in the unit of m
            'Lp'   : solvent permeability on the clean membrane in Eqs. (12) and (13) 
                                                      in the unit of m/(Pa sec)
            'eta0' : pure solvent viscosity at the operating temperature 
                                                      in the unit of Pa sec
            'membrane_geometry' : geometry of membrane either 'HF', 'FMM', or 'FMS' (see new manuscript)
            'lam1' : dimensionless quantity bridges between P and U (see new manuscript)
            'lam2' : dimensionless quantity bridge between u0 and V (see new manuscript)
            'define_permeability' : which permeability value is provided either by kappa_Darcy or Lp
            'h'    : thickness of membrane. if Lp is given, 'h' is not important. However, it just set with R/2 as a reference.
            'kappa_Darcy' : Darcy's permeability. If Lp is provided, this value is just recalculated based on h=R/2. 
            'BC_inlet' : Specified boundary condition at the inlet either 'pressure' or 'velocity'


        Pin      = Pressure inlet boundary condition  in the unit of Pa
        Pout     = Pressure outlet boundary condition in the unit of Pa
        Pper     = Pressure in permeate which affect to Darcy-Starling law in Eq. (12) 
                                                      in the unit of Pa
    New update (9 JUN 2021): 
        pre_cond now contains 'membrane_geometry' which identify 'HF' (default), 'FMM', and 'FMS'
        in addition, lam1 and lam2 that related with the geometrical aspect is also introduced.
        Note that k = lam1*lam2* ... values, which means the code in cal_phiw_from_input.py should use k immediately from here.
    """
    cond = pre_cond.copy()
    COND_TYPE = 'PS' # describe this condition dictionary is type of PS (pure solvent)
    
    # k = pre_cond['k'] # it is a given parameter
    
    DLP = Pin - Pout                    # Longitudinal pressure difference
    DTP_HP = cal_DTP_HP(Pin, Pout, Pper) # Length-averaged TMP with linear pressure approximation in Eq. (8)
    DTP_PS = cal_DTP_PS(Pin, Pout, Pper, pre_cond['k']) # Length-averaged TMP for pure solvent flow in Eq. (7)
    vw0 = pre_cond['Lp']*DTP_HP         # v^\ast in Eq. (21)
    alpha_ast = DTP_HP/DLP              # alpha^\ast in Eq. (23)
    beta_ast = pre_cond['k']**2.0 * alpha_ast       # beta^\ast in Eq. (24) and (26)
    u_ast = pre_cond['R']**2.0 * DLP/(4.*pre_cond['eta0']*pre_cond['L']) # u_ast = u_HP when pressure inlet BC is used
    
    Ap = get_Apm_BCP(+1.0, pre_cond['k'], alpha_ast)
    Am = get_Apm_BCP(-1.0, pre_cond['k'], alpha_ast)

    cond['Ap']           = Ap
    cond['Am']           = Am
    cond['Pin']          = Pin
    cond['Pout']         = Pout
    cond['Pper']         = Pper
    cond['DLP']          = DLP
    cond['u_ast']         = u_ast
    cond['vw0']          = vw0
    cond['alpha_ast']    = alpha_ast
    cond['beta_ast']     = beta_ast
    cond['Pper_div_DLP'] = Pper/DLP
    cond['COND']         = COND_TYPE
    cond['DTP_HP']       = DTP_HP
    cond['DTP_PS']       = DTP_PS
# {            'u_ast':u_ast, 'vw0':vw0, 'alpha_ast':alpha_ast, 'beta_ast':beta_ast,\
#             'Pper_div_DLP':Pper/DLP, 'COND':COND_TYPE,\
#             'DTP_HP':DTP_HP, 'DTP_PS':DTP_PS}
    return cond


def get_Apm_BCP(pm, k, alpha_ast):
    """ Get dimensionless Apm using Eq. (32)
    """
    return pm*(1./(4.*sinh(k)))*(2.*alpha_ast - 1. - (2.*alpha_ast + 1)*exp(-pm*k))

def get_Apm_BCP_conv(pm, cond):
    """ Convinience version for get_Apm_BCP using cond
    """
    return get_Apm_BCP(pm, cond['k'], cond['alpha_ast'])

def get_Apm_BCu(pm, k, alpha_ast):
    """ Get dimensionless Apm using Eq. (32)
    """
    return pm*(1./(4.*sinh(k)))*(2.*alpha_ast - 1. - (2.*alpha_ast + 1)*exp(-pm*k))

def get_Apm_BCu_conv(pm, cond):
    """ Convinience version for get_Apm_BCP using cond
    """
    return get_Apm_BCP(pm, cond['k'], cond['alpha_ast'])


def get_P(z_div_L, k, Ap, Am, Pper_div_DLP):
    """ Using Eq. (31) (the first expression)
    """
    return Pper_div_DLP + Ap*exp(k*z_div_L) + Am*exp(-k*z_div_L)

def get_P_conv(z_div_L, cond):
    return get_P(z_div_L, cond['k'], cond['Ap'], cond['Am'], cond['Pper_div_DLP'])

def get_u(r_div_R, z_div_L, k, Ap, Am, lam1):
    """ Using Eq. (31) (the second expression)
    """
    uR_HP = (1. - r_div_R**2.0)*lam1/2.
    uZ_PS = -k*(exp( k*z_div_L)*Ap - exp(-k*z_div_L)*Am)
    return uZ_PS*uR_HP

def get_u_conv(r_div_R, z_div_L, cond):
    return get_u(r_div_R, z_div_L, cond['k'], cond['Ap'], cond['Am'], cond['lam1'])

def get_v(r_div_R, z_div_L, k, alpha_ast, Ap, Am, membrane_geometry):
    """ Using Eq. (31) (the third expression)
    """
    sign = +1.
    # vR = 2.*r_div_R - r_div_R**3.0
    vR = fcn_VR(r_div_R, membrane_geometry)
    vw =(exp( k*z_div_L)*Ap + exp(-k*z_div_L)*Am)/alpha_ast

    return sign*vR*vw

def get_v_conv(r_div_R, z_div_L, cond):
    return get_v(r_div_R, z_div_L, cond['k'], cond['alpha_ast'], cond['Ap'], cond['Am'], cond['membrane_geometry'])


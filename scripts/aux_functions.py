from numpy import *

def length_average_f(x_arr, f_arr, Lx, dx=0):
    Nx = size(f_arr)

    if dx>0:
        return (dx/Lx)*(sum(f_arr) - 0.5*(f_arr[0] + f_arr[-1]))
    else:
        re = 0.        
        for i in range(1, Nx):
            x1 = x_arr[i-1]
            x2 = x_arr[i]
            re += 0.5 * (x2-x1) * (f_arr[i] + f_arr[i-1])
    return re/Lx


def get_psi(s_bar, phi_b, phi_w): 
    """Return psi 
        psi = particle distribution function along y in Eq. (55)
            : get_psi(s_bar(y_bar, z), phi_b, phi_w(z)) = psi(y_bar, z)

    Parameters:
        s_bar = important exponent described in Eq. (43)
        phi_b = particle volume fraction in bulk/feed 
        phi_w = particle wall volume fraction 

    """
    return exp(-s_bar) - (phi_b/(phi_w - phi_b))*(s_bar*exp(-s_bar))

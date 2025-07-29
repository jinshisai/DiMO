import numpy as np
from numba import jit, njit, prange, config


@njit(parallel=True)# cache=True)
def fastbuild_twocompdisk(R, phi, z, Rmid, 
    log_N_gc, rc_g, gamma_g, Tg0, qg,
    log_Sig_dc, rc_d, gamma_d, Td0, qd,
    dv0, pdv, r0,
    msG, inc_rad, vsys,
    mu, mmol, dv_mode, pterm,
    kb, mH, auTOcm):

    # size
    nd = R.size
    ndd = Rmid.size

    # out array
    # gas
    T_g = np.zeros(nd)
    n_g = np.zeros(nd)
    vlos = np.zeros(nd)
    dv = np.zeros(nd)
    # dust
    T_d = np.zeros(ndd)
    n_d = np.zeros(ndd)


    # vrot
    if pterm:
        vrot = lambda r, msG, z, rc, gamma, q, cs2: \
        vrot_ssdisk(r, msG, z, rc, gamma, q, cs2)
    else:
        vrot = lambda r, msG, z, rc, gamma, q, cs2: vkep(r, msG, z)

    # linewidth
    if dv_mode == 'total':
        linewidth = lambda r, dv0, r0, pdv, kb, Tg, mmol, mH: dv0 * (r / r0)**(- pdv)
    elif dv_mode == 'thermal':
        linewidth = lambda r, dv0, r0, pdv, kb, Tg, mmol, mH: \
        linewidth_tnt(r, dv0, r0, pdv, kb, Tg, mmol, mH)

    # to calculate scale height later
    _cs = np.sqrt(
        sq_sound_speed(kb, Tg0, mu, mH)) # scale height at r0
    _Omega = np.sqrt(msG / (r0*auTOcm)**3. )
    h0 = _cs/_Omega / auTOcm # in au
    ph = - 0.5 * qg + 1.5

    for i in prange(nd):
        Ri = R[i]
        phii = phi[i]
        zi = z[i]

        # gas temperature
        T = Tg0 * (Ri / r0)**(-qg)
        T_g[i] = T
        cs2 = sq_sound_speed(kb, T, mu, mH)

        # density
        # surface density
        N_g = sigma_ssdisk(Ri, 10.**log_N_gc, rc_g, gamma_g,)
        # scale height
        h = h0 * (Ri / r0)**ph
        # to density
        n_g[i] = puff_up_layer(N_g, Ri, zi, h) / auTOcm # cm^-3

        # velocity
        vphi = vrot(Ri * auTOcm, msG, zi * auTOcm,
            rc_g * auTOcm, gamma_g, qg, cs2)
        vlos[i] = vphi * np.cos(phii) * np.sin(inc_rad) * 1.e-5 + vsys

        # dv
        dv[i] = linewidth(Ri, dv0, r0, pdv, kb, T, mmol, mH)


    for i in prange(ndd):
        Rmidi = Rmid[i]
        T_d[i] = Td0 * (Rmidi / r0)**(-qd)
        n_d[i] = sigma_ssdisk(Rmidi, 10.**log_Sig_dc, rc_d, gamma_d,)


    return T_g, n_g, vlos, dv, T_d, n_d


@jit(nopython=True)
def sigma_ssdisk(r, sigma_c, rc, gamma,):
    return sigma_c * (r/rc)**(- gamma) * np.exp(-(r/rc)**(2. - gamma))


@jit(nopython=True)
def puff_up_layer(sig, R, z, H):
    '''
    Without approximation of z<<R.
    '''
    rho0 = sig / np.sqrt(2. * np.pi) / H
    exp = np.exp(R**2. / H**2. * ((1. + z**2/R**2)**(-0.5) - 1.))
    return rho0 * exp


@jit(nopython=True)
def vrot_ssdisk(r, msG, z, rc, gamma, q, cs2):
    '''
    The pressure gradient term will be analytically calculated
    '''
    vkep2 = vkep(r, msG, z)**2.
    vrot2 = vkep2 - cs2 * ((2.-gamma) * (r/rc)**(2.-gamma) + q + gamma)
    return np.sqrt(vrot2)


@jit(nopython=True)
def vkep(r, msG, z):
    return np.sqrt(msG * r * r / (r*r + z*z)**(1.5))


@jit(nopython=True)
def sq_sound_speed(kb, T, mu, mH):
    return kb * T / mu / mH


@jit(nopython=True)
def linewidth_tnt(r, dv0, r0, pdv, kb, Tg, mmol, mH):
    vth = np.sqrt(2. * kb * Tg / mmol / mH) * 1.e-5 # km/s
    vnth = dv0 * (r / r0)**(- pdv)
    return np.sqrt(vth * vth + vnth * vnth)
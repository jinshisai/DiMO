import numpy as np
from astropy import constants, units
from numba import jit, njit, prange


from ..fast_grid import fast_3d_collapse


# constants
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant


@jit(nopython=True)
def glnprof(t0: float, 
    v: np.ndarray, 
    v0: float, delv: float, fn: float):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (float): Total optical depth or integrated intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. t0 will be in units of inverse of velocity if fn is not given.
    '''
    exponent =  - (v - v0) **2 / delv**2
    return t0 / (np.sqrt(np.pi) * delv) * np.exp(exponent) * fn


@jit(nopython=True)
#@njit(parallel=True)
def interp(x, xp, fp):
    """
    Perform linear interpolation or extrapolation.

    Parameters:
    -----------
    x : float
        The x-value for which to interpolate.
    xp : ndarray
        The x-coordinates of the data points, must be sorted.
    fp : ndarray
        The y-coordinates of the data points.

    Returns:
    --------
    float
        Interpolated or extrapolated value at x.
    """
    if x < xp[0]:  # Below the first point, extrapolate
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return fp[0] + slope * (x - xp[0])
    elif x > xp[-1]:  # Above the last point, extrapolate
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return fp[-1] + slope * (x - xp[-1])
    else:  # Interpolate within bounds
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])


@njit(parallel=True) # fastmath=True
def Tndv_to_cube(T_g, n_gf, n_gr, vlos, dv, ve, ds,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute a 4D cube from temperature, density, and velocity information,
    with optical depth calculation and integration termination based on tau.

    Parameters:
        T_g: 3D numpy array for temperature.
        n_gf: 3D numpy array for density (foreground).
        n_gr: 3D numpy array for density (rear side).
        vlos: 3D numpy array for line-of-sight velocities.
        dv: 3D numpy array for velocity dispersions.
        ve: 1D numpy array for velocity edges.
        ds: Step size (float).

    Returns:
        Tncube: 4D numpy array of the resulting cube.
    """
    nx, ny, nz = T_g.shape
    nv = len(ve) - 1
    delv = (ve[1] - ve[0]) * 1.e5 # channel width (cm s^-1)
    Tncube = np.zeros((4, nx, ny, nv))

    for i in prange(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l + 1])
                T_gf_sum = 0.0
                T_gr_sum = 0.0
                n_gf_sum = 0.0
                n_gr_sum = 0.0
                tau_gf = 0.0
                tau_gr = 0.0

                for k in range(nz):
                    if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                        #print('Went over 30!')
                        break

                    dv_cell = dv[i, j, k]
                    vlos_cell = vlos[i, j, k]
                    if (vl - vlos_cell) ** 2 < 25.0 * dv_cell**2. * 0.5: # if less than 5 sigma
                        T_cell = T_g[i, j, k]

                        # Foreground side
                        if tau_gf < 30.0:
                            n_v_gf = glnprof(n_gf[i, j, k] * ds, vl, vlos_cell, dv_cell, 1.0)
                            n_gf_sum += n_v_gf
                            T_gf_sum += T_cell * n_v_gf

                            if n_gf_sum > 0.0:
                                T_gf_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gf_mn, Qgrid[0], Qgrid[1])
                                tau_gf = NT_to_tau(n_gf_sum, T_gf_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                        # Rear side
                        if tau_gr < 30.0:
                            n_v_gr = glnprof(n_gr[i, j, k] * ds, vl, vlos_cell, dv_cell, 1.0)
                            n_gr_sum += n_v_gr
                            T_gr_sum += T_cell * n_v_gr

                            if n_gr_sum > 0.0:
                                T_gr_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gr_mn, Qgrid[0], Qgrid[1])
                                tau_gr = NT_to_tau(n_gr_sum, T_gr_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                # Calculate mean temperature and store results
                if n_gf_sum > 0.0:
                    Tncube[0, i, j, l] = T_gf_mn
                    Tncube[2, i, j, l] = tau_gf

                if n_gr_sum > 0.0:
                    Tncube[1, i, j, l] = T_gr_mn
                    Tncube[3, i, j, l] = tau_gr

    return Tncube



@jit(nopython=True)
def NT_to_tau(Ntot, Tex, freq, Aul, Eu, gu, Qrot, delv):
    '''
    Calculate the line optical depth.

    Parameters
    ----------
     N (float or ndarray): Number column density of the molecule (cm^-2).
     T (float or ndarray): Excitation temperature for the line (K).
     freq (float): Frequency of the line (Hz).
     Eu (float): Energy of the upper energy state (K).
     gu (float): g of the upper energy state.
     Aul (float): Einstein A coeffient of the transition.
     Qrot (float or ndarray): Partition function.
     delv (float): Linewidth or channel width (cm s^-1).
    '''
    return (clight*clight*clight)/(8.*np.pi*freq*freq*freq)*(gu/Qrot)\
    *np.exp(-Eu/Tex)*Ntot*Aul*(np.exp(hp*freq/(kb*Tex)) - 1.) / delv



@jit(nopython=True)
def nT_to_alpha(ntot, Tex, freq, Aul, Eu, gu, Qrot, delv):
    '''
    Calculate the line optical depth.

    Parameters
    ----------
     N (float or ndarray): Number column density of the molecule (cm^-2).
     T (float or ndarray): Excitation temperature for the line (K).
     freq (float): Frequency of the line (Hz).
     Eu (float): Energy of the upper energy state (K).
     gu (float): g of the upper energy state.
     Aul (float): Einstein A coeffient of the transition.
     Qrot (float or ndarray): Partition function.
     delv (float): Linewidth or channel width (cm s^-1).
    '''
    return (clight*clight*clight)/(8.*np.pi*freq*freq*freq)*(gu/Qrot)\
    *np.exp(-Eu/Tex)*ntot*Aul*(np.exp(hp*freq/(kb*Tex)) - 1.) / delv



@njit(parallel=True) # fastmath=True
def Tnlnp_to_cube(Tg, n_gf, n_gr, 
    lnprof_series, dz,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute a 4D cube from temperature, density, and velocity information,
    with optical depth calculation. and integration termination based on tau.

    Parameters
    ----------
     T_g: 3D numpy array for temperature.
     n_gf: 3D numpy array for density (foreground).
     n_gr: 3D numpy array for density (rear side).
     lnprof_series: 4D numpy array for line profile function.
     delv: 3D array for line width
     dv: velocity resolution
     dz: cell size along z-axis (cm).
     freq: Frequency
     Aul: Einstein A coefficient
     Eu: Energy of upper state (K)
     gu:
     Qgrid: 

    Returns:
        Tncube: 4D numpy array of the resulting cube.
    """
    nv, nx, ny, nz = lnprof_series.shape

    # output
    Tv_gf = np.zeros((nv, nx, ny))
    Tv_gr = np.zeros((nv, nx, ny))
    tau_v_gf = np.zeros((nv, nx, ny))
    tau_v_gr = np.zeros((nv, nx, ny))

    for i in prange(nv):
        for j in range(nx):
            for k in range(ny):
                T_gf_sum = 0.0
                T_gr_sum = 0.0
                tau_gf = 0.0
                tau_gr = 0.0

                for l in range(nz):
                    if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                        #print('Went over 30!')
                        break

                    lnpf_i = lnprof_series[i,j,k,l]
                    if lnpf_i > 0: # no line contribution
                        # temperature
                        _Tg = Tg[j,k,l]
                        Qrot = interp(_Tg, Qgrid[0], Qgrid[1])

                        if n_gf[j,k,l] > 0.:
                            nv_gf = lnpf_i * n_gf[j,k,l]
                            alpha_v_gf = nT_to_alpha(
                            nv_gf, _Tg, freq, Aul, Eu, gu, Qrot, 1.
                            ) # /delv is already included in line profile function

                            # tau & temperature
                            tau_gf += alpha_v_gf # integration
                            T_gf_sum += _Tg * alpha_v_gf # weighted summation

                        if n_gr[j,k,l] > 0.:
                            nv_gr = lnpf_i * n_gr[j,k,l]
                            alpha_v_gr = nT_to_alpha(nv_gr, _Tg, freq, Aul, Eu, gu, Qrot, 1.)

                            # tau & temperature
                            tau_gr += alpha_v_gr # integration
                            T_gr_sum += _Tg * alpha_v_gr # weighted summation


                # Calculate mean temperature and store results
                if tau_gf > 0.0:
                    tau_v_gf[i, j, k] = tau_gf * dz
                    Tv_gf[i, j, k] = T_gf_sum / tau_gf

                if tau_gr > 0.0:
                    tau_v_gr[i, j, k] = tau_gr * dz
                    Tv_gr[i, j, k] = T_gr_sum / tau_gr

    return Tv_gf, Tv_gr, tau_v_gf, tau_v_gr


@njit(parallel=True) # fastmath=True
def Tnv_to_cube_old(Tg, nv_g, zs, dzs,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute a 4D cube from temperature, density, and velocity information,
    with optical depth calculation. and integration termination based on tau.

    Parameters
    ----------
     T_g: 3D numpy array for temperature.
     nv_gf: 4D numpy array for density (foreground).
     nv_gr: 4D numpy array for density (rear side).
     delv: 3D array for line width
     dv: velocity resolution
     dz: cell size along z-axis (cm).
     freq: Frequency
     Aul: Einstein A coefficient
     Eu: Energy of upper state (K)
     gu:
     Qgrid: 

    Returns:
        Tncube: 4D numpy array of the resulting cube.
    """
    nv, nx, ny, nz = nv_g.shape

    # output
    Tv_gf = np.zeros((nv, nx, ny))
    Tv_gr = np.zeros((nv, nx, ny))
    tau_v_gf = np.zeros((nv, nx, ny))
    tau_v_gr = np.zeros((nv, nx, ny))

    for i in prange(nv):
        for j in range(nx):
            for k in range(ny):
                T_gf_sum = 0.0
                T_gr_sum = 0.0
                tau_gf = 0.0
                tau_gr = 0.0

                for l in range(nz):
                    z = zs[j,k,l]
                    dz = dzs[j,k,l]
                    _nv_g = nv_g[i,j,k,l]
                    if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                        #print('Went over 30!')
                        break

                    if (_nv_g > 0.):
                         # temperature
                        _Tg = Tg[j,k,l]
                        Qrot = interp(_Tg, Qgrid[0], Qgrid[1])

                        # front side
                        if (z <= 0) and (tau_gf < 30.0):
                            alpha_v_gf = nT_to_alpha(
                            _nv_g, _Tg, freq, Aul, Eu, gu, Qrot, 1.
                            ) # /delv is already included in line profile function

                            # tau & temperature
                            tau_gf += alpha_v_gf * dz # integration
                            T_gf_sum += _Tg * alpha_v_gf * dz # weighted summation

                        # rear side
                        if (z >= 0) and (tau_gr < 30.0):
                            alpha_v_gr = nT_to_alpha(
                                _nv_g, _Tg, freq, Aul, Eu, gu, Qrot, 1.)

                            # tau & temperature
                            tau_gr += alpha_v_gr * dz # integration
                            T_gr_sum += _Tg * alpha_v_gr * dz # weighted summation


                # Calculate mean temperature and store results
                if tau_gf > 0.0:
                    tau_v_gf[i, j, k] = tau_gf
                    Tv_gf[i, j, k] = T_gf_sum / tau_gf

                if tau_gr > 0.0:
                    tau_v_gr[i, j, k] = tau_gr
                    Tv_gr[i, j, k] = T_gr_sum / tau_gr

    return Tv_gf, Tv_gr, tau_v_gf, tau_v_gr




@njit(parallel=True) # fastmath=True
def Tnv_to_cube(Tg, nv_g, zs, dzs,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute a 4D cube from temperature, density, and velocity information,
    with optical depth calculation. and integration termination based on tau.

    Parameters
    ----------
     T_g: 3D numpy array for temperature.
     nv_gf: 4D numpy array for density (foreground).
     nv_gr: 4D numpy array for density (rear side).
     delv: 3D array for line width
     dv: velocity resolution
     dz: cell size along z-axis (cm).
     freq: Frequency
     Aul: Einstein A coefficient
     Eu: Energy of upper state (K)
     gu:
     Qgrid: 

    Returns:
        Tncube: 4D numpy array of the resulting cube.
    """
    nv, nxy, nz = nv_g.shape

    # output
    Tv_gf = np.zeros((nv, nxy))
    Tv_gr = np.zeros((nv, nxy))
    tau_v_gf = np.zeros((nv, nxy))
    tau_v_gr = np.zeros((nv, nxy))

    for i in prange(nv):
        for j in range(nxy):
            T_gf_sum = 0.0
            T_gr_sum = 0.0
            tau_gf = 0.0
            tau_gr = 0.0

            for k in range(nz):
                z = zs[j,k]
                dz = dzs[j,k]
                _nv_g = nv_g[i,j,k]
                if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                    #print('Went over 30!')
                    break

                if (_nv_g > 0.):
                     # temperature
                    _Tg = Tg[j,k]
                    Qrot = interp(_Tg, Qgrid[0], Qgrid[1])

                    # front side
                    if (z <= 0) and (tau_gf < 30.0):
                        alpha_v_gf = nT_to_alpha(
                        _nv_g, _Tg, freq, Aul, Eu, gu, Qrot, 1.
                        ) # /delv is already included in line profile function

                        # tau & temperature
                        tau_gf += alpha_v_gf * dz # integration
                        T_gf_sum += _Tg * alpha_v_gf * dz # weighted summation

                    # rear side
                    if (z >= 0) and (tau_gr < 30.0):
                        alpha_v_gr = nT_to_alpha(
                            _nv_g, _Tg, freq, Aul, Eu, gu, Qrot, 1.)

                        # tau & temperature
                        tau_gr += alpha_v_gr * dz # integration
                        T_gr_sum += _Tg * alpha_v_gr * dz # weighted summation


            # Calculate mean temperature and store results
            if tau_gf > 0.0:
                tau_v_gf[i, j] = tau_gf
                Tv_gf[i, j] = T_gf_sum / tau_gf

            if tau_gr > 0.0:
                tau_v_gr[i, j] = tau_gr
                Tv_gr[i, j] = T_gr_sum / tau_gr

    return Tv_gf, Tv_gr, tau_v_gf, tau_v_gr



@njit(parallel=True) # fastmath=True
def Nv_to_tauv(T, Nv,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute tau_v from N_v.

    Parameters
    ----------
     T_g: 2D numpy array for temperature with a shape of (nv, nxy).
     Nv_g: 2D numpy array for density with a shape of (nv, nxy).
     freq: Frequency
     Aul: Einstein A coefficient
     Eu: Energy of upper state (K)
     gu:
     Qgrid: 
    """
    nv, nxy = Nv.shape

    # output
    tau_v = np.zeros((nv, nxy))

    for i in prange(nv):
        for j in range(nxy):
            _Nv = Nv[i, j]
            if (_Nv > 0.):
                _T = T[i,j]
                # partition function
                Qrot = interp(_T, Qgrid[0], Qgrid[1])
                # tau
                tau_v[i, j] = nT_to_alpha(
                    _Nv, _T, freq, Aul, Eu, gu, Qrot, 1.
                    ) # /delv is already included in line profile function
    return tau_v
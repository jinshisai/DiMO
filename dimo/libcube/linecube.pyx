import numpy as np
cimport numpy as cnp
cimport cython # compile


# to use numpy array
cnp.import_array()

# Define data type
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# laod functions from standard C library
cdef extern from "<math.h>": # from math
    DEF HUGE_VAL = 1e500     # define macro
    double exp(double x)     # function to use
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double M_PI


# Temperature and tau to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Tt_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    ):
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Ttcube = np.zeros((4, nv, nx, ny))
    cdef double T_gf_sum, T_gr_sum, tau_rho_gf_sum, tau_rho_gr_sum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                T_gf_sum = 0.
                T_gr_sum = 0.
                tau_rho_gf_sum = 0.
                tau_rho_gr_sum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        T_gf_sum += T_g[i,j,k] * tau_rho_gf[i,j,k] * ds
                        T_gr_sum += T_g[i,j,k] * tau_rho_gr[i,j,k] * ds
                        tau_rho_gf_sum += tau_rho_gf[i,j,k] * ds
                        tau_rho_gr_sum += tau_rho_gr[i,j,k] * ds
                if tau_rho_gf_sum > 0.:
                    Ttcube[0,l,i,j] = T_gf_sum / tau_rho_gf_sum # weighted mean
                    Ttcube[2,l,i,j] = tau_rho_gf_sum # sum
                if tau_rho_gr_sum > 0.:
                    Ttcube[1,l,i,j] = T_gr_sum / tau_rho_gr_sum # weighted mean
                    Ttcube[3,l,i,j] = tau_rho_gr_sum # sum
    return Ttcube



# xyz temperature and density to xyzv
# For nested grid
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] to_xyzv(
    cnp.ndarray[DTYPE_t, ndim=2] data,
    cnp.ndarray[DTYPE_t, ndim=2] lnprof,
    ):
    '''
    Convert flattend cubic data into xyzv 4D data with given line profiles.

    Parameters
    ----------
     data (2D array): flattend cubic data. Must be in shape of (nq, nd), where
      nq is number of quantities and nd is number of data that is nx * ny * nz.
     lnprof (2D array): Line profile for each cell. Must be in shape of (nv, nd),
      where nv is number of velocity cells and nd is number of data.
    '''
    cdef int nv = lnprof.shape[0]
    cdef int nd = lnprof.shape[1]
    cdef int nq = data.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=3] xyzv = np.zeros((nq, nv, nd))
    cdef int i, j, k

    for i in range(nq):
        for j in range(nv):
            for k in range(nd):
                xyzv[i,j,k] = data[i,k] * lnprof[j,k]

    return xyzv


# temperature, density and linewidth to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Tndv_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] n_gf,
    cnp.ndarray[DTYPE_t, ndim=3] n_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=3] dv,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    ):
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Tncube = np.zeros((4, nx, ny, nv))
    cdef double n_v_gf, n_v_gr
    cdef double T_gf_sum, T_gr_sum, n_gf_sum, n_gr_sum
    cdef double vl
    cdef delv = ve[1] - ve[0] # velocity resolution
    cdef int i, j, k, l
    cdef double vlos_ijk, dv_ijk, T_g_ijk

    #for l in range(nv):
    for i in range(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l+1])
                T_gf_sum = 0.
                T_gr_sum = 0.
                n_gf_sum = 0.
                n_gr_sum = 0.

                for k in range(nz):
                    vlos_ijk = vlos[i,j,k]
                    dv_ijk = dv[i,j,k]
                    T_g_ijk = T_g[i,j,k]
                    # calculate smearing effect
                    # only when the velocity separation is less than 5 Gaussian sigma
                    if (((vl - vlos_ijk)**2 < 25. * dv_ijk**2 * 0.5) and (delv < dv_ijk)):
                        # fore side
                        n_v_gf = glnprof(
                            n_gf[i,j,k] * ds, vl, vlos_ijk,
                            dv_ijk, 1.
                            ) * 1.e-5 # cm^-3 (cm s^-1)^-1

                        # rear side
                        n_v_gr = glnprof(
                            n_gr[i,j,k] * ds, vl, vlos_ijk,
                            dv_ijk, 1.
                            ) * 1.e-5 # cm^-3 (cm s^-1)^-1

                        # sumup
                        n_gf_sum += n_v_gf
                        T_gf_sum += T_g_ijk * n_v_gf
                        n_gr_sum += n_v_gr
                        T_gr_sum += T_g_ijk * n_v_gr
                    elif ((ve[l] <= vlos_ijk) and (vlos_ijk < ve[l+1])):
                        n_v_gf = n_gf[i,j,k] * ds / delv * 1.e-5 # cm^-3 (cm s^-1)^-1
                        n_v_gr = n_gr[i,j,k] * ds / delv * 1.e-5 # cm^-3 (cm s^-1)^-1

                        # sumup
                        n_gf_sum += n_v_gf
                        T_gf_sum += T_g_ijk * n_v_gf
                        n_gr_sum += n_v_gr
                        T_gr_sum += T_g_ijk * n_v_gr


                if n_gf_sum > 0.:
                    Tncube[0,i,j,l] = T_gf_sum / n_gf_sum
                    Tncube[2,i,j,l] = n_gf_sum
                if n_gr_sum > 0.:
                    Tncube[1,i,j,l] = T_gr_sum / n_gr_sum
                    Tncube[3,i,j,l] = n_gr_sum

    return Tncube


# Gaussian line broadening
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef inline double glnprof(
    double t0,
    double v,
    double v0,
    double delv,
    double fn):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (ndarray): Total optical depth or integrated intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. t0 will be in units of velocity if fn is not given.
    '''
    return t0 / sqrt(M_PI) / delv * exp( - (v - v0) * (v - v0) / delv / delv) * fn


# to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] tocube(
    cnp.ndarray[DTYPE_t, ndim=2] tau, 
    cnp.ndarray[DTYPE_t, ndim=2] vlos, 
    cnp.ndarray[DTYPE_t, ndim=1] ve):
    cdef int ny = tau.shape[0]
    cdef int nx = tau.shape[1]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] tau_v = np.zeros([nv, ny, nx], dtype = DTYPE)
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                if (ve[k] <= vlos[j,i]) and ( ve[k+1] > vlos[j,i]):
                    tau_v[k, j, i] = tau[j,i]
    return tau_v


# solve radiative transfer for three-layer model
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] solve_3LRT(
    cnp.ndarray[DTYPE_t, ndim=2] Sv_gf, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_gr, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_d, 
    cnp.ndarray[DTYPE_t, ndim=3] tau_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_gr,
    cnp.ndarray[DTYPE_t, ndim=2] tau_d,
    double Sv_bg,
    int nv):
    cdef int ny = Sv_d.shape[0]
    cdef int nx = Sv_d.shape[1]
    cdef double Iv_d
    cdef cnp.ndarray[DTYPE_t, ndim=3] Iv = np.zeros((nv, ny, nx))
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                Iv_d = (Sv_d[j,i] - Sv_bg) * (1. - exp(- tau_d[j,i]))
                Iv[k, j, i] = Sv_bg * (
                    exp(- tau_gf[k,j,i] - tau_d[j,i] - tau_gr[k,j,i]) - 1.) \
                + Sv_gr[j,i] * (1. - exp(- tau_gr[k,j,i])) \
                * exp(- tau_gf[k,j,i]  - tau_d[j,i] ) \
                + Sv_d[j,i] * (1. - exp(- tau_d[j,i])) * exp(- tau_gf[k,j,i]) \
                + Sv_gf[j,i] * (1. - exp(- tau_gf[k,j,i])) \
                - Iv_d
    return Iv


# solve radiative transfer for multi-layer model
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] solve_MLRT(
    cnp.ndarray[DTYPE_t, ndim=3] Sv_gf, 
    cnp.ndarray[DTYPE_t, ndim=3] Sv_gr, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_d, 
    cnp.ndarray[DTYPE_t, ndim=3] tau_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_gr,
    cnp.ndarray[DTYPE_t, ndim=2] tau_d,
    double Sv_bg,
    int nv):
    cdef int ny = Sv_d.shape[0]
    cdef int nx = Sv_d.shape[1]
    cdef double Iv_d
    cdef cnp.ndarray[DTYPE_t, ndim=3] Iv = np.zeros((nv, ny, nx))
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                Iv_d = (Sv_d[j,i] - Sv_bg) * (1. - exp(- tau_d[j,i]))
                Iv[k, j, i] = Sv_bg * (
                    exp(- tau_gf[k,j,i] - tau_d[j,i] - tau_gr[k,j,i]) - 1.) \
                + Sv_gr[k,j,i] * (1. - exp(- tau_gr[k,j,i])) \
                * exp(- tau_gf[k,j,i]  - tau_d[j,i] ) \
                + Sv_d[j,i] * (1. - exp(- tau_d[j,i])) * exp(- tau_gf[k,j,i]) \
                + Sv_gf[k,j,i] * (1. - exp(- tau_gf[k,j,i])) \
                - Iv_d
    return Iv


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] collapse_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k]
                cube[l,i,j] = wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] integrate_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k] * ds
                cube[l,i,j] = wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] average_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum, navg
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                navg = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k]
                        navg =+ 1.
                if navg > 0.:
                    cube[l,i,j] = wsum / navg
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] waverage_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] q,
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    '''
    Collapse 4D data into cube through weighted averaging.
    '''
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum, navg
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                qwsum = 0.
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        qwsum += q[i,j,k] * w[i,j,k]
                        wsum += w[i,j,k]
                if wsum > 0.:
                    cube[l,i,j] = qwsum / wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=2] xrot(
    cnp.ndarray[DTYPE_t, ndim=1] x, 
    cnp.ndarray[DTYPE_t, ndim=1] y, 
    cnp.ndarray[DTYPE_t, ndim=1] z, 
    double inc):
    cdef int nsize = x.size
    cdef cnp.ndarray[DTYPE_t, ndim=2] xyzp = np.zeros((3, nsize))
    cdef double cosi = cos(inc)
    cdef double sini = sin(inc)
    for i in range(nsize):
        xyzp[0,i] = x[i]
        xyzp[1,i] = y[i] * cosi - z[i] * sini
        xyzp[2,i] = y[i] * sini + z[i] * cosi
    return xyzp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] xrot3D(
    cnp.ndarray[DTYPE_t, ndim=3] x, 
    cnp.ndarray[DTYPE_t, ndim=3] y, 
    cnp.ndarray[DTYPE_t, ndim=3] z, 
    double inc):
    cdef int nx = x.shape[0]
    cdef int ny = x.shape[1]
    cdef int nz = x.shape[2]
    cdef cnp.ndarray[DTYPE_t, ndim=2] xyzp = np.zeros((3, nx,ny,nz))
    cdef double cosi = cos(inc)
    cdef double sini = sin(inc)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xyzp[0,i,j,k] = x[i,j,k]
                xyzp[1,i,j,k] = y[i,j,k] * cosi - z[i,j,k] * sini
                xyzp[2,i,j,k] = y[i,j,k] * sini + z[i,j,k] * cosi
    return xyzp


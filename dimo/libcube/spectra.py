import numpy as np
from numba import njit, prange, config
from math import erf


# Set threading layer
#config.THREADING_LAYER = 'tbb'

@njit(parallel=True, cache=True)
def glnprof_series(v, v0, delv, unit_scale = 1.):
    '''
    Generate series of normalized Gaussian line profiles.

    Parameters
    ----------
     v (1D array): velocity axis
     v0 (1D array): series of line centre
     delv (1D array): series of linewidths
    '''
    nd = v0.size
    nv = len(v)
    dv_cell = v[1] - v[0]
    v_min = v[0]
    v_max = v[-1]

    #lnprof = np.zeros((nv, nd)) # this order for later
    lnprof = np.zeros((nd, nv)) # new
    for i in prange(nd):
        v0i = v0[i]
        delvi = delv[i]
        # old version
        '''
        profi = np.exp( - (v - v0i)**2. / delvi**2.)
        sampled_fraction = 0.5 * (erf((v_max - v0i) / (delvi)) # np.sqrt(2.)
            - erf((v_min - v0i) / (delvi)))
        profi = profi / np.sum(profi * dv_cell) * sampled_fraction * unit_scale
        thr = 3.7e-5 / np.sqrt(np.pi * delvi) * unit_scale
        for k in range(nv):
            if (profi[k] <= thr):
                lnprof[i,k] = 0. # apart more than 5 sigma
            else:
                lnprof[i,k] = profi[k]
        '''

        # new
        expterm = (v - v0i)**2. / delvi**2.
        for j in range(nv):
            if expterm[j] >= 12.5: # apart more than 5 sigma
                #profi[j] = 0.
                lnprof[i,j] = 0.
            else:
                #profi[j] = np.exp(-expterm[j])
                lnprof[i,j] = np.exp(-expterm[j])
        sampled_fraction = 0.5 * (erf((v_max - v0i) / (delvi)) # np.sqrt(2.)
            - erf((v_min - v0i) / (delvi)))
        prof_int = np.sum(lnprof[i,:] * dv_cell)
        lnprof[i,:] *= sampled_fraction * unit_scale / prof_int

    return lnprof


@njit(parallel=True, cache=True)
def normalize_glnprofs(profs, v, v0, delv, unit_scale = 1.):
    '''
    Generate series of normalized Gaussian line profiles.

    Parameters
    ----------
     v (1D array): velocity axis
     v0 (1D array): series of line centre
     delv (1D array): series of linewidths
    '''
    nd = v0.size
    nv = len(v)
    dv_cell = v[1] - v[0]
    v_min = v[0]
    v_max = v[-1]

    lnprof = np.zeros((nv, nd))
    for i in prange(nd):
        profi = profs[i,:]
        v0i = v0[i]
        delvi = delv[i]
        sampled_fraction = 0.5 * (erf((v_max - v0i) / (delvi)) # np.sqrt(2.)
            - erf((v_min - v0i) / (delvi)))
        profi = profi / np.sum(profi * dv_cell) * sampled_fraction * unit_scale
        for k in range(nv):
            if (profi[k] <= 3.7e-5 / np.sqrt(np.pi * delvi) * unit_scale):
                profi[k] = 0. # apart more than 5 sigma
            else:
                pass

    return profs


# xyz temperature and density to xyzv
# For nested grid
@njit(parallel=True, cache=True)
def to_xyzv(data,lnprof,):
    '''
    Convert flattend cubic data into xyzv 4D data with given line profiles.

    Parameters
    ----------
     data (2D array): flattend cubic data. Must be in shape of (nq, nd), where
      nq is number of quantities and nd is number of data that is nx * ny * nz.
     lnprof (2D array): Line profile for each cell. Must be in shape of (nv, nd),
      where nv is number of velocity cells and nd is number of data.
    '''
    nd, nv = lnprof.shape
    xyzv = np.zeros((nd, nv))

    for i in prange(nd):
        for j in range(nv):
                xyzv[i,j] = data[i] * lnprof[i,j]

    return xyzv


@njit(parallel=True)
def Tn_to_cube(Tg, n_gf, n_gr, lnprof_series, dz):
    nv, nx, ny, nz = lnprof_series.shape

    # output
    Tv_gf = np.zeros((nv, nx, ny))
    Tv_gr = np.zeros((nv, nx, ny))
    Nv_gf = np.zeros((nv, nx, ny))
    Nv_gr = np.zeros((nv, nx, ny))

    # loop
    for i in prange(nv):
        for j in range(nx):
            for k in range(ny):
                _Nv_gf = 0.
                _Nv_gr = 0.
                _Tv_gf = 0.
                _Tv_gr = 0.

                for l in range(nz):
                    lnpf_i = lnprof_series[i,j,k,l]
                    _Tg = Tg[j,k,l]
                    nv_gf = lnpf_i * n_gf[j,k,l]
                    nv_gr = lnpf_i * n_gr[j,k,l]

                    # sum up
                    _Nv_gf += nv_gf
                    _Nv_gr += nv_gr
                    _Tv_gf += _Tg * nv_gf
                    _Tv_gr += _Tg * nv_gr

                _Nv_gf *= dz
                _Nv_gr *= dz
                if _Nv_gf > 1.e-10:
                    Nv_gf[i,j,k] = _Nv_gf
                    Tv_gf[i,j,k] = _Tv_gf * dz / _Nv_gf

                if _Nv_gr > 1.e-10:
                    Nv_gr[i,j,k] = _Nv_gr
                    Tv_gr[i,j,k] = _Tv_gr * dz / _Nv_gr

    return Tv_gf, Tv_gr, Nv_gf, Nv_gr

import numpy as np
from numba import njit, prange, config
from math import erf


# Set threading layer
#config.THREADING_LAYER = 'tbb'

@njit(parallel=True)
def glnprof_series(v, v0, delv):
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
    v_min, v_max = v[0], v[-1]

    lnprof = np.zeros((nv, nd))
    for i in prange(nd):
        profi = np.exp( - (v - v0[i])**2. / delv[i]**2.)
        sampled_fraction = 0.5 * (erf((v_max - v0[i]) / (delv[i])) # np.sqrt(2.)
            - erf((v_min - v0[i]) / (delv[i])))
        #print(sampled_fraction)
        lnprof[:,i] = profi / np.sum(profi * dv_cell) * sampled_fraction * 1.e-5
        lnprof[:,i][lnprof[:,i] <= 3.7e-5 / np.sqrt(np.pi * delv[i]) * 1.e-5 ] = 0. # 5 sigma

    return lnprof


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



@njit(parallel=True)
def to_xyzv_nested(
    ngrids,
    vlos,
    dv,
    v
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
    nl = len(ngrids)

    for l in prange(nl):
        nx, ny, nz = ngrids[l]

        _vlos = vlos[l]
        _dv = dv[l]
        # new version
        lnprofs = spectra.glnprof_series(self.v, _vlos, _dv)

        #start = time.time()
        n_cube = linecube.to_xyzv(
            np.array([n_gf[l], n_gr[l]]), lnprofs)
        #end = time.time()
        #print('to_xyzv takes %.2f'%(end-start))

        #start = time.time()
        #print('appending... at level %i'%l)
        for i in range(self.nv):
            _Tv_g[i].append(T_g[l])
            _nv_gf[i].append(n_cube[0,i,:])
            _nv_gr[i].append(n_cube[1,i,:])

    for i in range(nq):
        for j in range(nv):
            for k in range(nd):
                xyzv[i,j,k] = data[i,k] * lnprof[j,k]

    return xyzv

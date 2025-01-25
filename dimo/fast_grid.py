import numpy as np
from numba import jit, njit, prange, typed


@njit#(parallel=True) # fastmath=True
def fast_3d_collapse(d_in, nlevels,
    ngrids, nsub, xinest, yinest, zinest,
    upto = None):
    '''
    Collapse given data to the mother grid.

    Parameters
    ----------
    d (list): List of data on the nested grid
    '''

    # Use Numba array
    d = typed.List()
    for _di in d_in:
        d.append(_di)

    d_col = d[-1].reshape(ngrids[-1]) # starting from the inner most grid
    lmax = 0 if upto is None else upto
    for _l in prange(nlevels - lmax - 2):
        l = nlevels - 1 - _l # count down

        nsub_l = nsub[l-1]
        ximin = xinest[l][0]
        ximax = xinest[l][1]
        yimin = yinest[l][0]
        yimax = yinest[l][1]
        zimin = zinest[l][0]
        zimax = zinest[l][1]
        # collapse data on the inner grid
        _d = fast_binning_3d(d_col, nsub_l)

        # go next layer
        nx = ngrids[l-1][0] # size of the upper layer
        ny = ngrids[l-1][1]
        nz = ngrids[l-1][2]
        d_col = np.zeros((nx, ny, nz), dtype=np.float64) # np.full((nx, ny, nz), np.nan)
        #print('Upper layer', d_col.shape)
        #print('Insert', d_col[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1].shape)

        # insert collapsed data
        #d_col[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d
        d_col = array_substitute(d_col, _d, ximin, ximax, yimin, yimax, zimin, zimax)

        '''
        # fill upper layer data
        # Region 1: x from zero to ximin, all y and z
        d_col[:ximin, :, :] = \
        d[l-1][:ximin * ny * nz].reshape((ximin, ny, nz))
        # Region 2: x from ximax to nx, all y and z
        i0 = ximin * ny * nz
        i1 = i0 + (nx - ximax - 1) * ny * nz
        d_col[ximax+1:, :, :] = \
        d[l-1][i0:i1].reshape(
            (nx - ximax - 1, ny, nz))
        # Region 3
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * yimin * nz
        d_col[ximin:ximax+1, :yimin, :] = \
        d[l-1][i0:i1].reshape(
            (ximax + 1 - ximin, yimin, nz))
        # Region 4
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
        d_col[ximin:ximax+1, yimax+1:, :] = \
        d[l-1][i0:i1].reshape(
            (ximax + 1 - ximin, ny - yimax - 1, nz))
        # Region 5
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
        d_col[ximin:ximax+1, yimin:yimax+1, :zimin] = \
        d[l-1][i0:i1].reshape(
            (ximax + 1 - ximin, yimax + 1 - yimin, zimin))
        # Region 6
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
        d_col[ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
        d[l-1][i0:].reshape(
            (ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

        #print(l)
        #print(np.nonzero(np.isnan(d_col)))
        '''

    return d_col



@njit(parallel = True)
def array_substitute(a1, a2, ximin, ximax, yimin, yimax, zimin, zimax):
    for i in prange(ximax + 1 - ximin):
        for j in range(yimax + 1 - yimin):
            for k in range(zimax + 1 - zimin):
                #print(a1[ximin + i, yimin + j, zimin + k], a2[i, j, k])
                a1[ximin + i, yimin + j, zimin + k] = a2[i, j, k]
    return a1


@njit(parallel=True) # fastmath=True
def fast_binning_3d(data, nbin):
    nx, ny, nz = data.shape
    d_avg = np.zeros((nx // nbin, ny // nbin, nz // nbin), dtype = np.float64)

    for i in prange(nbin):
        d_avg += data[i::nbin, i::nbin, i::nbin]

    return d_avg / np.float64(nbin)
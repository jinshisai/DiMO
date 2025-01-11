# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from scipy.interpolate import griddata
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units
import dataclasses
from dataclasses import dataclass
import time

from .funcs import beam_convolution, gaussian2d, glnprof_conv
from .grid import Nested3DObsGrid, Nested2DGrid, Nested1DGrid, SubGrid2D
#from .linecube import tocube, solve_3LRT, waverage_to_cube, integrate_to_cube, solve_box3LRT
from .libcube.linecube import solve_MLRT, Tndv_to_cube, Tt_to_cube
from .molecule import Molecule
from .libcube import spectra, transfer, linecube
from .fast_grid import fast_3d_collapse


### constants
Ggrav  = constants.G.cgs.value        # Gravitational constant
Msun   = constants.M_sun.cgs.value    # Solar mass (g)
Lsun   = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
Rsun   = constants.R_sun.cgs.value    # Solar radius (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mH     = constants.m_p.cgs.value      # Proton mass (g)
hp     = constants.h.cgs.value # Planck constant [erg s]

# unit
auTOcm = units.au.to('cm') # 1 au (cm)

# Ignore divide-by-zero warning
np.seterr(divide='ignore')



class Builder(object):
    """docstring for Observer"""
    def __init__(self, x, y, z, v, model,
        xlim: list | None = None, ylim: list | None = None,
        nsub: list | None = None, zstrech: list | None = None, 
        reslim: float = 10, rin: float = 1.,
        adoptive_zaxis: bool = True, cosi_lim: float = 0.5, 
        beam: list | None = None, line: str | None = None, iline: int | None = None,
        Tmin: float = 1., Tmax: float = 2000., nTex: int = 4096,):
        '''
        Set up model grid and initialize model.

        Parameters
        ----------
        x, y, z (3D numpy ndarrays): Three dimensional coordinates aligned plane of sky (au).
        '''
        super(Builder, self).__init__()

        # make observer grid
        self.nx, self.ny, self.nz = len(x), len(y), len(z)
        self.grid = Nested3DObsGrid(
        x, y, z, xlim, ylim, nsub, zstrech, reslim, preserve_z = True) # Plane of sky coordinates
        self.grid2D = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)
        # Plane of sky coordinates
        self.xs = self.grid.xnest
        self.ys = self.grid.ynest
        self.zs = self.grid.znest
        # disk-local coordinates
        self.xps = None
        self.yps = None
        self.zps = None
        self.Rs = None # R in cylindarical coordinates
        self.phs = None # phi in cylindarical coordinates
        # dust layer
        self.Rmid = None
        self.rin = rin

        # velocity
        self.nv = len(v)
        self.delv = np.mean(v[1:] - v[:-1]) # km / s
        self.v = v
        self.ve = np.hstack([v - self.delv * 0.5, v[-1] + 0.5 * self.delv])

        # model
        self.model = model()

        # line
        self.line = line
        self.iline = iline
        if (line is not None) * (iline is not None):
            self.mol = Molecule(line)
            self.mol.moldata[line].partition_grid(Tmin, Tmax, nTex, scale = 'linear')
            self.mmol = self.mol.moldata[line].weight
            self.Qgrid = (self.mol.moldata[line]._Tgrid, self.mol.moldata[line]._PFgrid)
            self.trans, self.freq, self.Aul, self.gu, self.gl, self.Eu, self.El = \
            self.mol.moldata[line].params_trans(iline)

        # beam
        if beam is not None:
            self.define_beam(beam)
        else:
            self.beam = beam


    def define_beam(self, beam):
        '''
        Parameters
        ----------
         beam (list): Observational beam. Must be given in a format of 
                      [major (au), minor (au), pa (deg)].
        '''
        # save beam info
        self.beam = beam
        # define Gaussian beam
        nx, ny = self.grid2D.nx, self.grid2D.ny
        gaussbeam = gaussian2d(self.grid2D.xx.copy(), self.grid2D.yy.copy(), 1., 
            self.grid2D.xx[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
        self.grid2D.yy[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
        beam[1] / 2.35, beam[0] / 2.35, beam[2], peak=True)
        gaussbeam /= np.sum(gaussbeam)
        self.gaussbeam = gaussbeam


    def deproject_grid(self, 
        adoptive_zaxis = True, 
        cosi_lim = 0.5):
        '''
        Transfer the plane of sky coordinates to disk local coordinates.
        '''
        xp = self.xs
        yp = self.ys
        zp = self.zs
        # rotate by PA
        x, y = rot2d(xp - self.dx0, yp - self.dy0, self._pa_rad - 0.5 * np.pi)
        # rot = - (- (pa - 90.)); two minuses are for coordinate rotation and definition of pa
        # adoptive z axis
        if adoptive_zaxis & (np.abs(np.cos(self._inc_rad)) > cosi_lim):
            # center origin of z axis in the disk midplane
            zoffset = - np.tan(self._inc_rad) * y # zp_mid(xp, yp)
            self.zoffset = zoffset
            _zp = zp + zoffset # shift z center
            x, y, z = xrot(x, y, _zp, self._inc_rad) # rot = - (-inc)
        else:
            x, y, z = xrot(x, y, zp, self._inc_rad) # rot = - (-inc)
            self.zoffset = np.zeros(x.size)

        self.xps = x
        self.yps = y
        self.zps = z

        # cylindarical coordinates
        Rs = np.sqrt(x * x + y * y) # radius
        Rs[Rs < self.rin] = np.nan
        self.Rs = Rs # radius
        self.phs = np.arctan2(y, x) # azimuthal angle (rad)

        # for dust layer
        x, y = rot2d(self.grid2D.xnest - self.dx0, 
            self.grid2D.ynest - self.dy0, self._pa_rad - 0.5 * np.pi) # in 2D
        y /= np.cos(self._inc_rad)
        self.Rmid = np.sqrt(x * x + y * y) # radius
        self.adoptive_zaxis = adoptive_zaxis


    def set_model(self, params):
        self.model.set_params(**params)
        # geometric parameters
        self.dx0 = params['dx0']
        self.dy0 = params['dy0']
        self.inc = params['inc']
        self._inc_rad = np.radians(self.inc)
        self.pa = params['pa']
        self._pa_rad = np.radians(self.pa)


    def build_model(self, dv_mode):
        T_g, n_g, vlos, dv, T_d, tau_d = self.model.build(
            self.Rs, self.phs, self.zps, self.Rmid,
            dv_mode = dv_mode, mmol = self.mmol)
        return T_g, n_g, vlos, dv, T_d, tau_d


    def build_cube(self, 
        Tcmb = 2.73, f0 = 230., 
        dist = 140., dv_mode = 'total', 
        contsub = True, return_Ttau = False):
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode = dv_mode)

        # dust
        #T_d = self.grid2D.collapse(T_d)
        #tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature 
        #  of each gas layer at every velocity channel.
        if (self.model.dv > 0.) | (dv_mode == 'thermal'):
            # line profile function
            lnprofs = spectra.glnprof_series(self.v, vlos.ravel(), dv.ravel())

            # get nv
            #start = time.time()
            nv_cube = linecube.to_xyzv(
                np.array([n_g.ravel()]), lnprofs)
            nv_g = nv_cube[0].reshape((self.nv, self.grid.nxy, self.nz))

            # to cube
            Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = transfer.Tnv_to_cube(
                T_g, nv_g, self.grid.znest,
                self.grid.dznest * auTOcm,
                self.freq, self.Aul, self.Eu, self.gu, self.Qgrid)
            #Tv_gf = self.grid.collapse2D(Tv_gf)
            #Tv_gr = self.grid.collapse2D(Tv_gr)
            #tau_v_gf = self.grid.collapse2D(tau_v_gf)
            #tau_v_gr = self.grid.collapse2D(tau_v_gr)

            # v, x, y to v, y, x
            #Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = np.transpose(
            #    np.array([Tv_gf, Tv_gr, tau_v_gf, tau_v_gr]), axes = (0,1,3,2))
        else:
            Tv_gf, Tv_gr, Nv_gf, Nv_gr = np.transpose(
            Tt_to_cube(T_g, n_gf, n_gr, vlos, self.ve, self.grid.dz * auTOcm,),
            (0,1,3,2,))

        Tv_gf = Tv_gf.clip(1., None) # safety net to avoid zero division
        Tv_gr = Tv_gr.clip(1., None)


        # density to tau
        #print('Tv_gf max, q: %13.2e, %.2f'%(np.nanmax(Tv_gf), self.qg))
        #print('Nv_gf max: %13.2e'%(np.nanmax(Nv_gf)))
        #if (self.line is not None) * (self.iline is not None):
        #    tau_v_gf = self.mol.get_tau(self.line, self.iline, 
        #        Nv_gf, Tv_gf, delv = None, grid_approx = True)
        #    tau_v_gr = self.mol.get_tau(self.line, self.iline, 
        #        Nv_gr, Tv_gr, delv = None, grid_approx = True)
        #else:
        #    # ignore temperature effect on conversion from column density to tau
        #    tau_v_gf = Nv_gf
        #    tau_v_gr = Nv_gr
        #print('tau_v_gf max: %13.2e'%(np.nanmax(tau_v_gf)))


        if return_Ttau:
            return np.array([Tv_gf, tau_v_gf, Tv_gr, tau_v_gr])

        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, self.grid.dx, self.grid.dy, 
            dist = dist, au = True)
        #_Bv = lambda T, v: Bv(T, v)
        _Bv_cmb = _Bv(Tcmb, f0)
        _Bv_gf  = _Bv(Tv_gf, f0)
        _Bv_gr  = _Bv(Tv_gr, f0)
        _Bv_d   = _Bv(T_d, f0)
        Iv = solve_MLRT(_Bv_gf, _Bv_gr, _Bv_d, 
            tau_v_gf, tau_v_gr, tau_d, _Bv_cmb, self.nv)

        # contsub
        if contsub == False:
            Iv_d = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_d))
            Iv_d = np.tile(Iv_d, (self.nv,1,))
            Iv += Iv_d # add continuum back


        # collapse to cube
        Iv = np.transpose(
            self.grid.collapse2D(Iv),
            axes = (0,2,1)) # (v, x, y) to (v, y, x)

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(self.grid2D.xx.copy(), self.grid2D.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        return Iv


    def show_model_sideview(self, 
        dv_mode='total', cmap = 'viridis', 
        savefig = False, showfig = True, 
        outname = 'model_sideview', vmax = 0.90, vmin = 0.):
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode)
        #n_g = self.grid.collapse(n_g)

        vmax *= np.nanmax(n_g)

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        fig, axes = plt.subplots(1,2)
        ax1, ax2 = axes


        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for l in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[l,:]
            xmin, xmax = _grid.xlim[l]
            zmin, zmax = _grid.zlim[l]

            d_plt = _grid.collapse(n_g, upto = l)

            # hide parental layer
            if l <= _grid.nlevels-2:
                ximin, ximax = _grid.xinest[(l+1)*2:(l+2)*2]
                yimin, yimax = _grid.yinest[(l+1)*2:(l+2)*2]
                d_plt[ximin:ximax+1,yimin:yimax+1] = np.nan

            #if self.adoptive_zaxis:
            #    if _grid.preserve_z:
            #        zoff = self.zoffset[_grid.xypartition[l]:_grid.xypartition[l+1],:]
            #    else:
            #        zoff = self.zoffset[_grid.partition[l]:_grid.partition[l+1]]

            #ax1.imshow(d_plt, extent = (zmin + zoff, zmax + zoff, xmin, xmax),
            #    alpha = 1., vmax = vmax, vmin = vmin, origin = 'upper', cmap = cmap)
            #print(l, zoff.shape)
            _xx, _yy, _zz = _grid.get_grid(l)

            if self.adoptive_zaxis:
                zoff = _grid.collapse(self.zoffset, upto = l)
                _zz += zoff

            ax1.pcolormesh(_zz[:, ny//2, :], _xx[:, ny//2, :], d_plt[:, ny//2, :], 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            ax2.pcolormesh(_zz[nx//2, :, :], _yy[nx//2, :, :], d_plt[nx//2, :, :], 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            #rect = plt.Rectangle((zmin, xmin), 
            #    zmax - zmin, xmax - xmin, edgecolor = 'white', facecolor = "none",
            #    linewidth = 0.5, ls = '--')
            #ax1.add_patch(rect)

        #ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        ax1.set_ylim(_grid.xlim[0])
        ax2.set_ylim(_grid.ylim[0])

        ax1.set_ylabel(r'$x$ (au)')
        ax1.set_xlabel(r'$z$ (au)')
        ax2.set_ylabel(r'$y$ (au)')

        fig.tight_layout()

        '''
        _grid = copy.deepcopy(self.grid)
        if self.adoptive_zaxis:
            _grid.znest += self.zoffset
        _grid.visualize_xz(n_g, 
            ax = ax1, vmax = np.nanmax(n_g) * 0.01, cmap = cmap)
        '''

        if savefig: fig.savefig(outname + '.png', dpi = 300, transparent = True)
        if showfig: plt.show()
        plt.close()


def rot2d(x, y, ang):
    return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)

def xrot(x, y, z, ang):
    return x, y * np.cos(ang) - z * np.sin(ang), y * np.sin(ang) + z * np.cos(ang)

def yp2y(yp, z, inc):
    """
    Deproject y' of the plane of the sky (PoS) coordindates to y of the disk coordinates.
    The height on the disk coordindate, z, must be given.
     yp = y cos(i) + z sin(i)
    therefore,
     y = (yp - z sin(i)) / cos(i)

    Parameters
    ----------
     yp (float or array): y' of PoS coordinates (any unit of distance.
     z (float or array): z of the disk local coordinates (any unit of distance).
     inc (float): Inclination angle of the disk (rad).
    """
    return (yp - z * np.sin(inc)) / np.cos(inc)


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    #print((hp*v)/(kb*T))
    exp=np.exp((hp*v)/(kb*T)) - 1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    #print(exp, T, v)
    return Bv



# Planck function
def Bvppx(T, v, px, py, dist = 140., au = True):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    # unit
    v = v * 1.e9 # GHz --> Hz

    # Bv in cgs
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    fterm = (2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # From cgs to Jy/str
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(px / dist / 3600.) # au --> radian
        py = np.radians(py / dist / 3600.) # au --> radian
    else:
        px = np.radians(px) # deg --> rad
        py = np.radians(py) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(px*py)
    Bv *= one_pixel_area # Iv (Jy per pixel)
    return Bv



# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    bmaj, bmin: beamsize [arcsec]
    '''

    # units
    bmaj = np.radians(bmaj / 3600.) # arcsec --> radian
    bmin = np.radians(bmin / 3600.) # arcsec --> radian
    v = v * 1.e9 # GHz --> Hz

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj * bmin * C2  # beam --> str


    #print(np.nanmax((hp*v)/(kb*T)), np.nanmin(T))
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    #print(np.nanmax(exp))
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv
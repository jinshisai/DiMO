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
from scipy import special

from .funcs import beam_convolution, gaussian2d, glnprof_conv
from .grid import Nested3DObsGrid, Nested2DGrid, Nested1DGrid, SubGrid2D
#from .linecube import tocube, solve_3LRT, waverage_to_cube, integrate_to_cube, solve_box3LRT
from .libcube.linecube import solve_MLRT, Tndv_to_cube, Tt_to_cube, solve_MLRT_cube
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
            self.f = doppler_v2f(v *1.e5, self.freq)

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


    def getQrot(self, Ts):
        shape = Ts.shape
        Qrot = np.interp(Ts.ravel(), self.Qgrid[0], self.Qgrid[1])
        return Qrot.reshape(shape)


    def deproject_grid(self, 
        adoptive_zaxis = True, 
        cosi_lim = 0.5):
        '''
        Transfer the plane of sky coordinates to disk local coordinates.
        '''
        xp = self.xs
        yp = self.ys
        zp = self.zs
        dzp = self.grid.dznest
        # rotate by PA
        #_xp, _yp = xp, yp
        _xp, _yp = rot2d(xp - self.dx0, yp - self.dy0, self._pa_rad - 0.5 * np.pi)
        # rot = - (- (pa - 90.)); two minuses are for coordinate rotation and definition of pa
        # adoptive z axis
        if adoptive_zaxis & (np.abs(np.cos(self._inc_rad)) > cosi_lim):
            # consider z=0 is in the disk midplane
            zoffset = - np.tan(self._inc_rad) * _yp # zp_mid(xp, yp)
            #zoffset = dzp * (zoffset // dzp) # shift in steps of dz
            self.zoffset = zoffset
            _zp = zp + zoffset # shift z center back to rectanglar coordinates
            x, y, z = xrot(_xp, _yp, _zp, self._inc_rad) # rot = - (-inc)
            #y /= np.cos(self._inc_rad)
            #x = _xp
            #y = _yp * np.cos(self._inc_rad) - _zp * np.sin(self._inc_rad)
            #z = _yp * np.sin(self._inc_rad) + _zp * np.cos(self._inc_rad)
        else:
            x, y, z = xrot(_xp, _yp, zp, self._inc_rad) # rot = - (-inc)
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
        x, y = rot2d(self.grid.xnest[:,0] - self.dx0, 
            self.grid.ynest[:,0] - self.dy0, self._pa_rad - 0.5 * np.pi) # in 2D
        y /= np.cos(self._inc_rad)
        self.Rmid = np.sqrt(x * x + y * y) # radius
        self.adoptive_zaxis = adoptive_zaxis


    def set_model(self, params):
        self.model.set_params(**params)
        # geometric parameters
        self.dx0 = params['dx0']
        self.dy0 = params['dy0']
        self.inc = params['inc']
        _inc_rad = np.radians(self.inc)
        self._inc_rad = _inc_rad
        self.pa = params['pa']
        self._pa_rad = np.radians(self.pa)
        self.side = np.sign(np.cos(_inc_rad)) # cos(-i) = cos(i)


    def build_model(self, dv_mode, pterm):
        #T_g, n_g, vlos, dv, T_d, tau_d = self.model.build(
        #   self.Rs, self.phs, self.zps, self.Rmid,
        #    dv_mode = dv_mode, mmol = self.mmol, pterm = pterm)
        T_g, n_g, vlos, dv, T_d, tau_d = self.model.fastbuild(
            self.Rs, self.phs, self.zps, self.Rmid,
            dv_mode = dv_mode, mmol = self.mmol, pterm = pterm)
        return T_g, n_g, vlos, dv, T_d, tau_d


    def build_cube(self, 
        Tcmb = 2.73, f0 = 230., 
        dist = 140., dv_mode = 'total', 
        pterm = True, contsub = True, return_Ttau = False):
        #start = time.time()
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode = dv_mode, pterm = pterm)
        #end = time.time()
        #print('building model took %.2fs'%(end-start))


        # dust
        #T_d = self.grid2D.collapse(T_d)
        #tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature 
        #  of each gas layer at every velocity channel.
        if (self.model.dv > 0.) | (dv_mode == 'thermal'):
            # line profile function
            #start = time.time()
            lnprofs = spectra.glnprof_series(self.v, 
                vlos.ravel(), dv.ravel(), unit_scale = 1.e-5)
            #end = time.time()
            #print('making spectra took %.2fs'%(end-start))

            # get nv
            #start = time.time()
            nv_g = spectra.to_xyzv(n_g.ravel(), lnprofs)
            #nv_g = n_g.ravel()[:, np.newaxis] * lnprofs
            nv_g = nv_g.reshape((self.grid.nxy, self.nz, self.nv))
            #end = time.time()
            #print('to xyzv took %.2fs'%(end-start))


            #start = time.time()
            Qrots = self.getQrot(T_g)
            #end = time.time()
            #print('get Qrot took %.2fs'%(end-start))

            # to cube
            #start = time.time()
            if self.side == 1:
                Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = transfer.Tnv_to_cube(
                    T_g, nv_g, self.grid.znest,
                    self.grid.dznest * auTOcm,
                    self.freq, self.Aul, self.Eu, self.gu, Qrots)
            else:
                Tv_gr, Tv_gf, tau_v_gr, tau_v_gf = transfer.Tnv_to_cube(
                    T_g, nv_g, self.grid.znest,
                    self.grid.dznest * auTOcm,
                    self.freq, self.Aul, self.Eu, self.gu, Qrots)
            #end = time.time()
            #print('to cube took %.2fs'%(end-start))
        else:
            Tv_gf, Tv_gr, Nv_gf, Nv_gr = np.transpose(
            Tt_to_cube(T_g, n_gf, n_gr, vlos, self.ve, self.grid.dz * auTOcm,),
            (0,1,3,2,))

        Tv_gf = Tv_gf.clip(1., None) # safety net to avoid zero division
        Tv_gr = Tv_gr.clip(1., None)


        if return_Ttau:
            Tv_gf = self.grid.collapse2D(Tv_gf)
            Tv_gr = self.grid.collapse2D(Tv_gr)
            tau_v_gf = self.grid.collapse2D(tau_v_gf)
            tau_v_gr = self.grid.collapse2D(tau_v_gr)
            return np.array([Tv_gf, tau_v_gf, Tv_gr, tau_v_gr])

        # radiative transfer
        #start = time.time()
        _Bv = lambda T, v: Bvppx(T, v, self.grid.dx, self.grid.dy, 
            dist = dist, au = True) # in unit of Jy/pixel with the final pixel size
        #_Bv = lambda T, v: Bv(T, v)
        f = doppler_v2f((self.v - self.model.vsys) *1.e5, self.freq)
        fs = self.f[np.newaxis,:]
        _Bv_cmb = _Bv(Tcmb, fs)
        _Bv_gf  = _Bv(Tv_gf, fs)
        _Bv_gr  = _Bv(Tv_gr, fs)
        _Bv_d   = _Bv(T_d[:,np.newaxis], fs)#[:,np.newaxis]
        #_Bv_d   = _Bv(T_d, self.freq)[:,np.newaxis]
        tau_d   = tau_d[:,np.newaxis]
        #Iv = solve_MLRT(_Bv_gf, _Bv_gr, _Bv_d, 
        #    tau_v_gf, tau_v_gr, tau_d, _Bv_cmb, self.nv)
        Iv = solveRT_TL(_Bv_gf, _Bv_gr, _Bv_d, _Bv_cmb,
            tau_v_gf, tau_v_gr, tau_d,)
        #end = time.time()
        #print('radiative transfer took %.2fs'%(end-start))


        # contsub
        if contsub == False:
            Iv_d = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_d))
            Iv_d = np.tile(Iv_d, (self.nv,1,))
            Iv += Iv_d # add continuum back

        Iv = np.transpose(
            self.grid.collapse2D(Iv.T, collapse_mode = 'mean'),
            axes = (0,2,1)) # (v, x, y, v) to (v, y, x)

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(self.grid2D.xx.copy(), self.grid2D.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        return Iv


    def show_model_sideview(self, 
        dv_mode='total', pterm = True, cmap = 'viridis', 
        savefig = False, showfig = True, 
        outname = 'model_sideview', vmax = 1.0, vmin = 1.e-5):
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(
            dv_mode=dv_mode, pterm = pterm)

        #n_g = np.log10(n_g) # in log scale
        #n_g = self.Rs
        #n_g = self.xps
        #vmin, vmax = np.nanmin(n_g), np.nanmax(n_g)
        vmax *= np.log10(np.nanmax(n_g))
        vmin *= np.log10(np.nanmax(n_g))

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
            d_plt = np.log10(d_plt) # in log scale

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
                zoffset = _grid.collapse(self.zoffset, upto = l)
                _zz += zoffset
                #_xx, _yy = rot2d(_xx - self.dx0, _yy - self.dy0, 
                #    self._pa_rad - 0.5 * np.pi)

            im1 = ax1.pcolormesh(_zz[:, ny//2, :], _xx[:, ny//2, :], d_plt[:, ny//2, :], 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            im2 = ax2.pcolormesh(_zz[nx//2, :, :], _yy[nx//2, :, :], d_plt[nx//2, :, :], 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            #rect = plt.Rectangle((zmin, xmin), 
            #    zmax - zmin, xmax - xmin, edgecolor = 'white', facecolor = "none",
            #    linewidth = 0.5, ls = '--')
            #ax1.add_patch(rect)

        #ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        gmin = np.min([np.min(_grid.xlim[0]), np.min(_grid.ylim[0])])
        gmax = np.max([np.max(_grid.xlim[0]), np.max(_grid.ylim[0])])
        for ax in (axes):
            ax.set_xlim(gmin, gmax)
            ax.set_ylim(gmin, gmax)
            ax.set_aspect(1)

        ax1.set_ylabel(r'$x^\prime$ (au)')
        ax1.set_xlabel(r'$z^\prime$ (au)')
        ax2.set_ylabel(r'$y^\prime$ (au)')
        ax2.set_xlabel(r'$z^\prime$ (au)')

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


    def show_model_faceview(self, 
        dv_mode='total', cmap = 'viridis', 
        savefig = False, showfig = True, 
        outname = 'model_sideview', vmax = 0.90, vmin = 0.):
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode)
        #n_g = self.grid.collapse(n_g)

        vmax *= np.nanmax(n_g)

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        fig, ax1 = plt.subplots(1,1)
        #ax1 = axes


        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for l in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[l,:]
            xmin, xmax = _grid.xlim[l]
            ymin, ymax = _grid.ylim[l]

            d_plt = _grid.collapse(n_g, upto = l)

            # hide parental layer
            if l <= _grid.nlevels-2:
                ximin, ximax = _grid.xinest[(l+1)*2:(l+2)*2]
                yimin, yimax = _grid.yinest[(l+1)*2:(l+2)*2]
                d_plt[ximin:ximax+1,yimin:yimax+1] = np.nan

            _xx, _yy, _zz = _grid.get_grid(l)

            if self.adoptive_zaxis:
                zoff = _grid.collapse(self.zoffset, upto = l)
                _zz += zoff

            ax1.pcolormesh(_xx[:, :, nz//2], _yy[:, :, nz//2], d_plt[:, :, nz//2], 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)
            rect = plt.Rectangle((xmin, ymin), 
                xmax - xmin, ymax - ymin, edgecolor = 'white', facecolor = "none",
                linewidth = 0.5, ls = '--')
            ax1.add_patch(rect)

        #ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        ax1.set_xlim(_grid.xlim[0])
        ax1.set_ylim(_grid.ylim[0])

        ax1.set_xlabel(r'$x$ (au)')
        ax1.set_ylabel(r'$y$ (au)')

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



    def show_model_4Dview(self, 
        dv_mode='total', cmap = 'viridis', 
        savefig = False, showfig = True, 
        outname = 'model_sideview', vmax = 0.90, vmin = 0.,
        nsparse = 10):
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode)
        #n_g = self.grid.collapse(n_g)

        vmax *= np.nanmax(n_g)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')


        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for l in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[l,:]
            xmin, xmax = _grid.xlim[l]
            ymin, ymax = _grid.ylim[l]

            d_plt = _grid.collapse(n_g, upto = l)

            # hide parental layer
            if l <= _grid.nlevels-2:
                ximin, ximax = _grid.xinest[(l+1)*2:(l+2)*2]
                yimin, yimax = _grid.yinest[(l+1)*2:(l+2)*2]
                d_plt[ximin:ximax+1,yimin:yimax+1,:] = np.nan

            _xx, _yy, _zz = _grid.get_grid(l)

            if self.adoptive_zaxis:
                zoff = _grid.collapse(self.zoffset, upto = l)
                _zz += zoff

            x_sparse = nx//nsparse
            y_sparse = ny//nsparse
            z_sparse = nz//nsparse
            d_plt = d_plt[::x_sparse, ::y_sparse, ::z_sparse].ravel()
            _xx_plt = _xx[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            _yy_plt = _yy[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            _zz_plt = _zz[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            d_plt = d_plt[~np.isnan(d_plt)]
            ax.scatter(_xx_plt, _yy_plt, _zz_plt, c = d_plt, 
                alpha = 1., vmax = vmax, vmin = vmin, cmap = cmap)

        #ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        #ax1.set_xlim(_grid.xlim[0])
        #ax1.set_ylim(_grid.ylim[0])

        ax.set_xlabel(r'$x$ (au)')
        ax.set_ylabel(r'$y$ (au)')
        ax.set_zlabel(r'$z$ (au)')

        fig.tight_layout()

        if savefig: fig.savefig(outname + '.png', dpi = 300, transparent = True)
        if showfig: plt.show()
        plt.close()


class Builder_SSDisk(object):
    '''
    A disk model with Two Thick Layers (TTL) with a thin dust layer.

    '''

    def __init__(self, model,
        axes_model: list, axes_sky: list, 
        xlim: list | None = None, ylim: list | None = None,
        nsub: list | None = None, reslim: float = 10,
        beam: list | None = None,
        coordinate_type = 'polar'):
        '''
        Set up model grid and initialize model.

        Parameters
        ----------
        axes (list of axes): Three dimensional coordinates aligned plane of sky (au).
        '''
        super(Builder_SSDisk, self).__init__()

        # model
        self.model = model()
        self._model = model

        # model grid
        if coordinate_type == 'polar':
            self.build_polar_grid(axes_model)

        # sky grid
        self.reslim = reslim
        self.nsub = nsub
        x, y, v  = axes_sky
        self.v = v
        self.nv = len(v)
        self.nx = len(x)
        self.ny = len(y)
        self.build_sky_grid(x, y, xlim, ylim, nsub, reslim)

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
        nx, ny = self.skygrid.nx, self.skygrid.ny
        xx = self.skygrid.xx.copy()
        yy = self.skygrid.yy.copy()
        gaussbeam = gaussian2d(xx, yy, 1., 
            xx[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            yy[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            beam[1] / 2.35, beam[0] / 2.35, beam[2], peak=True)
        gaussbeam /= np.sum(gaussbeam)
        self.gaussbeam = gaussbeam


    def build_sky_grid(self, x, y, xlim = None, ylim = None, nsub = None, reslim = 10):
        self.skygrid = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)


    def build_polar_grid(self, axes):
        r, phi = axes
        self.r = r
        self.phi = phi
        self.nr = len(r)
        self.nphi = len(phi)

        rr, phph = np.meshgrid(r, phi, indexing = 'ij')
        self.rs = rr.ravel()
        self.phis = phph.ravel()


    def set_model(self, params):
        # geometric parameters
        self.dx0 = params['dx0']
        self.dy0 = params['dy0']
        self.inc = params['inc']
        self.__inc_rad = np.radians(self.inc)
        self.pa = params['pa']
        self.__pa_rad = np.radians(self.pa)

        # set model parameters
        #_p = dict(params)
        #del _p['dx0']
        #del _p['dy0']
        self.model.set_params(**params)


    def skygrid_info(self):
        self.skygrid.gridinfo()


    def build_model(self, build_args = None):
        if build_args is not None:
            I_int, vlos, dv = self.model.build(self.rs, self.phis, *build_args)
        else:
            I_int, vlos, dv = self.model.build(self.rs, self.phis,)
        return I_int, vlos, dv


    def project_grid(self):
        # Convert spherical to Cartesian coordinates
        x = self.rs * np.cos(self.phis)
        y = self.rs * np.sin(self.phis)

        # Apply inclination and position angle
        y_rot = y * np.cos(self.__inc_rad)
        rot_ang = self.__pa_rad + 0.5 * np.pi
        x_rot = x * np.cos(rot_ang) + y_rot * np.sin(rot_ang)
        y_rot = -x * np.sin(rot_ang) + y_rot * np.cos(rot_ang)

        self.xproj = x_rot - self.dx0
        self.yproj = y_rot - self.dy0


    def project_quantity(self, q):
        # interpolator
        q_proj = griddata(
            (self.xproj, self.yproj), q, (self.skygrid.xnest, self.skygrid.ynest),
            method = 'linear', fill_value = 0.)
        return q_proj



    def build_cube(self, 
        build_args =  None,
        I_pre = None, vlos_pre = None, dv_pre = None):
        # model build
        _I_int, _vlos, _dv = self.build_model(build_args)
        # if any pre-calculations are provided or not
        I_int = _I_int if I_pre is None else I_pre
        vlos = _vlos if vlos_pre is None else vlos_pre
        dv = _dv if dv_pre is None else dv_pre
        #print(I_int.shape, dv.shape)

        # calculate sky projection
        self.project_grid()

        ''' new version; line profile first
        lnprofs = spectra.glnprof_series(self.v, vlos, dv) # x,y,v
        lnprofs[np.isnan(lnprofs)] = 0.
        _Iv = np.tile(I_int, (self.nv, 1,)) * lnprofs

        Iv = np.array([
            self.project_quantity(_Iv[i,:]) for i in range(self.nv)
            ])
        '''

        #'''old version; projection first
        I_proj = self.project_quantity(I_int)
        v_proj = self.project_quantity(vlos)
        dv_proj = self.project_quantity(dv)

        # line profile function
        lnprofs = spectra.glnprof_series(self.v, v_proj, dv_proj) # x,y,v
        Iv = np.tile(I_proj, (self.nv, 1,)) * lnprofs
        #'''

        # collapse
        Iv = self.skygrid.high_dimensional_collapse(Iv, 
            fill = 'zero',
            collapse_mode = 'mean')

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.skygrid.xx.copy(), 
                self.skygrid.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        return Iv


    def visualize_grid(self, 
        keys = ['intensity', 'vlos', 'dv'], savefig = False,
        showfig = True, outname = None, cmap = 'viridis',
        build_args = None):
        # visualize grid
        I_int, vlos, dv = self.build_model(build_args)
        self.project_grid()

        qs = []
        for q, l in zip([I_int, vlos, dv], 
            ['intensity', 'vlos', 'dv']):
            if l in keys:
                _q = self.project_quantity(q)
                if l == 'intensity':
                    _q = np.log10(_q)
                    _q[_q < 0.] = np.nan
                qs.append(_q)

        for q, l in zip(qs, keys):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            _outname = outname + '_%s.png'%l if outname is not None else None
            self.skygrid.visualize_grid(q, ax = ax,
                showfig = False, savefig = False,
                outname = _outname, cmap = cmap)
            ax.text(0.1, 0.9, l, transform = ax.transAxes, ha = 'left', va = 'top')
            if savefig: fig.savefig(outname + '_%s.png'%l, dpi = 300)
            if showfig: plt.show()
            plt.close()


class Builder_SLD(Builder_SSDisk):
    '''
    Builder for Two-Layer Disk Model.

    '''

    def __init__(self, model,
        axes_model: list, axes_sky: list, 
        xlim: list | None = None, ylim: list | None = None,
        nsub: list | None = None, reslim: float = 10,
        beam: list | None = None,
        coordinate_type: str = 'polar',
        line: str | None = None, iline: int | None = None, 
        dust_opacity: float = None,
        Tmin: float = 1., Tmax: float = 2000., nTex: int = 4096,):
        '''
        Set up model grid and initialize model.

        Parameters
        ----------
        axes (list of axes): Three dimensional coordinates aligned plane of sky (au).
        '''
        super().__init__(model, axes_model = axes_model,
            axes_sky = axes_sky, xlim = xlim, ylim = ylim,
            nsub = nsub, reslim = reslim, beam = beam,
            coordinate_type = coordinate_type,)
        #super().__init__()

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

        # dust
        self.kappa = dust_opacity


    def build_model(self, build_args = None):
        if build_args is not None:
            T_g, N_g, vlos, dv, T_d, Sig_d = self.model.build(self.rs, self.phis, *build_args)
        else:
            T_g, N_g, vlos, dv, T_d, Sig_d = self.model.build(self.rs, self.phis,)
        return T_g, N_g, vlos, dv, T_d, Sig_d


    def build_cube(self,
        build_args =  None,
        Tcmb = 2.73, dist = 140., f0 = None, contsub = True,
        Ng_pre = None, Tg_pre = None, vlos_pre = None, 
        Td_pre = None, Sigd_pre = None, dv_pre = None,
        return_tau = False):
        # model build
        _T_g, _N_g, _vlos, _dv, _T_d, _Sig_d = self.build_model(build_args)
        # if any pre-calculations are provided or not
        Ng = _N_g if Ng_pre is None else Ng_pre
        Tg = _T_g if Tg_pre is None else Tg_pre
        vlos = _vlos if vlos_pre is None else vlos_pre
        Td = _T_d if Td_pre is None else Tg_pre
        Sigd = _Sig_d if Sigd_pre is None else Sigd_pre
        dv = _dv if dv_pre is None else dv_pre

        # calculate sky projection
        self.project_grid()

        # projection
        Ng_proj = self.project_quantity(Ng)
        Tg_proj = self.project_quantity(Tg)
        Sigd_proj = self.project_quantity(Sigd)
        Td_proj = self.project_quantity(Td)
        v_proj = self.project_quantity(vlos)
        dv_proj = self.project_quantity(dv)
        #print(v_proj, dv_proj)

        # line profile function
        lnprofs = spectra.glnprof_series(self.v, v_proj, dv_proj, unit_scale = 1.e-5) # x,y,v
        Ng_v = np.tile(Ng_proj, (self.nv, 1,)) * lnprofs
        Tg_v = np.tile(Tg_proj, (self.nv, 1,)) #* lnprofs
        Sigd_v = np.tile(Sigd_proj, (self.nv, 1,)) #* lnprofs
        Td_v = np.tile(Td_proj, (self.nv, 1,)) #* lnprofs

        # Nv to tau_v
        tau_v_g = transfer.Nv_to_tauv(
                    Tg_v, Ng_v, 
                    self.freq, self.Aul, self.Eu, self.gu, self.Qgrid)
        Tg_v = Tg_v.clip(1., None)
        tau_v_d = Sigd_v * self.kappa

        if return_tau:
            # collapse
            Iv = self.skygrid.high_dimensional_collapse(tau_v_g, 
                fill = 'zero',
                collapse_mode = 'mean')
            return Iv

        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, 
            self.skygrid.dx, self.skygrid.dy, 
            dist = dist, au = True) # in unit of Jy/pixel with the final pixel size
        if f0 is None:
            f0 = self.freq * 1.e-9
        _Bv_cmb = _Bv(Tcmb, f0)
        _Bv_g = _Bv(Tg_v, f0)
        _Bv_d = _Bv(Td_v, f0)
        #Iv = solve_MLRT(_Bv_gf, _Bv_gr, _Bv_d, 
        #    tau_v_gf, tau_v_gr, tau_d, _Bv_cmb, self.nv)
        Iv = _Bv_cmb * ( - 1. + np.exp(- tau_v_g - tau_v_d)) \
        + _Bv_d * (1. - np.exp(-tau_v_d)) * np.exp(- tau_v_g) \
        + _Bv_g * (1. - np.exp(-tau_v_g))

        # contsub
        #if contsub:
        #    Iv_d = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_v_d))
        #    Iv -= Iv_d # add continuum back

        # collapse
        Iv = self.skygrid.high_dimensional_collapse(Iv, 
            fill = 'zero',
            collapse_mode = 'mean')

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.skygrid.xx.copy(), 
                self.skygrid.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        return Iv


    def visualize_grid(self, 
        keys = ['intensity', 'vlos', 'dv'], savefig = False,
        showfig = True, outname = None, cmap = 'viridis',
        build_args = None):
        # visualize grid
        I_int, vlos, dv = self.build_model(build_args)
        self.project_grid()

        qs = []
        for q, l in zip([I_int, vlos, dv], 
            ['intensity', 'vlos', 'dv']):
            if l in keys:
                _q = self.project_quantity(q)
                if l == 'intensity':
                    _q = np.log10(_q)
                    _q[_q < 0.] = np.nan
                qs.append(_q)

        for q, l in zip(qs, keys):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            _outname = outname + '_%s.png'%l if outname is not None else None
            self.skygrid.visualize_grid(q, ax = ax,
                showfig = False, savefig = False,
                outname = _outname, cmap = cmap)
            ax.text(0.1, 0.9, l, transform = ax.transAxes, ha = 'left', va = 'top')
            if savefig: fig.savefig(outname + '_%s.png'%l, dpi = 300)
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
    v: frequency [Hz]
    '''
    #v = v * 1.e9 # GHz --> Hz
    #print((hp*v)/(kb*T))
    exp=np.exp((hp*v)/(kb*T)) - 1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    return fterm/exp


# Planck function
def Bvppx(T, v, px, py, dist = 140., au = True):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [Hz]
    '''
    # unit
    #v = v * 1.e9 # GHz --> Hz

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


def cgs_to_Jyppx(dx, dy, dist = 140., au = True):
    # From cgs to Jy/str
    f = 1.e-7 * 1.e4 # cgs --> MKS
    f *= 1.0e26 # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(dx / dist / 3600.) # au --> radian
        py = np.radians(dy / dist / 3600.) # au --> radian
    else:
        px = np.radians(dx) # deg --> rad
        py = np.radians(dy) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(dx*dy)
    f *= one_pixel_area # Iv (Jy per pixel)
    return f



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


def solveRT_TL(Sv_gf, Sv_gr, Sv_d, Sv_bg,
    tau_v_gf, tau_v_gr, tau_d):
    Iv_d = (Sv_d - Sv_bg) * (1. - np.exp(- tau_d))
    Iv = Sv_bg * (
        np.exp(- tau_v_gf - tau_d - tau_v_gr) - 1.) \
            + Sv_gr * (1. - np.exp(- tau_v_gr)) \
            * np.exp(- tau_v_gf  - tau_d) \
            + Sv_d * (1. - np.exp(- tau_d)) * np.exp(- tau_v_gf) \
            + Sv_gf * (1. - np.exp(- tau_v_gf)) \
            - Iv_d
    return Iv


def doppler_f2v(f, f0, definition = 'radio'):
    return (f0 - f) / f0 * clight

def doppler_df2dv(df, f0, definition = 'radio'):
    return - df * clight / f0

def doppler_v2f(v, f0, definition = 'radio'):
    return f0 - v * f0 / clight

def doppler_dv2df(dv, f0, definition = 'radio'):
    return - dv * f0 / clight
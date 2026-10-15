# import modules
import copy
from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from numpy.typing import ArrayLike
from scipy.interpolate import griddata
from scipy.signal import convolve

from .funcs import beam_convolution, gaussian2d, gauss1d
from .grid import Nested3DObsGrid, Nested2DGrid, SubGrid1D
from .libcube import spectra, transfer
from .plt_utils import *
from .molecule import Molecule


# constants
Ggrav = constants.G.cgs.value  # Gravitational constant
Msun = constants.M_sun.cgs.value  # Solar mass (g)
Lsun = constants.L_sun.cgs.value  # Solar luminosity (erg s^-1)
Rsun = constants.R_sun.cgs.value  # Solar radius (cm)
clight = constants.c.cgs.value  # light speed (cm s^-1)
kb = constants.k_B.cgs.value  # Boltzman coefficient
sigsb = constants.sigma_sb.cgs.value  # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mH = constants.m_p.cgs.value  # Proton mass (g)
hp = constants.h.cgs.value  # Planck constant [erg s]

# unit
auTOcm = units.au.to("cm")  # 1 au (cm)

# Ignore divide-by-zero warning
np.seterr(divide="ignore")


# Builders
class Builder(object):
    """
    Standard 3D model builder.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate axes of the observer grid in au.
    v : np.ndarray
        Velocity axis in km/s.
    model : Callable
        Model class or factory used to instantiate the disk model.
    xlim, ylim : list, optional
        Nested-grid limits in au.
    nsub : list, optional
        Subdivision factors for nested-grid levels.
    zstrech : list, optional
        Stretch factors for the z-axis.
    reslim : float
        Default nested-grid half-width in pixels.
    rin : float
        Inner radius masked from model calculations in au.
    adoptive_zaxis : bool
        Whether to shift the z-axis center for inclined disks.
    cosi_lim : float
        Cosine inclination threshold for the adaptive z-axis.
    beam : list, optional
        Beam in the format ``[major, minor, pa]``.
    width : float
        Spectral smoothing FWHM in km/s.
    f_nvbin : float
        Velocity sub-binning factor relative to ``width``.
    line : str, optional
        Molecule or line identifier.
    iline : int, optional
        Transition index or frequency, depending on ``ilinemode``.
    ilinemode : str
        Transition selection mode.
    database : str
        Molecular database name.
    Tmin, Tmax : float
        Temperature range for partition-function grids.
    nTex : int
        Number of temperature samples.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        model: Callable,
        xlim: list | None = None,
        ylim: list | None = None,
        nsub: list | None = None,
        zstrech: list | None = None,
        reslim: float = 10,
        rin: float = 1.0,
        adoptive_zaxis: bool = True,
        cosi_lim: float = 0.5,
        beam: list | None = None,
        width: float = -1,
        f_nvbin: float = 0.33,
        line: str | None = None,
        iline: int | None = None,
        ilinemode: str = "index",
        database: str = "lamda",
        Tmin: float = 1.0,
        Tmax: float = 500.0,
        nTex: int = 1024,
    ):
        """Set up the model grid and initialize the model."""
        super(Builder, self).__init__()

        # make observer grid
        self._x = x
        self._y = y
        self._z = z
        self._v = v
        self.nx, self.ny, self.nz = len(x), len(y), len(z)
        self.grid = Nested3DObsGrid(
            x, y, z, xlim, ylim, nsub, zstrech, reslim, preserve_z=True
        )  # Plane of sky coordinates
        self.grid2D = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)
        self.cosi_lim = cosi_lim
        self.reslim = reslim
        # Plane of sky coordinates
        self.xs = self.grid.xnest
        self.ys = self.grid.ynest
        self.zs = self.grid.znest
        # disk-local coordinates
        self.xps = None
        self.yps = None
        self.zps = None
        self.Rs = None  # R in cylindarical coordinates
        self.phs = None  # phi in cylindarical coordinates
        # dust layer
        self.Rmid = None
        self.rin = rin

        # velocity
        # refine v-axis if width is given and delv is not sufficient
        if width > 0.0:
            delv = np.mean(v[1:] - v[:-1])
            nvbin = int(delv // (width * f_nvbin))  # at least five
            if nvbin == 0:
                nvbin = 1
        else:
            nvbin = 1
        self.width = width
        self.sigma = width / 2.35482
        self.nvbin = nvbin
        # velocity axis
        self.vaxis = SubGrid1D(v, nvbin)
        self.nv = self.vaxis.nx_sub
        self.delv = self.vaxis.dx_sub
        self.ve = self.vaxis.xe_sub
        self.v = self.vaxis.x_sub
        if width > 0.0:
            self.define_window(width)
        # self._nv = len(v)
        # self._delv = np.mean(v[1:] - v[:-1]) # km / s
        # self._v = v
        # self._ve = np.hstack([v - self.delv * 0.5, v[-1] + 0.5 * self.delv])

        # model
        self.model = model()

        # line
        self.line = line
        self.iline = iline
        _freq = ((ilinemode == "freq") | (ilinemode == "frequency"))
        if (line is not None) * (iline is not None):
            self.mol = Molecule(line, database=database)
            self.mol.moldata[
                line
            ].partition_grid()  # Tmin, Tmax, nTex, scale = 'linear')
            self.mmol = self.mol.moldata[line].weight
            self.Qgrid = (self.mol.moldata[line].Tgrid, self.mol.moldata[line].Qgrid)
            self.trans, self.freq, self.Aul, self.gu, self.gl, self.Eu, self.El = (
                self.mol.moldata[line].params_trans(iline, freq=_freq)
            )
            self.f = doppler_v2f(self.v * 1.0e5, self.freq)
            print(
                "Transition, freq: %s, %.6f GHz" % (self.trans, self.freq * 1.0e-9)
            )  # self.Aul, self.gu, self.gl, self.Eu, self.El)

        # beam
        if beam is not None:
            self.define_beam(beam)
        else:
            self.beam = beam

    def define_beam(self, beam: Sequence[float]):
        """
        Define a Gaussian beam kernel.

        Parameters
        ----------
        beam : sequence of float
            Observational beam in the format ``[major, minor, pa]``.
            The major and minor axes are in au, and ``pa`` is in degrees.
        """
        # save beam info
        self.beam = beam
        # define Gaussian beam
        nx, ny = self.grid2D.nx, self.grid2D.ny
        gaussbeam = gaussian2d(
            self.grid2D.xx.copy(),
            self.grid2D.yy.copy(),
            1.0,
            self.grid2D.xx[ny // 2 - 1 + ny % 2, nx // 2 - 1 + nx % 2],
            self.grid2D.yy[ny // 2 - 1 + ny % 2, nx // 2 - 1 + nx % 2],
            beam[1] / 2.35482,
            beam[0] / 2.35482,
            beam[2],
            peak=True,
        )
        gaussbeam /= np.sum(gaussbeam)
        self.gaussbeam = gaussbeam

    def define_window(self, width: float):
        """
        Define a window function for spectral smoothing.

        Parameters
        ----------
        width : float
            FWHM of the Gaussian window in km/s.
        """
        gausswindow = gauss1d(
            self.v, 1.0, self.v[self.nv // 2 - 1 + self.nv % 2], width / 2.35482
        )
        gausswindow /= np.sum(gausswindow)
        gausswindow = gausswindow.reshape((self.nv, 1, 1))
        self.gausswindow = gausswindow

    def getQrot(self, Ts: np.ndarray):
        """
        Interpolate the rotational partition function.

        Parameters
        ----------
        Ts : np.ndarray
            Temperature array in K.
        """
        shape = Ts.shape
        Qrot = np.interp(Ts.ravel(), self.Qgrid[0], self.Qgrid[1])
        return Qrot.reshape(shape)

    def renest(self):
        """Rebuild the nested grids around the current model offsets."""
        # 3D grid
        self.grid.get_nestinglim(
            self.reslim, dx0=self.dx0, dy0=self.dy0
        )  # Plane of sky coordinates
        self.grid.nest(preserve_z=True)
        # 2D grid
        self.grid2D.get_nestinglim(self.reslim, dx0=self.dx0, dy0=self.dy0)
        self.grid2D.nest()
        # Plane of sky coordinates
        self.xs = self.grid.xnest
        self.ys = self.grid.ynest
        self.zs = self.grid.znest
        # disk-local coordinates
        self.xps = None
        self.yps = None
        self.zps = None
        self.Rs = None
        self.phs = None
        # dust layer
        self.Rmid = None

    def deproject_grid(
        self,
        adoptive_zaxis: bool = True,
        cosi_shift: float = 0.0,
    ):
        """
        Transfer the plane of sky coordinates to disk local coordinates.

        Parameters
        ----------
        adoptive_zaxis : bool
            Whether to shift the z-axis center for inclined disks.
        cosi_shift : float
            Cosine inclination threshold for the additional z-axis shift.
        """
        xp = self.xs
        yp = self.ys
        zp = self.zs

        # rotate by PA
        _xp, _yp = rot2d(xp - self.dx0, yp - self.dy0, self._pa_rad - 0.5 * np.pi)
        _dxp, _dyp = absrot2d(
            self.grid.dxnest, self.grid.dynest, self._pa_rad - 0.5 * np.pi
        )
        # rot = - (- (pa - 90.)); two minuses account for coordinate
        # rotation and the definition of pa.
        # adoptive z axis
        if adoptive_zaxis & (np.abs(np.cos(self._inc_rad)) > self.cosi_lim):
            # set z=0 is in the disk midplane
            zoffset = -np.tan(self._inc_rad) * _yp

            # add further shift for an inclined case
            if np.abs(np.cos(self._inc_rad)) < cosi_shift:
                theta = np.arctan(z_rlim)
                # some math
                la = np.tan(self._inc_rad - theta) * _yp
                lb = np.tan(self._inc_rad) * _yp - la
                lc = np.tan(self._inc_rad + theta) * _yp - la - lb
                zoffset += lb - lc
            self.zoffset = zoffset

            _zp = zp + zoffset  # shift z center back to rectanglar coordinates
            x, y, z = xrot(_xp, _yp, _zp, self._inc_rad)  # rot = - (-inc)
            dx = _dxp
            dy = _dyp / np.abs(np.cos(self._inc_rad)) + self.grid.dznest * np.abs(
                np.sin(self._inc_rad)
            )
            dz = self.grid.dznest * np.abs(
                np.cos(self._inc_rad)
            )  # zoffset cancels out _dyp contribution
        else:
            x, y, z = xrot(_xp, _yp, zp, self._inc_rad)  # rot = - (-inc)
            dx, dy, dz = xabsrot(_dxp, _dyp, self.grid.dznest, self._inc_rad)
            self.zoffset = np.zeros(x.size)

        self.xps = x
        self.yps = y
        self.zps = z
        self.dxps = dx
        self.dyps = dy
        self.dzps = dz

        # cylindarical coordinates
        Rs = np.sqrt(x * x + y * y)  # cylindrical radius
        Rs[Rs < self.rin] = np.nan
        self.Rs = Rs
        self.phs = np.arctan2(y, x)  # azimuthal angle (rad)

        # for dust layer
        x, y = rot2d(
            self.grid.xnest[:, 0] - self.dx0,
            self.grid.ynest[:, 0] - self.dy0,
            self._pa_rad - 0.5 * np.pi,
        )  # in 2D
        y /= np.cos(self._inc_rad)
        self.Rmid = np.sqrt(x * x + y * y)  # radius
        self.adoptive_zaxis = adoptive_zaxis

    def set_model(self, params: Mapping[str, float]):
        """
        Set model and geometric parameters.

        Parameters
        ----------
        params : mapping
            Model parameters. Must include ``dx0``, ``dy0``, ``inc``, and ``pa``.
        """
        self.model.set_params(**params)
        # geometric parameters
        self.dx0 = params["dx0"]
        self.dy0 = params["dy0"]
        self.inc = params["inc"]
        if (self.inc < 0.0) | (self.inc > 180.0):
            print("ERROR\tset_model: inc must be 0 < i < 180.")
            return 0
        _inc_rad = np.radians(self.inc)
        self._inc_rad = _inc_rad
        self.pa = params["pa"]
        self._pa_rad = np.radians(self.pa)
        self.side = 1  # fix side cuz it is automatically determined

    def build_model(self, dv_mode: str, pterm: bool):
        """
        Build model fields on the deprojected grid.

        Parameters
        ----------
        dv_mode : str
            Line-width mode passed to the model.
        pterm : bool
            Whether to include the pressure-gradient term.
        """
        T_g, n_g, vlos, dv, T_d, tau_d = self.model.build(
            self.Rs,
            self.phs,
            self.zps,
            self.Rmid,
            dv_mode=dv_mode,
            mmol=self.mmol,
            pterm=pterm,
        )
        return T_g, n_g, vlos, dv, T_d, tau_d

    def build_cube(
        self,
        Tcmb: float = 2.73,
        f0: float | None = None,
        dist: float = 140.0,
        dv_mode: str = "total",
        pterm: bool = True,
        contsub: bool = True,
        return_Ttau: bool = False,
    ):
        """
        Build a spectral line cube.

        Parameters
        ----------
        Tcmb : float
            Background temperature in K.
        f0 : float, optional
            Rest frequency in GHz. Currently the molecule frequency is used.
        dist : float
            Source distance in pc.
        dv_mode : str
            Line-width mode passed to the model.
        pterm : bool
            Whether to include the pressure-gradient term.
        contsub : bool
            Whether to subtract the continuum contribution.
        return_Ttau : bool
            If True, return temperature and optical-depth components.
        """
        # start = time.time()
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode, pterm=pterm)
        # end = time.time()
        # print('building model took %.2fs'%(end-start))

        # dust
        # T_d = self.grid2D.collapse(T_d)
        # tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature
        #  of each gas layer at every velocity channel.
        # line profile function
        if (self.model.dv > 0.0) | (dv_mode == "thermal"):
            lnprofs = spectra.glnprof_series(
                self.v, vlos.ravel(), dv.ravel(), unit_scale=1.0e-5
            )
        else:
            lnprofs = spectra.boxlnprof_series(self.v, vlos.ravel(), unit_scale=1.0e-5)

        # get nv
        # start = time.time()
        nv_g = spectra.to_xyzv(n_g.ravel(), lnprofs)
        # nv_g = n_g.ravel()[:, np.newaxis] * lnprofs
        nv_g = nv_g.reshape((self.grid.nxy, self.nz, self.nv))
        # end = time.time()
        # print('to xyzv took %.2fs'%(end-start))

        # start = time.time()
        Qrots = self.getQrot(T_g)
        # end = time.time()
        # print('get Qrot took %.2fs'%(end-start))

        # to cube
        # start = time.time()
        Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = transfer.Tnv_to_cube(
            T_g,
            nv_g,
            self.grid.znest,
            self.grid.dznest * auTOcm,
            self.freq,
            self.Aul,
            self.Eu,
            self.gu,
            Qrots,
        )  # return (nxy, nv)
        # end = time.time()
        # print('to cube took %.2fs'%(end-start))

        Tv_gf = Tv_gf.clip(1.0, None)  # safety net to avoid zero division
        Tv_gr = Tv_gr.clip(1.0, None)

        if return_Ttau:
            Tv_gf = self.grid.collapse2D(Tv_gf)
            Tv_gr = self.grid.collapse2D(Tv_gr)
            tau_v_gf = self.grid.collapse2D(tau_v_gf)
            tau_v_gr = self.grid.collapse2D(tau_v_gr)
            return np.array([Tv_gf, tau_v_gf, Tv_gr, tau_v_gr])

        # radiative transfer
        # start = time.time()
        def _Bv(T, v):
            return Bvppx(
                T, v, self.grid.dx, self.grid.dy, dist=dist, au=True
            )  # in unit of Jy/pixel with the final pixel size

        f = doppler_v2f((self.v - self.model.vsys) * 1.0e5, self.freq)
        fs = f[np.newaxis, :]
        _Bv_cmb = _Bv(Tcmb, fs)
        _Bv_gf = _Bv(Tv_gf, fs)
        _Bv_gr = _Bv(Tv_gr, fs)
        _Bv_d = _Bv(T_d[:, np.newaxis], fs)
        tau_d = tau_d[:, np.newaxis]
        Iv = solveRT_TL(
            _Bv_gf, _Bv_gr, _Bv_d, _Bv_cmb,
            tau_v_gf, tau_v_gr, tau_d, contsub=contsub
        )
        # end = time.time()
        # print('radiative transfer took %.2fs'%(end-start))

        Iv = np.transpose(
            self.grid.collapse2D(Iv.T, collapse_mode="mean"), axes=(0, 2, 1)
        )
        # Input Iv shape (nxy, nv)
        # ->[.T] (nv, nxy)
        # ->[collapse2D] (nv, nx, ny)
        # ->[transpose] (nv, ny, nx)

        # spectral smoothing
        if self.width > 0.0:
            Iv = convolve(Iv, self.gausswindow, mode="same")
            if self.nvbin > 1:
                Iv_avg = np.array(
                    [Iv[i :: self.nvbin, :, :] for i in range(self.nvbin)]
                )
                Iv = np.nanmean(Iv_avg, axis=0)

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.grid2D.xx.copy(),
                self.grid2D.yy.copy(),
                Iv,
                self.beam,
                self.gaussbeam,
            )

        return Iv

    def build_cont(
        self,
        freq: float,
        Tcmb: float = 2.73,
        dist: float = 140.0,
        return_Ttau: bool = False,
    ):
        """
        Build a continuum image.

        Parameters
        ----------
        freq : float
            Frequency in Hz.
        Tcmb : float
            Background temperature in K.
        dist : float
            Source distance in pc.
        return_Ttau : bool
            If True, return collapsed dust temperature and optical depth.
        """
        # start = time.time()
        _, _, _, _, T_d, tau_d = self.build_model(
            dv_mode = 'total', pterm = False
        )  # dv_mode, pterm do not matter for dust
        # end = time.time()
        # print('building model took %.2fs'%(end-start))

        # return tau
        if return_Ttau:
            return self.grid.collapse2D(T_d).T, self.grid.collapse2D(tau_d).T

        # radiative transfer
        def _Bv(T, v):
            return Bvppx(
                T, v, self.grid2D.dx, self.grid2D.dy, dist=dist, au=True
            )  # in unit of Jy/pixel with the final pixel size

        _Bv_cmb = _Bv(Tcmb, freq)
        _Bv_d = _Bv(T_d, freq)

        # dust intensity
        Iv_d = (_Bv_d - _Bv_cmb) * (1.0 - np.exp(-tau_d))  # in 1D (nxy,)
        Iv = self.grid.collapse2D(Iv_d,).T  # (x, y) to (y, x)

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.grid2D.xx.copy(),
                self.grid2D.yy.copy(),
                Iv,
                self.beam,
                self.gaussbeam,
            )

        return Iv

    def show_model_sideview(
        self,
        dv_mode: str = "total",
        pterm: bool = True,
        cmap: str = "viridis",
        savefig: bool = False,
        showfig: bool = True,
        outname: str = "model_sideview",
        vmax: float = 1.0,
        vmin: float = 1.0e-5,
        clip_min: float = 1.0e-10,
    ):
        """
        Show side-view slices of the gas density model.

        Parameters
        ----------
        dv_mode : str
            Line-width mode passed to the model.
        pterm : bool
            Whether to include the pressure-gradient term.
        cmap : str
            Matplotlib colormap name.
        savefig, showfig : bool
            Whether to save or display the figure.
        outname : str
            Output filename stem.
        vmax, vmin : float
            Fractions of the maximum density used for log scaling.
        clip_min : float
            Minimum density before log scaling.
        """
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode, pterm=pterm)

        # n_g = np.log10(self.n_g) # in log scale
        # n_g = self.yps
        # n_g = self.zoffset
        # vmin, vmax = np.nanmin(n_g) * 0.01, np.nanmax(n_g) * 0.01
        vmax = np.log10(np.nanmax(n_g) * vmax)
        vmin = np.log10(np.nanmax(n_g) * vmin)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes

        # index of disk center
        # xi0 = np.argmin((self._x - self.dx0)**2.)
        # yi0 = np.argmin((self._y - self.dy0)**2.)

        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for level in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[level, :]
            xmin, xmax = _grid.xlim[level]
            zmin, zmax = _grid.zlim[level]

            d_plt = _grid.collapse(n_g, upto=level)
            d_plt.clip(clip_min, None)  # to avoide -inf
            d_plt = np.log10(d_plt)  # in log scale

            # hide parental layer
            if level <= _grid.nlevels - 2:
                ximin, ximax = _grid.xinest[(level + 1) * 2 : (level + 2) * 2]
                yimin, yimax = _grid.yinest[(level + 1) * 2 : (level + 2) * 2]
                d_plt[ximin : ximax + 1, yimin : yimax + 1] = np.nan
                # xi0 = (ximax + ximin) // 2
                # yi0 = (yimin + yimax) // 2

            _xx, _yy, _zz = _grid.get_grid(level)
            xi0, yi0 = np.unravel_index(
                np.argmin(
                    (_xx[:, :, nz // 2] - self.dx0) ** 2
                    + (_yy[:, :, nz // 2] - self.dy0) ** 2
                ),
                _xx[:, :, nz // 2].shape,
            )

            if self.adoptive_zaxis:
                zoffset = _grid.collapse(self.zoffset, upto=level)
                _zz += zoffset

            im1 = ax1.pcolormesh(
                _zz[:, yi0, :],
                _xx[:, yi0, :],
                d_plt[:, yi0, :],  # [:, ny//2, :],
                alpha=1.0,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
            )
            im2 = ax2.pcolormesh(
                _zz[xi0, :, :],
                _yy[xi0, :, :],
                d_plt[xi0, :, :],  # [nx//2, :, :]
                alpha=1.0,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
            )
            # rect = plt.Rectangle((zmin, xmin),
            #    zmax - zmin, xmax - xmin, edgecolor = 'white', facecolor = "none",
            #    linewidth = 0.5, ls = '--')
            # ax1.add_patch(rect)

        # ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        gmin = np.min([np.min(_grid.xlim[0]), np.min(_grid.ylim[0])])
        gmax = np.max([np.max(_grid.xlim[0]), np.max(_grid.ylim[0])])
        for ax in axes:
            ax.set_xlim(gmin, gmax)
            ax.set_ylim(gmin, gmax)
            ax.set_aspect(1)

        for ax, im in zip(axes, [im1, im2]):
            add_colorbar_toaxis(ax, im, loc="top", width="3%", pad="3%")

        ax1.set_ylabel(r"$x^\prime$ (au)")
        ax1.set_xlabel(r"$z^\prime$ (au)")
        ax2.set_ylabel(r"$y^\prime$ (au)")
        ax2.set_xlabel(r"$z^\prime$ (au)")

        fig.tight_layout()

        if savefig:
            fig.savefig(outname + ".png", dpi=300, transparent=True)
        if showfig:
            plt.show()
        plt.close()

    def show_model_faceview(
        self,
        dv_mode: str = "total",
        cmap: str = "viridis",
        savefig: bool = False,
        showfig: bool = True,
        outname: str = "model_sideview",
        vmax: float = 0.90,
        vmin: float = 0.0,
    ):
        """
        Show a face-on slice of the gas density model.

        Parameters
        ----------
        dv_mode : str
            Line-width mode passed to the model.
        cmap : str
            Matplotlib colormap name.
        savefig, showfig : bool
            Whether to save or display the figure.
        outname : str
            Output filename stem.
        vmax, vmin : float
            Color scale limits. ``vmax`` is interpreted as a fraction of the
            maximum density.
        """
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode)
        # n_g = self.grid.collapse(n_g)

        vmax *= np.nanmax(n_g)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig, ax1 = plt.subplots(1, 1)
        # ax1 = axes

        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for level in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[level, :]
            xmin, xmax = _grid.xlim[level]
            ymin, ymax = _grid.ylim[level]

            d_plt = _grid.collapse(n_g, upto=level)

            # hide parental layer
            if level <= _grid.nlevels - 2:
                ximin, ximax = _grid.xinest[(level + 1) * 2 : (level + 2) * 2]
                yimin, yimax = _grid.yinest[(level + 1) * 2 : (level + 2) * 2]
                d_plt[ximin : ximax + 1, yimin : yimax + 1] = np.nan

            _xx, _yy, _zz = _grid.get_grid(level)

            if self.adoptive_zaxis:
                zoff = _grid.collapse(self.zoffset, upto=level)
                _zz += zoff

            ax1.pcolormesh(
                _xx[:, :, nz // 2],
                _yy[:, :, nz // 2],
                d_plt[:, :, nz // 2],
                alpha=1.0,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
            )
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor="white",
                facecolor="none",
                linewidth=0.5,
                ls="--",
            )
            ax1.add_patch(rect)

        # ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        ax1.set_xlim(_grid.xlim[0])
        ax1.set_ylim(_grid.ylim[0])

        ax1.set_xlabel(r"$x$ (au)")
        ax1.set_ylabel(r"$y$ (au)")

        fig.tight_layout()

        if savefig:
            fig.savefig(outname + ".png", dpi=300, transparent=True)
        if showfig:
            plt.show()
        plt.close()

    def show_model_4Dview(
        self,
        dv_mode: str = "total",
        cmap: str = "viridis",
        savefig: bool = False,
        showfig: bool = True,
        outname: str = "model_sideview",
        vmax: float = 0.90,
        vmin: float = 0.0,
        nsparse: int = 10,
    ):
        """
        Show a sparse 3D scatter view of the gas density model.

        Parameters
        ----------
        dv_mode : str
            Line-width mode passed to the model.
        cmap : str
            Matplotlib colormap name.
        savefig, showfig : bool
            Whether to save or display the figure.
        outname : str
            Output filename stem.
        vmax, vmin : float
            Color scale limits. ``vmax`` is interpreted as a fraction of the
            maximum density.
        nsparse : int
            Grid stride divisor used to sparsify the scatter plot.
        """
        T_g, n_g, vlos, dv, T_d, tau_d = self.build_model(dv_mode=dv_mode)
        # n_g = self.grid.collapse(n_g)

        vmax *= np.nanmax(n_g)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # from upper to lower
        _grid = copy.deepcopy(self.grid)
        for level in range(_grid.nlevels):
            nx, ny, nz = _grid.ngrids[level, :]
            xmin, xmax = _grid.xlim[level]
            ymin, ymax = _grid.ylim[level]

            d_plt = _grid.collapse(n_g, upto=level)

            # hide parental layer
            if level <= _grid.nlevels - 2:
                ximin, ximax = _grid.xinest[(level + 1) * 2 : (level + 2) * 2]
                yimin, yimax = _grid.yinest[(level + 1) * 2 : (level + 2) * 2]
                d_plt[ximin : ximax + 1, yimin : yimax + 1, :] = np.nan

            _xx, _yy, _zz = _grid.get_grid(level)

            if self.adoptive_zaxis:
                zoff = _grid.collapse(self.zoffset, upto=level)
                _zz += zoff

            x_sparse = nx // nsparse
            y_sparse = ny // nsparse
            z_sparse = nz // nsparse
            d_plt = d_plt[::x_sparse, ::y_sparse, ::z_sparse].ravel()
            _xx_plt = _xx[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            _yy_plt = _yy[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            _zz_plt = _zz[::x_sparse, ::y_sparse, ::z_sparse].ravel()[~np.isnan(d_plt)]
            d_plt = d_plt[~np.isnan(d_plt)]
            ax.scatter(
                _xx_plt,
                _yy_plt,
                _zz_plt,
                c=d_plt,
                alpha=1.0,
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
            )

        # ax1.set_xlim(_grid.zlim[0][0], _grid.zlim[0][1])
        # ax1.set_xlim(_grid.xlim[0])
        # ax1.set_ylim(_grid.ylim[0])

        ax.set_xlabel(r"$x$ (au)")
        ax.set_ylabel(r"$y$ (au)")
        ax.set_zlabel(r"$z$ (au)")

        fig.tight_layout()

        if savefig:
            fig.savefig(outname + ".png", dpi=300, transparent=True)
        if showfig:
            plt.show()
        plt.close()


class Builder_SSDisk(object):
    """
    Builder for a geometrically thin disk model.

    Parameters
    ----------
    model : Callable
        Model class or factory used to instantiate the disk model.
    axes_model : list
        Model-coordinate axes. For ``coordinate_type="polar"``, use
        ``[r, phi]``.
    axes_sky : list
        Sky-coordinate axes in the format ``[x, y, v]``.
    xlim, ylim : list, optional
        Nested-grid limits in au.
    nsub : list, optional
        Subdivision factors for nested-grid levels.
    reslim : float
        Default nested-grid half-width in pixels.
    beam : list, optional
        Beam in the format ``[major, minor, pa]``.
    coordinate_type : str
        Coordinate system used by ``axes_model``.
    """

    def __init__(
        self,
        model: Callable,
        axes_model: list,
        axes_sky: list,
        xlim: list | None = None,
        ylim: list | None = None,
        nsub: list | None = None,
        reslim: float = 10,
        beam: list | None = None,
        coordinate_type: str = "polar",
    ):
        """Set up the model grid and initialize the model."""
        super(Builder_SSDisk, self).__init__()

        # model
        self.model = model()
        self._model = model

        # model grid
        if coordinate_type == "polar":
            self.build_polar_grid(axes_model)

        # sky grid
        self.reslim = reslim
        self.nsub = nsub
        x, y, v = axes_sky
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

    def define_beam(self, beam: Sequence[float]):
        """
        Define a Gaussian beam kernel.

        Parameters
        ----------
        beam : sequence of float
            Observational beam in the format ``[major, minor, pa]``.
            The major and minor axes are in au, and ``pa`` is in degrees.
        """
        # save beam info
        self.beam = beam
        # define Gaussian beam
        nx, ny = self.skygrid.nx, self.skygrid.ny
        xx = self.skygrid.xx.copy()
        yy = self.skygrid.yy.copy()
        gaussbeam = gaussian2d(
            xx,
            yy,
            1.0,
            xx[ny // 2 - 1 + ny % 2, nx // 2 - 1 + nx % 2],
            yy[ny // 2 - 1 + ny % 2, nx // 2 - 1 + nx % 2],
            beam[1] / 2.35482,
            beam[0] / 2.35482,
            beam[2],
            peak=True,
        )
        gaussbeam /= np.sum(gaussbeam)
        self.gaussbeam = gaussbeam

    def build_sky_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlim: list | None = None,
        ylim: list | None = None,
        nsub: list | None = None,
        reslim: float = 10,
    ):
        """
        Build the nested sky grid.

        Parameters
        ----------
        x, y : np.ndarray
            Plane-of-sky coordinate axes in au.
        xlim, ylim : list, optional
            Nested-grid limits in au.
        nsub : list, optional
            Subdivision factors for nested-grid levels.
        reslim : float
            Default nested-grid half-width in pixels.
        """
        self.skygrid = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)

    def build_polar_grid(self, axes: Sequence[np.ndarray]):
        """
        Build the polar model grid.

        Parameters
        ----------
        axes : sequence of np.ndarray
            Radius and azimuthal axes in the format ``[r, phi]``.
        """
        r, phi = axes
        self.r = r
        self.phi = phi
        self.nr = len(r)
        self.nphi = len(phi)

        rr, phph = np.meshgrid(r, phi, indexing="ij")
        self.rs = rr.ravel()
        self.phis = phph.ravel()

    def set_model(self, params: Mapping[str, float]):
        """
        Set model and geometric parameters.

        Parameters
        ----------
        params : mapping
            Model parameters. Must include ``dx0``, ``dy0``, ``inc``, and ``pa``.
        """
        # geometric parameters
        self.dx0 = params["dx0"]
        self.dy0 = params["dy0"]
        self.inc = params["inc"]
        self.__inc_rad = np.radians(self.inc)
        self.pa = params["pa"]
        self.__pa_rad = np.radians(self.pa)
        # set parameters to model
        self.model.set_params(**params)

    def skygrid_info(self):
        """Print sky-grid information."""
        self.skygrid.gridinfo()

    def build_model(self, build_args: Sequence | None = None):
        """
        Build model fields on the polar grid.

        Parameters
        ----------
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        """
        if build_args is not None:
            I_int, vlos, dv = self.model.build(self.rs, self.phis, *build_args)
        else:
            I_int, vlos, dv = self.model.build(
                self.rs,
                self.phis,
            )
        return I_int, vlos, dv

    def project_grid(self):
        """Project the model grid to the plane of the sky."""
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

    def project_quantity(self, q: np.ndarray):
        """
        Interpolate a model quantity onto the nested sky grid.

        Parameters
        ----------
        q : np.ndarray
            Quantity sampled on the model grid.
        """
        # interpolator
        q_proj = griddata(
            (self.xproj, self.yproj),
            q,
            (self.skygrid.xnest, self.skygrid.ynest),
            method="linear",
            fill_value=0.0,
        )
        return q_proj

    def build_cube(
        self,
        build_args: Sequence | None = None,
        I_pre: np.ndarray | None = None,
        vlos_pre: np.ndarray | None = None,
        dv_pre: np.ndarray | None = None,
    ):
        """
        Build a spectral cube from a thin-disk intensity model.

        Parameters
        ----------
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        I_pre, vlos_pre, dv_pre : np.ndarray, optional
            Precomputed intensity, line-of-sight velocity, and line width.
        """
        # model build
        _I_int, _vlos, _dv = self.build_model(build_args)
        # if any pre-calculations are provided or not
        I_int = _I_int if I_pre is None else I_pre
        vlos = _vlos if vlos_pre is None else vlos_pre
        dv = _dv if dv_pre is None else dv_pre

        # calculate sky projection
        self.project_grid()

        """ new version; line profile first
        lnprofs = spectra.glnprof_series(self.v, vlos, dv) # x,y,v
        lnprofs[np.isnan(lnprofs)] = 0.
        _Iv = np.tile(I_int, (self.nv, 1,)) * lnprofs

        Iv = np.array([
            self.project_quantity(_Iv[i,:]) for i in range(self.nv)
            ])
        """

        #'''old version; projection first
        I_proj = self.project_quantity(I_int)
        v_proj = self.project_quantity(vlos)
        dv_proj = self.project_quantity(dv)

        # line profile function
        lnprofs = spectra.glnprof_series(self.v, v_proj, dv_proj)  # x,y,v
        # Iv = np.tile(I_proj, (1, self.nv)) * lnprofs
        Iv = I_proj[:, np.newaxis] * lnprofs
        Iv = np.transpose(Iv, axes=(1, 0))
        #'''

        # collapse
        Iv = self.skygrid.high_dimensional_collapse(
            Iv, fill="zero", collapse_mode="mean"
        )  # x,y,v

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.skygrid.xx.copy(),
                self.skygrid.yy.copy(),
                Iv,
                self.beam,
                self.gaussbeam,
            )

        return Iv

    def visualize_grid(
        self,
        keys: Sequence[str] = ["intensity", "vlos", "dv"],
        savefig: bool = False,
        showfig: bool = True,
        outname: str | None = None,
        cmap: str = "viridis",
        build_args: Sequence | None = None,
    ):
        """
        Visualize projected model-grid quantities.

        Parameters
        ----------
        keys : sequence of str
            Quantities to visualize. Supported values are ``"intensity"``,
            ``"vlos"``, and ``"dv"``.
        savefig, showfig : bool
            Whether to save or display the figure.
        outname : str, optional
            Output filename stem.
        cmap : str
            Matplotlib colormap name.
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        """
        # visualize grid
        I_int, vlos, dv = self.build_model(build_args)
        self.project_grid()

        qs = []
        for q, key in zip([I_int, vlos, dv], ["intensity", "vlos", "dv"]):
            if key in keys:
                _q = self.project_quantity(q)
                if key == "intensity":
                    _q = np.log10(_q)
                    _q[_q < 0.0] = np.nan
                qs.append(_q)

        for q, key in zip(qs, keys):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            _outname = outname + "_%s.png" % key if outname is not None else None
            self.skygrid.visualize_grid(
                q, ax=ax, showfig=False, savefig=False, outname=_outname, cmap=cmap
            )
            ax.text(0.1, 0.9, key, transform=ax.transAxes, ha="left", va="top")
            if savefig:
                fig.savefig(outname + "_%s.png" % key, dpi=300)
            if showfig:
                plt.show()
            plt.close()


class Builder_TSD(Builder_SSDisk):
    """
    Builder for Two Slab Disk Model.

    Parameters
    ----------
    model : Callable
        Model class or factory used to instantiate the disk model.
    axes_model : list
        Model-coordinate axes. For ``coordinate_type="polar"``, use
        ``[r, phi]``.
    axes_sky : list
        Sky-coordinate axes in the format ``[x, y, v]``.
    xlim, ylim : list, optional
        Nested-grid limits in au.
    nsub : list, optional
        Subdivision factors for nested-grid levels.
    reslim : float
        Default nested-grid half-width in pixels.
    beam : list, optional
        Beam in the format ``[major, minor, pa]``.
    coordinate_type : str
        Coordinate system used by ``axes_model``.
    line : str, optional
        Molecule or line identifier.
    iline : int, optional
        Transition index.
    dust_opacity : float, optional
        Dust opacity.
    Tmin, Tmax : float
        Temperature range for partition-function grids.
    nTex : int
        Number of temperature samples.
    """

    def __init__(
        self,
        model: Callable,
        axes_model: list,
        axes_sky: list,
        xlim: list | None = None,
        ylim: list | None = None,
        nsub: list | None = None,
        reslim: float = 10,
        beam: list | None = None,
        coordinate_type: str = "polar",
        line: str | None = None,
        iline: int | None = None,
        dust_opacity: float | None = None,
        Tmin: float = 1.0,
        Tmax: float = 500.0,
        nTex: int = 1024,
    ):
        """Set up the model grid and initialize the model."""
        super().__init__(
            model,
            axes_model=axes_model,
            axes_sky=axes_sky,
            xlim=xlim,
            ylim=ylim,
            nsub=nsub,
            reslim=reslim,
            beam=beam,
            coordinate_type=coordinate_type,
        )
        # super().__init__()

        # line
        self.line = line
        self.iline = iline
        if (line is not None) * (iline is not None):
            self.mol = Molecule(line)
            self.mol.moldata[line].partition_grid()  # Tmin, Tmax, nTex, scale="linear")
            self.mmol = self.mol.moldata[line].weight
            self.Qgrid = (self.mol.moldata[line]._Tgrid, self.mol.moldata[line]._PFgrid)
            self.trans, self.freq, self.Aul, self.gu, self.gl, self.Eu, self.El = (
                self.mol.moldata[line].params_trans(iline)
            )

        # dust
        self.kappa = dust_opacity

    def build_model(self, build_args: Sequence | None = None):
        """
        Build single-layer disk fields on the polar grid.

        Parameters
        ----------
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        """
        if build_args is not None:
            T_g, N_g, vlos, dv, T_d, Sig_d = self.model.build(
                self.rs, self.phis, *build_args
            )
        else:
            T_g, N_g, vlos, dv, T_d, Sig_d = self.model.build(
                self.rs,
                self.phis,
            )
        return T_g, N_g, vlos, dv, T_d, Sig_d

    def build_cube(
        self,
        build_args: Sequence | None = None,
        Tcmb: float = 2.73,
        dist: float = 140.0,
        f0: float | None = None,
        contsub: bool = True,
        Ng_pre: np.ndarray | None = None,
        Tg_pre: np.ndarray | None = None,
        vlos_pre: np.ndarray | None = None,
        Td_pre: np.ndarray | None = None,
        Sigd_pre: np.ndarray | None = None,
        dv_pre: np.ndarray | None = None,
        return_tau: bool = False,
    ):
        """
        Build a single-layer disk spectral cube.

        Parameters
        ----------
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        Tcmb : float
            Background temperature in K.
        dist : float
            Source distance in pc.
        f0 : float, optional
            Frequency in GHz. If omitted, the selected line frequency is used.
        contsub : bool
            Whether to subtract the continuum contribution.
        Ng_pre, Tg_pre, vlos_pre, Td_pre, Sigd_pre, dv_pre : np.ndarray, optional
            Precomputed gas column density, gas temperature, line-of-sight
            velocity, dust temperature, dust surface density, and line width.
        return_tau : bool
            If True, return the collapsed gas optical-depth cube.
        """
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
        # print(v_proj, dv_proj)

        # line profile function
        lnprofs = spectra.glnprof_series(
            self.v, v_proj, dv_proj, unit_scale=1.0e-5
        )  # x,y,v
        Ng_v = (
            np.tile(
                Ng_proj,
                (
                    self.nv,
                    1,
                ),
            )
            * lnprofs
        )
        Tg_v = np.tile(
            Tg_proj,
            (
                self.nv,
                1,
            ),
        )  # * lnprofs
        Sigd_v = np.tile(
            Sigd_proj,
            (
                self.nv,
                1,
            ),
        )  # * lnprofs
        Td_v = np.tile(
            Td_proj,
            (
                self.nv,
                1,
            ),
        )  # * lnprofs

        # Nv to tau_v
        tau_v_g = transfer.Nv_to_tauv(
            Tg_v, Ng_v, self.freq, self.Aul, self.Eu, self.gu, self.Qgrid
        )
        Tg_v = Tg_v.clip(1.0, None)
        tau_v_d = Sigd_v * self.kappa

        if return_tau:
            # collapse
            Iv = self.skygrid.high_dimensional_collapse(
                tau_v_g, fill="zero", collapse_mode="mean"
            )
            return Iv

        # radiative transfer
        def _Bv(T, v):
            return Bvppx(
                T, v, self.skygrid.dx, self.skygrid.dy, dist=dist, au=True
            )  # in unit of Jy/pixel with the final pixel size

        if f0 is None:
            f0 = self.freq * 1.0e-9
        _Bv_cmb = _Bv(Tcmb, f0)
        _Bv_g = _Bv(Tg_v, f0)
        _Bv_d = _Bv(Td_v, f0)
        # Iv = solve_MLRT(_Bv_gf, _Bv_gr, _Bv_d,
        #    tau_v_gf, tau_v_gr, tau_d, _Bv_cmb, self.nv)
        Iv = (
            _Bv_cmb * (-1.0 + np.exp(-tau_v_g - tau_v_d))
            + _Bv_d * (1.0 - np.exp(-tau_v_d)) * np.exp(-tau_v_g)
            + _Bv_g * (1.0 - np.exp(-tau_v_g))
        )

        # contsub
        # if contsub:
        #    Iv_d = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_v_d))
        #    Iv -= Iv_d # add continuum back

        # collapse
        Iv = self.skygrid.high_dimensional_collapse(
            Iv, fill="zero", collapse_mode="mean"
        )

        # Convolve beam if given
        if self.beam is not None:
            Iv = beam_convolution(
                self.skygrid.xx.copy(),
                self.skygrid.yy.copy(),
                Iv,
                self.beam,
                self.gaussbeam,
            )

        return Iv

    def visualize_grid(
        self,
        keys: Sequence[str] = ["intensity", "vlos", "dv"],
        savefig: bool = False,
        showfig: bool = True,
        outname: str | None = None,
        cmap: str = "viridis",
        build_args: Sequence | None = None,
    ):
        """
        Visualize projected single-layer model quantities.

        Parameters
        ----------
        keys : sequence of str
            Quantities to visualize. Supported values are ``"intensity"``,
            ``"vlos"``, and ``"dv"``.
        savefig, showfig : bool
            Whether to save or display the figure.
        outname : str, optional
            Output filename stem.
        cmap : str
            Matplotlib colormap name.
        build_args : sequence, optional
            Additional positional arguments passed to ``model.build``.
        """
        # visualize grid
        I_int, vlos, dv = self.build_model(build_args)
        self.project_grid()

        qs = []
        for q, key in zip([I_int, vlos, dv], ["intensity", "vlos", "dv"]):
            if key in keys:
                _q = self.project_quantity(q)
                if key == "intensity":
                    _q = np.log10(_q)
                    _q[_q < 0.0] = np.nan
                qs.append(_q)

        for q, key in zip(qs, keys):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            _outname = outname + "_%s.png" % key if outname is not None else None
            self.skygrid.visualize_grid(
                q, ax=ax, showfig=False, savefig=False, outname=_outname, cmap=cmap
            )
            ax.text(0.1, 0.9, key, transform=ax.transAxes, ha="left", va="top")
            if savefig:
                fig.savefig(outname + "_%s.png" % key, dpi=300)
            if showfig:
                plt.show()
            plt.close()


def rot2d(x: np.ndarray, y: np.ndarray, ang: float):
    """
    Rotate 2D coordinates.

    Parameters
    ----------
    x, y : np.ndarray
        Input coordinates.
    ang : float
        Rotation angle in radians.
    """
    return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)


def absrot2d(x: np.ndarray, y: np.ndarray, ang: float):
    """
    Rotate 2D cell widths using absolute trigonometric factors.

    Parameters
    ----------
    x, y : np.ndarray
        Input cell widths.
    ang : float
        Rotation angle in radians.
    """
    return x * np.abs(np.cos(ang)) + y * np.abs(np.sin(ang)), x * np.abs(
        np.sin(ang)
    ) + y * np.abs(np.cos(ang))


def xrot(x: np.ndarray, y: np.ndarray, z: np.ndarray, ang: float):
    """
    Rotate 3D coordinates around the x-axis.

    Parameters
    ----------
    x, y, z : np.ndarray
        Input coordinates.
    ang : float
        Rotation angle in radians.
    """
    return x, y * np.cos(ang) - z * np.sin(ang), y * np.sin(ang) + z * np.cos(ang)


def xabsrot(x: np.ndarray, y: np.ndarray, z: np.ndarray, ang: float):
    """
    Rotate 3D cell widths around the x-axis.

    Parameters
    ----------
    x, y, z : np.ndarray
        Input cell widths.
    ang : float
        Rotation angle in radians.
    """
    abscos = np.abs(np.cos(ang))
    abssin = np.abs(np.sin(ang))
    return x, y * abscos + z * abssin, y * abssin + z * abscos


def yp2y(yp: np.ndarray, z: np.ndarray, inc: float):
    """
    Deproject the plane-of-sky y-coordinate to the disk-local y-coordinate.

    Parameters
    ----------
    yp : np.ndarray
        Plane-of-sky y-coordinate.
    z : np.ndarray
        Disk-local z-coordinate.
    inc : float
        Inclination angle in radians.
    """
    return (yp - z * np.sin(inc)) / np.cos(inc)


# Planck function
def Bv(T: ArrayLike, v: ArrayLike):
    """
    Calculate the Planck function in cgs units.

    Parameters
    ----------
    T : array-like
        Temperature in K.
    v : array-like
        Frequency in Hz.
    """
    # v = v * 1.e9 # GHz --> Hz
    # print((hp*v)/(kb*T))
    exp = np.exp((hp * v) / (kb * T)) - 1.0
    fterm = (2.0 * hp * v * v * v) / (clight * clight)
    return fterm / exp


# Planck function
def Bvppx(
    T: ArrayLike,
    v: ArrayLike,
    px: ArrayLike,
    py: ArrayLike,
    dist: float = 140.0,
    au: bool = True,
):
    """
    Calculate the Planck function in Jy per pixel.

    Parameters
    ----------
    T : array-like
        Temperature in K.
    v : array-like
        Frequency in Hz.
    px, py : array-like
        Pixel sizes. Interpreted as au when ``au`` is True, otherwise degrees.
    dist : float
        Source distance in pc.
    au : bool
        Whether ``px`` and ``py`` are given in au.
    """
    # unit
    # v = v * 1.e9 # GHz --> Hz

    # Bv in cgs
    exp = np.exp((hp * v) / (kb * T)) - 1.0
    fterm = (2.0 * hp * v * v * v) / (clight * clight)
    Bv = fterm / exp

    # From cgs to Jy/str
    Bv = Bv * 1.0e-7 * 1.0e4  # cgs --> MKS
    Bv = Bv * 1.0e26  # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(px / dist / 3600.0)  # au --> radian
        py = np.radians(py / dist / 3600.0)  # au --> radian
    else:
        px = np.radians(px)  # deg --> rad
        py = np.radians(py)  # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(px * py)
    Bv *= one_pixel_area  # Iv (Jy per pixel)
    return Bv


def cgs_to_Jyppx(
    dx: ArrayLike,
    dy: ArrayLike,
    dist: float = 140.0,
    au: bool = True,
):
    """
    Convert cgs specific intensity to Jy per pixel.

    Parameters
    ----------
    dx, dy : array-like
        Pixel sizes. Interpreted as au when ``au`` is True, otherwise degrees.
    dist : float
        Source distance in pc.
    au : bool
        Whether ``dx`` and ``dy`` are given in au.
    """
    # From cgs to Jy/str
    f = 1.0e-7 * 1.0e4  # cgs --> MKS
    f *= 1.0e26  # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(dx / dist / 3600.0)  # au --> radian
        py = np.radians(dy / dist / 3600.0)  # au --> radian
    else:
        px = np.radians(dx)  # deg --> rad
        py = np.radians(dy)  # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(dx * dy)
    f *= one_pixel_area  # Iv (Jy per pixel)
    return f


# Jy/beam
def Bv_Jybeam(T: ArrayLike, v: ArrayLike, bmaj: float, bmin: float):
    """
    Calculate the Planck function in Jy per beam.

    Parameters
    ----------
    T : array-like
        Temperature in K.
    v : array-like
        Frequency in GHz.
    bmaj, bmin : float
        Beam major and minor axes in arcsec.
    """

    # units
    bmaj = np.radians(bmaj / 3600.0)  # arcsec --> radian
    bmin = np.radians(bmin / 3600.0)  # arcsec --> radian
    v = v * 1.0e9  # GHz --> Hz

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi / (4.0 * np.log(2.0))  # beam(rad) -> beam (sr)
    bTOstr = bmaj * bmin * C2  # beam --> str

    # print(np.nanmax((hp*v)/(kb*T)), np.nanmin(T))
    exp = np.exp((hp * v) / (kb * T)) - 1.0
    # print(np.nanmax(exp))
    fterm = (2.0 * hp * v * v * v) / (clight * clight)
    Bv = fterm / exp

    # cgs --> Jy/beam
    Bv = Bv * 1.0e-7 * 1.0e4  # cgs --> MKS
    Bv = Bv * 1.0e26  # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv * bTOstr  # Jy/str --> Jy/beam
    return Bv


def solveRT_TL(
    Sv_gf: np.ndarray,
    Sv_gr: np.ndarray,
    Sv_d: np.ndarray,
    Sv_bg: np.ndarray,
    tau_v_gf: np.ndarray,
    tau_v_gr: np.ndarray,
    tau_d: np.ndarray,
    contsub: bool = True,
):
    """
    Solve two-layer radiative transfer with dust emission.

    Parameters
    ----------
    Sv_gf, Sv_gr : np.ndarray
        Front and rear gas source functions.
    Sv_d : np.ndarray
        Dust source function.
    Sv_bg : np.ndarray
        Background source function.
    tau_v_gf, tau_v_gr : np.ndarray
        Front and rear gas optical depths.
    tau_d : np.ndarray
        Dust optical depth.
    contsub : bool
        Whether to subtract the continuum contribution.
    """
    Iv_d = (Sv_d - Sv_bg) * (1.0 - np.exp(-tau_d)) if contsub else 0.0
    Iv = (
        Sv_bg * (np.exp(-tau_v_gf - tau_d - tau_v_gr) - 1.0)
        + Sv_gr * (1.0 - np.exp(-tau_v_gr)) * np.exp(-tau_v_gf - tau_d)
        + Sv_d * (1.0 - np.exp(-tau_d)) * np.exp(-tau_v_gf)
        + Sv_gf * (1.0 - np.exp(-tau_v_gf))
        - Iv_d
    )
    return Iv


def doppler_f2v(f: ArrayLike, f0: float, definition: str = "radio"):
    """
    Convert frequency to Doppler velocity.

    Parameters
    ----------
    f : array-like
        Frequency.
    f0 : float
        Rest frequency.
    definition : str
        Doppler definition. Currently only the radio convention is used.
    """
    return (f0 - f) / f0 * clight


def doppler_df2dv(df: ArrayLike, f0: float, definition: str = "radio"):
    """
    Convert a frequency interval to a Doppler velocity interval.

    Parameters
    ----------
    df : array-like
        Frequency interval.
    f0 : float
        Rest frequency.
    definition : str
        Doppler definition. Currently only the radio convention is used.
    """
    return -df * clight / f0


def doppler_v2f(v: ArrayLike, f0: float, definition: str = "radio"):
    """
    Convert Doppler velocity to frequency.

    Parameters
    ----------
    v : array-like
        Doppler velocity.
    f0 : float
        Rest frequency.
    definition : str
        Doppler definition. Currently only the radio convention is used.
    """
    return f0 - v * f0 / clight


def doppler_dv2df(dv: ArrayLike, f0: float, definition: str = "radio"):
    """
    Convert a Doppler velocity interval to a frequency interval.

    Parameters
    ----------
    dv : array-like
        Doppler velocity interval.
    f0 : float
        Rest frequency.
    definition : str
        Doppler definition. Currently only the radio convention is used.
    """
    return -dv * f0 / clight

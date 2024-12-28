# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units
import dataclasses
from dataclasses import dataclass
import time

from .funcs import beam_convolution, gaussian2d, glnprof_conv
from .grid import Nested3DGrid, Nested2DGrid, Nested1DGrid, SubGrid2D
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


class MultiLayerDisk(object):
    '''
    A disk model with Two Thick Layers (TTL) with a thin dust layer.

    '''

    def __init__(self, x, y, z, v,
        xlim: list | None = None, ylim: list | None = None, zlim: list | None = None,
        nsub: list | None = None, reslim: float = 10,
        adoptive_zaxis: bool = True, cosi_lim: float = 0.5, 
        beam: list | None = None, line: str | None = None, iline: int | None = None,
        Tmin: float = 1., Tmax: float = 2000., nTex: int = 4096,
        Td0: float = 400., qd: float = 0.5, log_tau_dc: float = 0., 
        rc_d: float = 100., gamma_d: float = 1., 
        Tg0: float = 400., qg: float = 0.5, log_N_gc: float = 0., 
        rc_g: float = 100., gamma_g: float = 1., 
        z0: float = 0., pz: float = 1.25, h0: float = 0., ph: float = 0., 
        inc: float = 0., pa: float = 0., ms: float = 1., vsys: float = 0, 
        dx0: float = 0., dy0: float = 0., r0: float = 1., dv: float = 0., pdv: float = 0.25):
        '''
        Set up model grid and initialize model.

        Parameters
        ----------
        x, y, z (3D numpy ndarrays): Three dimensional coordinates aligned plane of sky (au).
        '''
        super(MultiLayerDisk, self).__init__()

        # grid
        self.nx, self.ny, self.nz = len(x), len(y), len(z)
        self.grid = Nested3DGrid(x, y, z, xlim, ylim, zlim, nsub, reslim) # Plane of sky coordinates
        self.grid2D = Nested2DGrid(x, y, xlim, ylim, nsub, reslim)
        # Plane of sky coordinates
        self.xs = self.grid.xnest
        self.ys = self.grid.ynest
        self.zs = self.grid.znest
        # disk-local coordinates
        self.xps = [None] * self.grid.nlevels
        self.yps = [None] * self.grid.nlevels
        self.zps = [None] * self.grid.nlevels
        self.Rs = [None] * self.grid.nlevels # R in cylindarical coordinates
        self.ts = [None] * self.grid.nlevels # theta
        # dust layer
        self.Rmid = [None] * self.grid.nlevels

        # velocity
        self.nv = len(v)
        self.delv = np.mean(v[1:] - v[:-1]) # km / s
        self.v = v
        self.ve = np.hstack([v - self.delv * 0.5, v[-1] + 0.5 * self.delv])

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

        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd  = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0 # gas temperature
        self.qg = qg
        self.log_N_gc = log_N_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
        self.z0 = z0
        self.pz = pz
        # gas layer width
        self.h0 = h0
        self.ph = ph
        # geometry & velocity
        self.inc = inc
        self.pa = pa
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.dv = dv
        self.pdv = pdv

        self.param_keys = ['Td0', 'qd', 'log_tau_dc', 'rc_d', 'gamma_d', 'Tg0', 'qg',
        'log_N_gc', 'rc_g', 'gamma_g', 'z0', 'pz', 'h0', 'ph', 'inc', 'pa', 'ms', 'vsys',
        'dx0', 'dy0', 'r0', 'dv', 'pdv']

        # angle in radians
        self._pa_rad = np.radians(self.pa)
        self._inc_rad = np.radians(self.inc)


        # disk-plane coordinates
        self.deproject_grid(adoptive_zaxis = adoptive_zaxis, cosi_lim = cosi_lim)


        # sub parameters
        self._pa_rad = np.radians(self.pa)
        self._inc_rad = np.radians(self.inc)
        self._fz = lambda r, z0, r0, pz: z0*(r/r0)**pz

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
        self.zoffset = [None] * self.grid.nlevels
        for l in range(self.grid.nlevels):
            xp = self.xs[l]
            yp = self.ys[l]
            zp = self.zs[l]
            # rotate by PA
            x, y = rot2d(xp - self.dx0, yp - self.dy0, self._pa_rad - 0.5 * np.pi)
            # rot = - (- (pa - 90.)); two minuses are for coordinate rotation and definition of pa
            # adoptive z axis
            if adoptive_zaxis & (np.abs(np.cos(self._inc_rad)) > cosi_lim):
                # center origin of z axis in the disk midplane
                zoffset = - np.tan(self._inc_rad) * y # zp_mid(xp, yp)
                self.zoffset[l] = zoffset
                _zp = zp + zoffset # shift z center
                x, y, z = xrot(x, y, _zp, self._inc_rad) # rot = - (-inc)
            else:
                x, y, z = xrot(x, y, zp, self._inc_rad) # rot = - (-inc)
                self.zoffset[l] = np.zeros(x.size)

            self.xps[l] = x
            self.yps[l] = y
            self.zps[l] = z


            # cylindarical coordinates
            self.Rs[l] = np.sqrt(x * x + y * y) # radius
            self.ts[l] = np.arctan2(y, x) # azimuthal angle (rad)

            # for dust layer
            x, y = rot2d(self.grid2D.xnest[l] - self.dx0, 
                self.grid2D.ynest[l] - self.dy0, self._pa_rad - 0.5 * np.pi) # in 2D
            y /= np.cos(self._inc_rad)
            self.Rmid[l] = np.sqrt(x * x + y * y) # radius
        self.adoptive_zaxis = adoptive_zaxis


    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_N_gc = 0., rc_g = 100., gamma_g = 1., 
        z0 = 0., pz = 1.25, h0 = 0., ph = 0., inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0 = 0., dy0 = 0., r0 = 1., dv = 0., pdv = 0.25):
        '''

        Parameters
        ----------
         Td0 (float): Temperature of dust layer at r0 (K).
         qd: (float): Power-law index for dust temperature profile.
         log_tau_dc (float): Logarithm of optical depth of dust layer at rc.
         rc_d (float): Charactaristic radius of dust layer (au).
         gamma_d (float): Power-law index gamma of viscous disk for dust layer.
         Tg0 (float): Temperature of gas layers at r0 (K).
         qg (float): Power-law index for gas temperature profile.
         log_N_gc (float): Logarithm of number surface density of gas layers at rc.
         rc_g (float): Charactaristic radius of gas layers (au).
         gamma_g (float): Power-law index gamma of viscous disk for gas layers.
         z0 (float): Height of gas layers at r0 (au).
         pz (float): Power-law index setting flaring of gas layers
         h0 (float): Thickness of gas layers at r0, which is defined in a form of
                     pressure scale height (au).
         ph (float): Power-law index setting flaring of thickness of gas layers
         inc (float): Inclination angle of the disk, where 90 corresponds edge-on view (deg).
                      When pa = 0, inc = 0--90 deg corresponds to a configuration where
                      near side of disk comes to south(?? needs to check), and far side of disk comes to north (?? needs to check).
         pa (float): Position angle of disk, defined as angle measured from north to west up to disk major axis (deg)
         ms (float): Stellar mass (Msun)
         vsys (float): Systemic velocity (km/s)
         dx0, dy0 (float): Offsets of geometric center from given origin of given plane-of-sky coordinates (au).
         r0 (float): Reference radius (au). Default value is 1 au.
         dv (float): Linewidth at r0, defined as width of Doppler line broadening.
                     It can be either total linewidth including any types of intrinsic broadening
                     or non-thermal line broadening, whihc will be summed to thermal line broadening.
         pdv (float): Power-law index setting radial distribution of linewidth.
        '''
        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd  = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0 # gas temperature
        self.qg = qg
        self.log_N_gc = log_N_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
        self.z0 = z0
        self.pz = pz
        # gas layer width
        self.h0 = h0
        self.ph = ph
        # geometry & velocity
        self.inc = inc
        self.pa = pa
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.dv = dv
        self.pdv = pdv

        # angle in radians
        self._pa_rad = np.radians(self.pa)
        self._inc_rad = np.radians(self.inc)


    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def copy(self):
        return self.__copy__()


    def check_params(self):
        print ({'Td0': self.Td0, 'qd': self.qd, 'log_tau_dc': self.log_tau_dc, 
            'rc_d': self.rc_d, 'gamma_d': self.gamma_d, 'Tg0': self.Tg0, 'qg': self.qg, 
             'log_N_gc': self.log_N_gc, 'rc_g': self.rc_g, 'gamma_g': self.gamma_g,
             'z0': self.z0, 'pz': self.pz, 'h0': self.h0, 'ph': self.ph, 'inc': self.inc, 
             'pa': self.pa, 'ms': self.ms, 'vsys': self.vsys, 'dx0': self.dx0, 'dy0': self.dy0, 
             'r0': self.r0, 'dv': self.dv, 'pdv': self.pdv})


    def get_Tt(self, R, T0, q0, tau_c, rc, gamma, rin = 0.1):
        # calculate T(R) & tau(R)
        # temperature
        T = T0 * (R / self.r0)**(-q0)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1.] = 1. # safty net

        # tau
        tau = ssdisk(R, tau_c, rc, gamma, beta = None)
        tau[np.isnan(tau)] = 0.  # to prevent computational errors
        tau[tau < 0.] = 0. # safty net

        T[R < rin] = 1.
        tau[R < rin] = 0.

        return T, tau


    def puff_up_layer(self, sig, z, z0, H):
        return sig * np.exp( - (z - z0)**2. / (2.*H*H)) / np.sqrt(2. * np.pi) / H


    def build_gas_layer(self, R, theta, z, rin = 0.1, dv_mode = 'total'):
        # line of sight velocity
        # take z-axis as the line of sight
        # take x-axis to be major axis of the disk
        vlos = vkep(R * auTOcm, self.ms * Msun, z * auTOcm) \
        * np.cos(theta) * np.sin(self._inc_rad) * 1.e-5 # cm/s --> km/s
        vlos[R < rin] = 0.

        # temperature and tau
        T_g, N_g = self.get_Tt(R, self.Tg0, self.qg,
            10.**self.log_N_gc, self.rc_g, self.gamma_g, rin = rin)

        # puff up layers
        # layer height
        zl = self._fz(R, self.z0, self.r0, self.pz) # height
        h_out = self.h0 * (R / self.r0)**(self.ph)
        h_in = h_out
        #h_in = zl * 0.2 # so that z0 is 5 sigma

        # gas number density
        n_gf = np.zeros(R.shape)
        n_gr = np.zeros(R.shape)


        # check which is fore or rear side
        side = np.sign(np.cos(self._inc_rad)) # cos(-i) = cos(i)

        # height of fore/rear layers
        z0f = zl * side
        z0r = - zl * side

        if side > 0.:
            # positive z is fore side
            zout_f = np.where(z - z0f > 0.) # outer side
            zin_f = np.where( (z <= z0f) * (z > 0.)) # inner side
            # negative z is rear side
            zout_r = np.where(z - z0r < 0.) # outer side
            zin_r = np.where( (z >= z0r) * (z < 0.)) # inner side
        else:
            # positive z is rear side
            zout_r = np.where(z - z0r > 0.) # outer side
            zin_r = np.where((z <= z0r) * (z > 0.)) # inner side
            # negative z is fore side
            zout_f = np.where(z - z0f < 0.) # outer side
            zin_f = np.where( (z >= z0f) * (z < 0.)) # inner side


        '''new
        # height of upper/lower layers
        z0u = zl # upper (positive-z) side
        z0l = - zl # lower (negative-z) side
        zout_f = np.where(z - z0u > 0.) # outer side
        zin_f = np.where( (z <= z0u) * (z > 0.)) # inner side
        # negative z is rear side
        zout_r = np.where(z - z0l < 0.) # outer side
        zin_r = np.where( (z >= z0l) * (z < 0.)) # inner side
        z0f = z0u
        z0r = z0l
        '''

        # fore layer
        n_gf[zout_f] = self.puff_up_layer(N_g[zout_f], z[zout_f], z0f[zout_f], h_out[zout_f]) / auTOcm # cm^-3
        n_gf[zin_f] = self.puff_up_layer(N_g[zin_f], z[zin_f], z0f[zin_f], h_in[zin_f]) / auTOcm # cm^-3

        # rear layer
        n_gr[zout_r] = self.puff_up_layer(N_g[zout_r], z[zout_r], z0r[zout_r], h_out[zout_r]) / auTOcm # cm^-3
        n_gr[zin_r] = self.puff_up_layer(N_g[zin_r], z[zin_r], z0r[zin_r], h_in[zin_r]) / auTOcm # cm^-3

        #n_gf = n_gf.clip(1.e-30, None)
        #n_gr = n_gr.clip(1.e-30, None)
        n_gf[n_gf <= 1.e-20] == 0.
        n_gr[n_gr <= 1.e-20] == 0.

        # line width
        if dv_mode == 'thermal':
            if self.mmol is not None:
                vth = np.sqrt(2. * kb * T_g / self.mmol / mH) * 1.e-5 # km/s
                vnth = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                print('ERROR\tbuild_gas_layer: line should be specified to use')
                print("ERROR\tbuild_gas_layer:  dv_mode == 'thermal'.")
                return 0
        elif dv_mode == 'total':
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv
        else:
            print('ERROR\tbuild_gas_layer: dv_mode must be thermal or total.')
            print("ERROR\tbuild_gas_layer: Ignore the input and assume dv is the total line width.")
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv

        return T_g, vlos, n_gf, n_gr, dv


    def build_dust_layer(self, R, rin = 0.1):
        T, tau = self.get_Tt(R, self.Td0, self.qd, 
            10.**self.log_tau_dc, self.rc_d, self.gamma_d, rin = rin)
        return T, tau


    def build(self, rin = 0.1, dv_mode = 'total',
        collapse = False):
        # for each nested level
        T_g = [None] * self.grid.nlevels
        T_d = [None] * self.grid.nlevels
        n_gf = [None] * self.grid.nlevels
        n_gr = [None] * self.grid.nlevels
        tau_d = [None] * self.grid.nlevels
        vlos = [None] * self.grid.nlevels
        dv = [None] * self.grid.nlevels

        if any([i is None for i in self.xps]) | any([i is None for i in self.yps])\
         | any([i is None for i in self.zps]):
            self.deproject_grid()

        for l in range(self.grid.nlevels):
            # get temperature and volume tau
            _T_g, _vlos, _n_gf, _n_gr, _dv = \
            self.build_gas_layer(self.Rs[l].copy(), 
                self.ts[l].copy(), self.zps[l].copy(), rin = rin, dv_mode = dv_mode)
            _T_d, _tau_d = self.build_dust_layer(self.Rmid[l].copy(), rin = rin)
            _vlos += self.vsys
            T_g[l] = _T_g
            vlos[l] = _vlos
            n_gf[l] = _n_gf
            n_gr[l] = _n_gr
            T_d[l] = _T_d
            tau_d[l] = _tau_d
            dv[l] = _dv

        if collapse:
            T_g = self.grid.collapse(T_g)
            vlos = self.grid.collapse(vlos)
            n_gf = self.grid.collapse(n_gf)
            n_gr = self.grid.collapse(n_gr)
            T_d = self.grid2D.collapse(T_d)
            tau_d = self.grid2D.collapse(tau_d)
            if (self.dv > 0.) | (dv_mode == 'thermal'):
                dv = self.grid.collapse(dv)

        return T_g, vlos, n_gf, n_gr, T_d, tau_d, dv


    def build_cube(self, Tcmb = 2.73, f0 = 230., 
        dist = 140., dv_mode = 'total', contsub = True,
        return_Ttau = False, rin = 0.1):
        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build(rin = rin, dv_mode = dv_mode)

        # dust
        T_d = self.grid2D.collapse(T_d)
        tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature 
        #  of each gas layer at every velocity channel.
        if (self.dv > 0.) | (dv_mode == 'thermal'):
            _Tv_g = [[] for _ in range(self.nv)]  # x,y,z,v
            _nv_gf = [[] for _ in range(self.nv)]
            _nv_gr = [[] for _ in range(self.nv)]

            #start = time.time()
            for l in range(self.grid.nlevels):
                nx, ny, nz = self.grid.ngrids[l]

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
                #end = time.time()
                #print('takes %.2f'%(end-start))
            #end = time.time()
            #print('takes %.2f'%(end-start))

            Nv_gf = np.zeros((self.nv, self.nx, self.ny))
            Tv_gf = np.zeros((self.nv, self.nx, self.ny))
            Nv_gr = np.zeros((self.nv, self.nx, self.ny))
            Tv_gr = np.zeros((self.nv, self.nx, self.ny))
            for i in range(self.nv):
                Tv_g = self.grid.collapse(_Tv_g[i])
                nv_gf = self.grid.collapse(_nv_gf[i])
                nv_gr = self.grid.collapse(_nv_gr[i])

                # fore side
                _Nv_gf = np.nansum(nv_gf * self.grid.dz * auTOcm, axis = 2)
                #print('max Nv_gf: %.3e'%(np.nanmax(_Nv_gf)))
                Nv_gf[i,:,:] = _Nv_gf
                where_nonzero = _Nv_gf > 0.
                Tv_gf[i,:,:][where_nonzero] = np.nansum(
                    Tv_g * nv_gf * self.grid.dz * auTOcm, axis = 2
                    )[where_nonzero] / _Nv_gf[where_nonzero]
                # rear side
                _Nv_gr = np.nansum(nv_gr * self.grid.dz * auTOcm, axis = 2)
                Nv_gr[i,:,:] = _Nv_gr
                where_nonzero = _Nv_gr > 0.
                Tv_gr[i,:,:][where_nonzero] = np.nansum(
                    Tv_g * nv_gr * self.grid.dz * auTOcm, axis = 2
                    )[where_nonzero] / _Nv_gr[where_nonzero]

            Nv_gf = np.transpose(Nv_gf, (0,2,1)) # v, y, x
            Tv_gf = np.transpose(Tv_gf, (0,2,1)) # v, y, x
            Nv_gr = np.transpose(Nv_gr, (0,2,1)) # v, y, x
            Tv_gr = np.transpose(Tv_gr, (0,2,1)) # v, y, x
            '''
            #print(self.grid.ngrids)

            Tv_gf = np.zeros((self.nv, self.nx, self.ny))
            print(self.grid.ngrids)
            for i in range(1):
                Tv_g = fast_3d_collapse(_Tv_g[i], self.grid.nlevels, 
                        self.grid.ngrids, self.grid.nsub, 
                        self.grid.xinest, self.grid.yinest, self.grid.zinest)

            Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = np.transpose(
                transfer.nested_Tnv_to_cube(
                    (self.grid.nlevels, 
                        self.grid.ngrids, self.grid.nsub, 
                        self.grid.xinest, self.grid.yinest, self.grid.zinest),
                    _Tv_g, _nv_gf, _nv_gr, 
                    self.nx, self.ny, self.nz, self.nv,
                    self.grid.dz * auTOcm,
                    self.freq, self.Aul, self.Eu, self.gu, self.Qgrid
                    ),
                (0,1,3,2))
            '''
        else:
            Tv_gf, Tv_gr, Nv_gf, Nv_gr = np.transpose(
            Tt_to_cube(T_g, n_gf, n_gr, vlos, self.ve, self.grid.dz * auTOcm,),
            (0,1,3,2,))

        Tv_gf = Tv_gf.clip(1., None) # safety net to avoid zero division
        Tv_gr = Tv_gr.clip(1., None)


        # density to tau
        #print('Tv_gf max, q: %13.2e, %.2f'%(np.nanmax(Tv_gf), self.qg))
        #print('Nv_gf max: %13.2e'%(np.nanmax(Nv_gf)))
        if (self.line is not None) * (self.iline is not None):
            tau_v_gf = self.mol.get_tau(self.line, self.iline, 
                Nv_gf, Tv_gf, delv = None, grid_approx = True)
            tau_v_gr = self.mol.get_tau(self.line, self.iline, 
                Nv_gr, Tv_gr, delv = None, grid_approx = True)
        else:
            # ignore temperature effect on conversion from column density to tau
            tau_v_gf = Nv_gf
            tau_v_gr = Nv_gr
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

        if contsub == False:
            Iv_d = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_d))
            Iv_d = np.tile(Iv_d, (self.nv,1,1,))
            Iv += Iv_d # add continuum back

        # Convolve beam if given
        #print('I_v_nobeam max: %13.2e'%(np.nanmax(Iv)))
        if self.beam is not None:
            Iv = beam_convolution(self.grid2D.xx.copy(), self.grid2D.yy.copy(), Iv, 
                self.beam, self.gaussbeam)

        #print('I_v max: %13.2e'%(np.nanmax(Iv)))
        return Iv


    def show_model_sideview(self):
        #x = self.grid.collapse(self.xs)
        #y = self.grid.collapse(self.ys)
        #z = self.grid.collapse(self.zs)
        zoffset = self.grid.collapse(self.zoffset)
        x, y, z = self.grid.xx, self.grid.yy, self.grid.zz
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        y = y[nx//2, :, :]
        #_x, _y = rot2d(x[nx//2, :, :] - self.dx0, y - self.dy0, self._pa_rad - 0.5 * np.pi)
        if self.adoptive_zaxis:
            #z = z[nx//2, :, :] - np.tan(self._inc_rad) * _y # adoptive z
            z = z[nx//2, :, :] + zoffset[nx//2, :, :]
        else:
            z = z[nx//2, :, :]

        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build()

        n_g = n_gf + n_gr

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.pcolormesh(z, y, n_g[nx//2, :, :])
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(zmin, zmax) # z axis
        plt.show()


    def show_model_diskview(self):
        #x, y, z = self.xps[0], self.yps[0], self.zps[0]
        x = self.grid.collapse(self.xps)
        y = self.grid.collapse(self.yps)
        z = self.grid.collapse(self.zps)
        R = self.grid.collapse(self.Rs)
        t = self.grid.collapse(self.ts)
        nx, ny, nz = x.shape
        x, y, z = self.grid.xx, self.grid.yy, self.grid.zz
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build()

        n_g = n_gf + n_gr

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.pcolormesh(x[:, :, nz//2], y[:, :, nz//2], n_g[:, :, nz//2])
        ax2.pcolormesh(R[:, ny//2, :], z[:, ny//2, :], n_g[:, ny//2, :])
        #ymin, ymax = np.nanmin(y), np.nanmax(y)
        #zmin, zmax = np.nanmin(z), np.nanmax(z)
        #ax.set_ylim(ymin, ymax)
        #ax.set_xlim(zmax, zmin) # z axis
        plt.show()


    def show_model_projectedview(self):
        #x, y, z = self.xps[0], self.yps[0], self.zps[0]
        x = self.grid.collapse(self.xps)
        y = self.grid.collapse(self.yps)
        z = self.grid.collapse(self.zps)
        nx, ny, nz = x.shape
        x, y, z = self.grid.xx, self.grid.yy, self.grid.zz
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        x = x[:, :, nz//2,]
        y = y[:, :, nz//2,]

        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build()

        n_g = n_gf + n_gr

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.pcolormesh(x, y, np.nansum(n_g, axis = 2))
        ax.pcolormesh(x, y, np.nansum((vlos - self.vsys) * n_gf, axis = 2) / np.nansum(n_gf, axis = 2),
            cmap = 'RdBu_r', vmin = -5, vmax = 5.)
        #ax.contour(x, y, np.nansum(tau_rho_gf, axis = 2), colors = 'red')
        #ax.contour(x, y, np.nansum(tau_rho_gr, axis = 2), colors = 'blue')
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        plt.show()



@dataclass(slots=True)
class ThreeLayerDisk:

    # params for dust layer
    Td0: float = 400.
    qd: float = 0.5
    log_tau_dc: float = 0. # tau of dust at rc
    rc_d: float = 1.
    gamma_d: float = 1.
    # params for gas layer
    Tg0: float = 400. # Gas temperature at r0
    qg: float = 0.5 # power-law index of temperature distribution
    log_tau_gc: float = 0.
    rc_g: float = 1.
    gamma_g: float = 1.
    z0: float = 0.
    hp: float = 1.25
    # geometry & velocity
    inc: float = 0.
    pa: float = 0.
    ms: float = 1.
    vsys: float = 0.
    # positional offsets
    dx0: float = 0.
    dy0: float = 0.
    # reference radius
    r0: float = 1.
    # line width
    delv: float = 0.

    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_tau_gc = 0., rc_g = 100., gamma_g = 1., 
        z0 = 0., hp = 1.25, inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., delv = 0.):
        '''

        Parameters
        ----------
         Td0
         qd
         Tg0 (float): K
         qg (float):
         z0 (float): au
         hp (float):
         r0 (float): au
         tau_dc (float):
         rc_d (float): au
         gamma_d (float):
         tau_gc (float):
         rc_g (float): au
         gamma_g (float):
         inc (float): deg
         pa (float): deg
         ms (float): Msun
         vsys (float): km/s
        '''
        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd  = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0 # gas temperature
        self.qg = qg
        self.log_tau_gc = log_tau_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
        self.z0 = z0
        self.hp = hp
        # geometry & velocity
        self.inc = inc
        self.pa = pa
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, r0, hp: 2. * y * 0.5 / np.sqrt(x*x + y*y) \
        / r0 * z0 * hp * (np.sqrt(x*x + y*y)/r0)*(hp - 1.)

        # calculate temperature (T), velocity (v) and tau (t)
        def get_Tvt(xx_sky, yy_sky, 
            _fz, zargs, inc, pa, ms, T0, q, r0, tau_c, rc, gamma, _dfz=None):
            # deprojection
            depr = sky_to_local(xx_sky, yy_sky, 
                inc, pa + 0.5 * np.pi, _fz, 
                zargs, _dfz, zarg_lims = [[-0.3, 0.3], [0.1, 100.1], [0., 2.]])
            if type(depr) == int:
                T = np.full(xx_sky.shape, 1.) # 1 instead of zero to prevent numerical errors
                vlos  = np.zeros(xx_sky.shape)
                tau = np.zeros(xx_sky.shape)
                return T, vlos, tau
            else:
                xx, yy = depr

            # local coordinates
            rr = np.sqrt(xx * xx + yy * yy) # radius
            phph = np.arctan2(yy, xx) # azimuthal angle (rad)
            zz = _fz(rr, *zargs) # height
            # prevent r=0
            rr[rr == 0.] = np.nan

            # quantities
            # temperature
            T = T0 * (rr / r0)**(-q)
            T[np.isnan(T)] = 1. # to prevent computational errors
            T[T <= 1.] = 1. # safty net

            # line of sight velocity
            # take y-axis as the line of sight
            vlos = vkep(rr * auTOcm, ms * Msun, zz * auTOcm) \
            * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 # cm/s --> km/s

            # tau
            tau = ssdisk(rr, tau_c, rc, gamma, beta = None)
            tau[np.isnan(tau)] = 0.  # to prevent computational errors
            tau[tau < 0.] = 0. # safty net

            T[rr < rin] = 0.
            vlos[rr < rin] = 0.
            tau[rr < rin] = 0.

            return T, vlos, tau


        # front gas layer
        T_gf, vlos_gf, tau_gf = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Tg0, self.qg, self.r0,
            10.**self.log_tau_gc, self.rc_g, self.gamma_g,)

        # dust layer
        T_d, _, tau_d = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [0., self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Td0, self.qd, self.r0,
            10.**self.log_tau_dc, self.rc_d, self.gamma_d,)

        # rear gas layer
        T_gr, vlos_gr, tau_gr = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [-self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Tg0, self.qg, self.r0,
            10.**self.log_tau_gc, self.rc_g, self.gamma_g,)

        return T_gf, T_gr, T_d, vlos_gf, vlos_gr, tau_gf, tau_gr, tau_d


    def build_cube(self, xx, yy, v, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # get quantities
        T_gf, T_gr, T_d, vlos_gf, vlos_gr, tau_gf, tau_gr, tau_d = self.build(xx, yy)
        vlos_gf += self.vsys
        vlos_gr += self.vsys

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # nested velocity grid
        #nstg = Nested1DGrid(v)
        #nstg.nest(5)
        #v_nst = nstg.x_sub
        #ve_nst = nstg.xe_sub
        #nv_nst = nstg.nx_sub

        # making a cube
        # tau_v
        _tau_gf = tocube(tau_gf, vlos_gf, ve)
        _tau_gr = tocube(tau_gr, vlos_gr, ve)
        # line width
        if self.delv > 0.:
            _tau_gf = glnprof_conv(_tau_gf, v, self.delv)
            _tau_gr = glnprof_conv(_tau_gr, v, self.delv)

        # get it back to original grid
        #_tau_gf = nstg.binning_onsubgrid(_tau_gf)
        #_tau_gr = nstg.binning_onsubgrid(_tau_gr)


        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        #_Bv = lambda T, v: Bv(T, v)
        _Bv_cmb = _Bv(Tcmb, f0)
        _Bv_gf  = _Bv(T_gf, f0)
        _Bv_gr  = _Bv(T_gr, f0)
        _Bv_d   = _Bv(T_d, f0)
        Iv = solve_3LRT(_Bv_gf, _Bv_gr, _Bv_d, 
            _tau_gf, _tau_gr, tau_d, _Bv_cmb, nv)
        '''
        I_cube = np.array([
            _Bv_cmb * (np.exp(- _tau_gf[i,:,:] - tau_d - _tau_gr[i,:,:]) - 1.) \
            + _Bv_gr * (1. - np.exp(- _tau_gr[i,:,:])) * np.exp(- _tau_gf[i,:,:] - tau_d) \
            + _Bv_d * (1. - np.exp(- tau_d)) * np.exp(- _tau_gf[i,:,:]) \
            + _Bv_gf * (1. - np.exp(- _tau_gf[i,:,:])) \
            - Idust # contsub
            for i in range(nv)])
        '''

        # Convolve beam if given
        if beam is not None:
            Iv = beam_convolution(xx, yy, Iv, [beam[0] * dist, beam[1] * dist, beam[2]])

        return Iv


    def build_cube_subgrid(self, xx, yy, v, nsub = 2,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        if nsub < 1:
            print('ERROR\tbuild_cube_subgrid: nsub must be >= 2.')
            return 0
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub
        I_cube_sub = self.build_cube(_xx, _yy, v, beam, dist, Tcmb, f0)
        nv = len(v)
        ny, nx = xx.shape
        return subgrid.binning_onsubgrid_layered(I_cube_sub)


    def build_nested_cube(self, xx, yy, v, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        I_cube = self.build_cube(xx, yy, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cube on the nested grid
        I_cube_sub = self.build_cube(xx_sub, yy_sub, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cut off edge
        if beam is None:
            xi0, xi1, yi0, yi1 = nstgrid.xi0, nstgrid.xi1, nstgrid.yi0, nstgrid.yi1
            for i in range(len(v)):
                I_cube[i, yi0:yi1+1, xi0:xi1+1] = \
                nstgrid.binning_onsubgrid(I_cube_sub[i,:,:])
        else:
            xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
                beam[0] * dist * 1.3, beam[0] * dist * 1.3)
            I_cube_sub = I_cube_sub[:, yi:-yi, xi:-xi]
            # replace
            for i in range(len(v)):
                I_cube[i, yi0:-yi0, xi0:-xi0] = \
                nstgrid.binning_onsubgrid(I_cube_sub[i,:,:])

        return I_cube



@dataclass(slots=True)
class SingleLayerDisk:

    # params for the disk
    # whichever gas or dust
    T0: float = 400. # temperature
    q: float = 0.5   # slope of temperature prof
    log_tau_c: float = 0. # tau at rc
    rc: float = 100.
    gamma: float = 1.
    # make it flared if you like
    z0: float = 0.
    hp: float = 1.25
    # geometry
    inc: float = 0.
    pa: float = 0.
    # stellar mass and vsys matter only for line case
    ms: float = 1.
    vsys: float = 0.
    # positional offsets
    dx0: float = 0.
    dy0: float = 0.
    # reference radius
    r0: float = 1.
    # line width
    delv: float = 0.


    def set_params(self, 
        T0 = 400., q = 0.5, log_tau_c = 0., rc = 100., gamma = 1., 
        z0 = 0., hp = 1.25, inc = 0., pa = 0., ms = 1., vsys = 0., 
        dx0=0., dy0=0., r0 = 1., delv = 0.):
        '''

        Parameters
        ----------
         T0
         q
         z0 (float): au
         hp (float):
         r0 (float): au
         log_tau_c (float):
         rc (float): au
         gamma (float):
         inc (float): deg
         pa (float): deg
         ms (float): Msun
         vsys (float): km/s
        '''
        # initialize parameters
        # dust layer
        self.T0 = T0
        self.q  = q
        self.log_tau_c = log_tau_c
        self.rc = rc
        self.gamma = gamma
        # height
        self.z0 = z0
        self.hp = hp
        # geometry
        self.inc = inc
        self.pa = pa
        # velocity
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, r0, hp: 2. * y * 0.5 / np.sqrt(x*x + y*y) \
        / r0 * z0 * hp * (np.sqrt(x*x + y*y)/r0)*(hp - 1.)

        # calculate temperature (T), velocity (v) and tau (t)
        def get_Tvt(xx_sky, yy_sky, 
            _fz, zargs, inc, pa, ms, T0, q, r0, tau_c, rc, gamma, _dfz=None):
            # deprojection
            #print('Start deprojection', self.z0, self.hp, self.inc)
            depr = sky_to_local(xx_sky, yy_sky, 
                inc, pa + 0.5 * np.pi, _fz, 
                zargs, _dfz, zarg_lims = [[-0.3, 0.3], [0.1, 100.1], [0., 2.]]) # inc_max = 85.
            #print('Done deprojection')
            if type(depr) == int:
                T = np.full(xx_sky.shape, 1.) # 1 instead of zero to prevent numerical errors
                vlos  = np.zeros(xx_sky.shape)
                tau = np.zeros(xx_sky.shape)
                return T, vlos, tau
            else:
                xx, yy = depr

            # local coordinates
            rr = np.sqrt(xx * xx + yy * yy) # radius
            phph = np.arctan2(yy, xx) # azimuthal angle (rad)
            zz = _fz(rr, *zargs) # height
            # prevent r=0
            rr[rr == 0.] = np.nan

            # quantities
            # temperature
            T = T0 * (rr / r0)**(-q)
            T[np.isnan(T)] = 1. # to prevent computational errors
            T[T <= 1.] = 1. # safty net

            # line of sight velocity
            # take y-axis as the line of sight
            vlos = vkep(rr * auTOcm, ms * Msun, zz * auTOcm) \
            * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 # cm/s --> km/s

            # tau
            tau = ssdisk(rr, tau_c, rc, gamma, beta = None)
            tau[np.isnan(tau)] = 0.  # to prevent computational errors
            tau[tau < 0.] = 0. # safty net

            T[rr < rin] = 0.
            vlos[rr < rin] = 0.
            tau[rr < rin] = 0.

            return T, vlos, tau


        # for a layer
        T, vlos, tau = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.T0, self.q, self.r0,
            10.**self.log_tau_c, self.rc, self.gamma,)

        return T, vlos, tau


    def build_cube(self, xx, yy, v, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,
        return_tau = False):
        # get quantities
        T, vlos, tau = self.build(xx, yy)
        vlos += self.vsys

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # making a cube
        _tau = np.zeros((nv, ny, nx))
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        # calculate tau_v
        for i in range(nv):
            #print('vrange: %.2f-%.2f'%(ve[i],ve[i+1]))
            _tau[i,:,:] = np.where(
                (ve[i] <= vlos) & (vlos < ve[i+1]),
                tau, 0.
                )
        # line width
        if self.delv > 0.:
            _tau = glnprof_conv(_tau, v, self.delv)

        if return_tau:
            return _tau #np.exp(- _tau) #(1. - np.exp(- _tau))

        #_tau[_tau > 0.] = 1000.
        # radiative transfer
        _Bv_bg = np.tile(_Bv(Tcmb, f0), (nv,1,1))
        _Bv_T  = np.tile(_Bv(T, f0), (nv,1,1))
        I_cube = (_Bv_T - _Bv_bg) * (1. - np.exp(- _tau))

        # Convolve beam if given
        if beam is not None:
            I_cube = beam_convolution(xx, yy, I_cube, 
                [beam[0] * dist, beam[1] * dist, beam[2]])

        # return intensity
        return I_cube


    def build_nested_cube(self, xx, yy, v, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230., return_tau = False):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        I_cube = self.build_cube(xx, yy, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0, return_tau = return_tau)

        # cube on the nested grid
        I_cube_sub = self.build_cube(xx_sub, yy_sub, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0, return_tau = return_tau)

        # cut off edge
        if (beam is not None) & (return_tau == False):
            xi, yi, xi0, yi0 = nstgrid.edgecut_indices(beam[0] * dist * 1.5, beam[0] * dist * 1.5)
            I_cube_sub = I_cube_sub[:, yi:-yi, xi:-xi]
            xi1, yi1 = nstgrid.nx - xi0, nstgrid.ny - yi0
        else:
            xi0, yi0 = nstgrid.xi0, nstgrid.yi0
            xi1, yi1 = nstgrid.xi1, nstgrid.yi1
        # replace
        for i in range(len(v)):
            #I_cube[i, yi0:yi1, xi0:xi1] = \
            I_cube[i, yi0:-yi0, xi0:-xi0] = \
            nstgrid.binning_onsubgrid(I_cube_sub[i,:,:])

        return I_cube


    def build_cube_subgrid(self, xx, yy, v, nsub = 2,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        if nsub < 1:
            print('ERROR\tbuild_cube_subgrid: nsub must be >= 2.')
            return 0
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub
        I_cube_sub = self.build_cube(_xx, _yy, v, beam, dist, Tcmb, f0)
        nv = len(v)
        ny, nx = xx.shape
        return subgrid.binning_onsubgrid_layered(I_cube_sub)


    def build_cont(self, xx, yy, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.):
        # get quantities
        T, _, tau = self.build(xx, yy)

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]

        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        Iv = (_Bv(T, f0) - _Bv(Tcmb, f0)) * (1. - np.exp(- tau))

        # Convolve beam if given
        if beam is not None:
            Iv = beam_convolution(xx, yy, Iv, 
                [beam[0] * dist, beam[1] * dist, beam[2]])

        # return intensity
        return Iv


    def build_cont_subgrid(self, xx, yy, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,
        nsub = 2):
        # subgrid
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub

        # cube on the original grid
        Iv = self.build_cont(_xx, _yy,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # return intensity
        return subgrid.binning_onsubgrid_layered(Iv)


    def build_nested_cont(self, xx, yy, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        Iv = self.build_cont(xx, yy,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cube on the nested grid
        Iv_sub = self.build_cont(xx_sub, yy_sub,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cut off edge
        xi, yi, xi0, yi0 = \
        nstgrid.edgecut_indices(beam[0] * dist * 1.3, beam[0] * dist * 1.3)
        Iv_sub = Iv_sub[yi:-yi, xi:-xi]
        # replace
        Iv[yi0:-yi0, xi0:-xi0] = nstgrid.binning_onsubgrid(Iv_sub)

        return Iv



@dataclass(slots=True)
class SSDisk:

    # intensity distribution
    Ic: float = 1. # intensity at rc
    rc: float = 1. # critical radius
    beta: float = 0. # beta
    gamma: float = 0. # gamma
    inc: float = 0.
    pa: float = 0.
    z0: float = 0.
    r0: float = 1.
    hp: float = 0.
    ms: float = 0.
    vsys: float = 0.
    delv: float = 0. # line width

    def set_params(self, Ic = 0, rc = 0, beta = 0, gamma = 0, 
        inc = 0, pa = 0, z0 = 0, r0 = 0, hp = 0, ms = 0, vsys = 0, delv = 0.):
        '''

        Parameters
        ----------
         Ic (float): 
         rc (float): au
         inc (float): deg
         pa (float): deg
         z0 (float): au
         r0 (float): au
         ms (float): Msun
        '''
        # initialize parameters
        self.Ic = Ic
        self.rc = rc
        self.beta = beta
        self.gamma = gamma
        self.inc = inc
        self.pa = pa
        self.z0 = z0
        self.r0 = r0
        self.hp = hp
        self.ms = ms
        self.vsys = vsys
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, h0, hp: 2. * y * 0.5 / np.sqrt(xp*xp + y*y) \
        / r0 * z0 * hp * (np.sqrt(xp*xp + y*y)/r0)*(hp - 1.)
        _zargs = [self.z0, self.r0, self.hp]

        # deprojection
        depr = sky_to_local(xx_sky.ravel(), yy_sky.ravel(), 
        _inc_rad, _pa_rad + 0.5 * np.pi, _fz, _zargs,)
        if type(depr) == int:
            I_int = np.zeros(xx_sky.shape)
            vlos  = np.zeros(xx_sky.shape)
            return I_int, vlos
        else:
            xx, yy = depr
            xx = xx.reshape(xx_sky.shape)
            yy = yy.reshape(yy_sky.shape)

        # local coordinates
        rr = np.sqrt(xx * xx + yy * yy) # radius
        rr[rr < rin] = np.nan
        phph = np.arctan2(yy, xx) # azimuthal angle (rad)
        zz = _fz(rr, *_zargs) # height

        # take y-axis as the line of sight
        vlos = vkep(rr * auTOcm, self.ms * Msun, zz * auTOcm) \
        * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s
        I_int = ssdisk(rr, self.Ic, self.rc, self.gamma, self.beta)

        return I_int, vlos


    def build_cube(self, xx, yy, v, beam = None, dist = 140.):
        I_int, vlos = self.model.build(xx, yy)

        ny, nx = xx.shape
        nv = len(v)
        delv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])
        I_cube = np.zeros((nv, ny, nx))
        # making a cube
        for i in range(nv):
            vindx = np.where((ve[i] <= vlos) & (vlos < ve[i+1]))
            I_cube[i,vindx[0], vindx[1]] = I_int[vindx]

        #print('convolve beam..')
        # Convolve beam if given
        if beam is not None:
            gaussbeam = gaussian2d(xx, yy, 1., 0., 0., 
            beam[1] * dist / 2.35, beam[0] * dist / 2.35, beam[2], peak=True)

            I_cube /= np.abs((xx[0,0] - xx[0,1])*(yy[1,0] - yy[0,0])) # per pixel to per arcsec^2
            I_cube *= np.pi/(4.*np.log(2.)) * beam[0] * beam[1] # per arcsec^2 --> per beam

            # beam convolution
            I_cube = np.where(np.isnan(I_cube), 0., I_cube)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')
        #print('done.')

        # return intensity
        return I_cube



# Radial profiles
def powerlaw_profile(r, p, I0, r0=1.):
    return I0*(r/r0)**(-p)

def mdpowerlaw_profile(r, a0, alpha, p):
    return a0*(1. + alpha*r)**(-p)

def nuker_profile(r, rt, It, alpha, beta, gamma):
    return It*(r/rt)**(-gamma) * (1. + (r/rt)**alpha)**((gamma-beta)/alpha)

def ssdisk(r, Ic, rc, gamma, beta = None):
    beta_p = gamma if beta is None else beta # - beta = - gamma - q
    return Ic * (r/rc)**(- beta_p) * np.exp(-(r/rc)**(2. - gamma))

def gaussian_profile(r, I0, sigr):
    return I0*np.exp(-r**2./(2.*sigr**2))


# Gaussians
# 1D
def gaussian1d(x, amp, mx, sig):
    return amp*np.exp(- 0.5 * (x - mx)**2/(sig**2)) #+ offset

# 2D
def gaussian2d(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
    '''
    Generate normalized 2D Gaussian

    Parameters
    ----------
     x: x value (coordinate)
     y: y value
     A: Amplitude. Not a peak value, but the integrated value.
     mx, my: mean values
     sigx, sigy: standard deviations
     pa: position angle [deg]. Counterclockwise is positive.
    '''
    if pa: # skip if pa == 0.
        x, y = rotate2d(x,y,pa)

    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    return coeff*expx*expy


# 2D rotation
def rotate2d(x, y, angle, deg=True, coords=False):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    array2d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    '''

    # degree --> radian
    if deg:
        angle = np.radians(angle)
    else:
        pass

    if coords:
        angle = -angle
    else:
        pass

    cos = np.cos(angle)
    sin = np.sin(angle)

    xrot = x*cos - y*sin
    yrot = x*sin + y*cos

    return xrot, yrot


def rot2d(x, y, ang):
    '''
    2D rotation. NOT rotation of coordinates. To convert coordinates, give -ang, 
     where ang is required rotation of coordinates.
    '''
    return x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)

def xrot(x, y, z, ang):
    '''
    3D rotation around x axis. NOT rotation of coordinates. To convert coordinates, give -ang, 
     where ang is required rotation of coordinates.
    '''
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


def sky_to_local(x, y, inc, pa, fz, zargs, dfz = None,
    inc_max = 85., zarg_lims = None, method = 'krylov'):
    '''
    Conversion from the plane-of-sky coordinate to the local coordinate.

    x, y (array): Plane-of-sky coordinates.
    inc (float): Inclination angle of the object.
    pa (float): Position angle of the object.

    method: krylov, anderson

    '''
    _inc_max_rad = np.radians(inc_max)
    jac = True if dfz is not None else False
    if (jac == True) & ((np.array(['hybr', 'lm']) != method).all()):
        method = 'hybr'
    # safety net
    if zarg_lims is not None:
        limit_check = np.array([
            (zargs[i] >= zarg_lims[i][0]) & (zargs[i] < zarg_lims[i][1]) 
            for i in range(len(zargs))])
        if limit_check.all():
            pass
        else:
            return 0

    # root function
    def y_solver(y, xp, yp, inc, fz, zargs, dfz = None):
        if dfz is None:
            return y * np.cos(inc) + fz(np.sqrt(xp*xp + y*y), *zargs) * np.sin(inc) - yp
        else:
            _jac = 1. / (np.cos(inc) + dfz(xp, y, *zargs) * np.sin(inc))
            return y * np.cos(inc) + fz(np.sqrt(xp*xp + y*y), *zargs) * np.sin(inc) - yp, np.diag(_jac)


    # rotation
    if pa != 0.:
        x, y = rotate2d(x, y, pa, coords=False, deg=False)

    # check inclination limit
    if inc < 0.:# inc = 0.
        return 0
    if inc > _inc_max_rad:# inc = _inc_max_rad
        return 0

    # deprojection limit
    # only valid for power-law flaring case
    z0, r0, hp = zargs
    #print('z0, hp, inc: %.2f %.2f %.2f'%(z0, hp, inc * 180./np.pi))
    if z0 == 0.:
        #sol = root(y_solver, y, 
        #    args=(x, y, inc, fz, zargs, dfz), method = method,)
        #ydep = sol.x
        ydep = y / np.cos(inc)
    elif hp == 1.:
        _r = 1.
        _z = np.abs(fz(_r, *zargs))
        th_lim = 0.5 * np.pi - np.arctan2(_z, _r) - inc
        if th_lim <= 0.:
            return 0
        else:
            sol = root(y_solver, y, 
                args=(x, y, inc, fz, zargs, dfz), method = method,)
            ydep = sol.x
    else:
        sol = root(y_solver, y, 
                args=(x, y, inc, fz, zargs, dfz), 
                method = method, options={'maxiter': 20})
        if sol.success:
            ydep = sol.x
        else:
            return 0

        '''
        y_ypmin = (1. / np.tan(inc) * r0**hp / np.abs(z0) / hp)**( 1. / (hp - 1.))
        yp_lim = y_ypmin * np.cos(inc) + fz(np.sqrt(y_ypmin * y_ypmin), *zargs) \
        * np.sin(inc)
        print(yp_lim)

        # deprojection
        ydep = np.empty(y.shape)
        cond = y > - yp_lim if z0 > 0. else y < yp_lim
        if np.count_nonzero(cond) > 0:
            #print('gonna pass')
            sol = root(y_solver, y[np.where(cond)], 
                args=(x[np.where(cond)], y[np.where(cond)], inc, fz, zargs, dfz), 
                method = method, options={'maxiter': 20})
            if sol.success:
                ydep[np.where(cond)] = sol.x
            else:
                return 0
            #print('passed')
            #print(sol.nit, sol.success)
            #ydep[np.where(cond)] = sol.x
        if np.count_nonzero(~cond) > 0:
            ydep[np.where(~cond)] = np.nan
        '''

    return x, ydep


def vkep(r, ms, z = 0.):
    return np.sqrt(Ggrav * ms * r * r / (r*r + z*z)**(1.5))


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

def doppler_f2v(f, f0, definition = 'radio'):
    return (f0 - f) / f0 * clight

def doppler_df2dv(df, f0, definition = 'radio'):
    return - df * clight / f0

def doppler_v2f(v, f0, definition = 'radio'):
    return f0 - v * f0 / clight

def doppler_dv2df(dv, f0, definition = 'radio'):
    return - dv * f0 / clight
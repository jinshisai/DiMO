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
from .grid import Nested3DGrid, Nested2DGrid, Nested1DGrid, Nested3DObsGrid
from .libcube.linecube import solve_MLRT, Tndv_to_cube, Tt_to_cube
from .molecule import Molecule
from .libcube import spectra, transfer, linecube

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



@dataclass(slots=True)
class TwoComponentDisk:
    '''
    Two-component Disk Model which consists of a thick gas layer and a thin dust layer.
    The thickness of gas is taken into account in the radiative transfer calculation using
    the channel-based three-layer approximation.
    '''

    # params for dust layer
    Td0: float = 400.
    qd: float = 0.5
    log_tau_dc: float = 0. # tau of dust at rc
    rc_d: float = 1.
    gamma_d: float = 1.
    # params for gas layer
    Tg0: float = 400. # Gas temperature at r0
    qg: float = 0.5 # power-law index of temperature distribution
    log_N_gc: float = 0.
    rc_g: float = 1.
    gamma_g: float = 1.
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
    dv: float = 0.
    pdv: float = 0.25


    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad)) # cos(-i) = cos(i)


    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_N_gc = 0., rc_g = 100., gamma_g = 1., 
        inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., dv = 0., pdv = 0.25):
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
        self.log_N_gc = log_N_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
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


        # hidden parameters
        self._inc_rad = np.radians(inc)
        self._pa_rad = np.radians(pa)
        self._side = np.sign(np.cos(self._inc_rad)) # cos(-i) = cos(i)


    @classmethod
    def get_paramkeys(cls):
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != '_']
        return paramkeys


    @classmethod
    def get_annotations(cls):
        d = {}
        for c in cls.mro():
            try:
                d.update(**c.__annotations__)
            except AttributeError:
                # object, at least, has no __annotations__ attribute.
                pass
        return d


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def gas_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Tg0 * (R / self.r0)**(-self.qg)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def dust_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Td0 * (R / self.r0)**(-self.qd)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def gas_speed(self, R, z):
        return vkep(R * auTOcm, self.ms * Msun, z * auTOcm) * 1.e-5 # cm/s --> km/s


    def _gas_velocity_old(self, R, phi, z):
        return vkep(R * auTOcm, self.ms * Msun, z * auTOcm) *\
         np.cos(phi) * np.sin(self._inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s


    def gas_velocity(self, R, phi, z, T, pterm = True, mu = 2.34):
        '''
        Calculate line-of-sight velocity, i.e., 
        projection of the rotational velocity.

        Parameters
        ----------
        R (array): Cylindarical radius (au)
        phi (array): Azimuthal angle (rad)
        z (array): Height (au)
        T (array): Gas temperature (K)
        pterm (bool): If include the pressure gradient term or not.
        '''
        #vphi = vrot(R * auTOcm, self.ms * Msun, rho, T,
        #    z = z * auTOcm, pterm = pterm, mu = mu,)
        vphi = vrot_ssdisk(R * auTOcm, self.ms * Msun, T, 
            self.rc_g * auTOcm, self.gamma_g, self.qg, 
            z = z * auTOcm, pterm = pterm, mu = mu)
        return vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s


    def speed_to_velocity(self, gs, phi,):
        return gs * np.cos(phi) * np.sin(self._inc_rad)


    def gas_density(self, R, z, mu = 2.34):
        # surface density
        N_g = ssdisk(R, 10.**self.log_N_gc, self.rc_g, self.gamma_g, beta = None)

        # scale height
        _cs = np.sqrt(kb * self.Tg0 / mu / mH) # scale height at r0
        _Omega = np.sqrt(Ggrav * self.ms * Msun / (self.r0*auTOcm)**3. )
        h0 = _cs/_Omega / auTOcm # in au
        ph = - 0.5 * self.qg + 1.5
        h = h0 * (R / self.r0)**(ph)
        #print(h0, ph,)
        #print(np.nanmin(h), np.nanmax(h))

        # density
        ng = self.puff_up_layer(N_g, R, z, h) / auTOcm # cm^-3
        return ng


    def _puff_up_layer_old(self, sig, z, z0, H):
        return sig * np.exp( - (z - z0)**2. / (2.*H*H)) / np.sqrt(2. * np.pi) / H


    def puff_up_layer(self, sig, R, z, H):
        '''
        Without approximation of z<<R.
        '''
        rho0 = sig / np.sqrt(2. * np.pi) / H
        exp = np.exp(R**2. / H**2. * ((1. + z**2/R**2)**(-0.5) - 1.))
        return rho0 * exp


    def linewidth(self, R, Tg = None, dv_mode = 'thermal', mmol = 2.34):
        # line width
        if dv_mode == 'thermal':
            vth = np.sqrt(2. * kb * Tg / mmol / mH) * 1.e-5 # km/s
            if self.dv > 0.:
                vnth = self.dv * (R / self.r0)**(- self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == 'total':
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv
        else:
            print('ERROR\tbuild_gas_layer: dv_mode must be thermal or total.')
            print("ERROR\tbuild_gas_layer: Ignore the input and assume dv is the total line width.")
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv

        return dv


    def dust_density(self, R,):
        return ssdisk(R, 10.**self.log_tau_dc, self.rc_d, self.gamma_d, beta = None)


    def build(self, R, phi, z, Rmid, 
        dv_mode = 'total', collapse = False, 
        mmol = 30., mu = 2.34, pterm = True,):
        '''
        deproject_grid frist.
        '''
        # gas
        T_g = self.gas_temperature(R)
        n_g = self.gas_density(R, z)
        #vlos = self.gas_velocity(R, phi, z)
        vlos = self.gas_velocity(R, phi, z, T_g, 
            pterm = pterm, mu = mu)
        dv = self.linewidth(R, dv_mode = dv_mode, Tg = T_g, mmol = mmol)

        # dust
        T_d = self.dust_temperature(Rmid)
        tau_d = self.dust_density(Rmid)

        return T_g, n_g, vlos, dv, T_d, tau_d


    def build_cube(self, Tcmb = 2.73, f0 = 230., 
        dist = 140., dv_mode = 'total', pterm = True, 
        contsub = True,
        return_Ttau = False, rin = 0.1):
        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build(rin = rin, dv_mode = dv_mode, pterm = pterm)

        # dust
        T_d = self.grid2D.collapse(T_d)
        tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature 
        #  of each gas layer at every velocity channel.
        if (self.dv > 0.) | (dv_mode == 'thermal'):
            # line profile function
            lnprofs = spectra.glnprof_series(self.v, vlos, dv)

            # get nv
            #start = time.time()
            nv_cube = linecube.to_xyzv(
                np.array([n_gf, n_gr]), lnprofs)
            #end = time.time()
            #print('to_xyzv takes %.2f'%(end-start))

            # collapse
            #start = time.time()
            T_g = self.grid.collapse(T_g)
            nv_gf = self.grid.high_dimensional_collapse(nv_cube[0,:,:], fill = 'zero')
            nv_gr = self.grid.high_dimensional_collapse(nv_cube[1,:,:], fill = 'zero')
            #end = time.time()
            #print('collapsing takes %.2f'%(end-start))

            # to cube
            Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = np.transpose(
                transfer.Tnv_to_cube(
                T_g, nv_gf, nv_gr,
                self.grid.dz * auTOcm,
                self.freq, self.Aul, self.Eu, self.gu, self.Qgrid),
                (0,1,3,2,))
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


    def side(self):
        return self._side



@dataclass(slots=True)
class MultiLayerDisk:
    '''
    Multi-layer Disk Model where thickness of gas layers are taken into account with
     channel-based three-layer approximation.
    The disk model consists of two gas layers having a finite thicnkess
     and a geometrically thin dust layer.
    '''

    # params for dust layer
    Td0: float = 400.
    qd: float = 0.5
    log_tau_dc: float = 0. # tau of dust at rc
    rc_d: float = 1.
    gamma_d: float = 1.
    # params for gas layer
    Tg0: float = 400. # Gas temperature at r0
    qg: float = 0.5 # power-law index of temperature distribution
    log_N_gc: float = 0.
    rc_g: float = 1.
    gamma_g: float = 1.
    z0: float = 0.
    pz: float = 1.25
    # parameters defining thickness of gas layers
    h0: float = 0.
    ph: float = 1.25
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
    dv: float = 0.
    pdv: float = 0.25


    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad)) # cos(-i) = cos(i)


    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_N_gc = 0., rc_g = 100., gamma_g = 1., 
        z0 = 0., pz = 1.25, h0 = 0., ph = 0., inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., dv = 0., pdv = 0.25):
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


        # hidden parameters
        self._inc_rad = np.radians(inc)
        self._pa_rad = np.radians(pa)
        self._side = np.sign(np.cos(self._inc_rad)) # cos(-i) = cos(i)


    @classmethod
    def get_paramkeys(cls):
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != '_']
        return paramkeys


    @classmethod
    def get_annotations(cls):
        d = {}
        for c in cls.mro():
            try:
                d.update(**c.__annotations__)
            except AttributeError:
                # object, at least, has no __annotations__ attribute.
                pass
        return d


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def gas_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Tg0 * (R / self.r0)**(-self.qg)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def dust_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Td0 * (R / self.r0)**(-self.qd)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def gas_speed(self, R, z):
        return vkep(R * auTOcm, self.ms * Msun, z * auTOcm) * 1.e-5 # cm/s --> km/s


    def _gas_velocity_old(self, R, phi, z):
        return vkep(R * auTOcm, self.ms * Msun, z * auTOcm) *\
         np.cos(phi) * np.sin(self._inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s


    def gas_velocity(self, R, phi, z, T, pterm = True, mu = 2.34):
        '''
        Calculate line-of-sight velocity, i.e., 
        projection of the rotational velocity.

        Parameters
        ----------
        R (array): Cylindarical radius (au)
        phi (array): Azimuthal angle (rad)
        z (array): Height (au)
        T (array): Gas temperature (K)
        pterm (bool): If include the pressure gradient term or not.
        '''
        #vphi = vrot(R * auTOcm, self.ms * Msun, rho, T,
        #    z = z * auTOcm, pterm = pterm, mu = mu,)
        vphi = vrot_ssdisk(R * auTOcm, self.ms * Msun, T, 
            self.rc_g * auTOcm, self.gamma_g, self.qg, 
            z = z * auTOcm, pterm = pterm, mu = mu)
        return vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s


    def speed_to_velocity(self, gs, phi,):
        return gs * np.cos(phi) * np.sin(self._inc_rad)


    def gas_density(self, R, z,):
        # surface density
        N_g = ssdisk(R, 10.**self.log_N_gc, self.rc_g, self.gamma_g, beta = None)
        #N_g[np.isnan(N_g)] = 0.  # to prevent computational errors
        #N_g[N_g < 0.] = 0. # safty net

        # puff up layer
        # layer height
        zl = self.z0 * (R / self.r0)**(self.pz) # height
        h_out = self.h0 * (R / self.r0)**(self.ph)
        h_in = h_out

        # check which is fore or rear side
        #side = np.sign(np.cos(self.__inc_rad)) # cos(-i) = cos(i)

        # height of fore/rear layers
        z0f = zl * self._side
        z0r = - zl * self._side

        if self._side > 0.:
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

        ng = np.zeros(N_g.shape)
        ng[zout_f] = self.puff_up_layer(N_g[zout_f], z[zout_f], z0f[zout_f], h_out[zout_f]) / auTOcm # cm^-3
        ng[zin_f] = self.puff_up_layer(N_g[zin_f], z[zin_f], z0f[zin_f], h_in[zin_f]) / auTOcm # cm^-3

        # rear layer
        ng[zout_r] = self.puff_up_layer(N_g[zout_r], z[zout_r], z0r[zout_r], h_out[zout_r]) / auTOcm # cm^-3
        ng[zin_r] = self.puff_up_layer(N_g[zin_r], z[zin_r], z0r[zin_r], h_in[zin_r]) / auTOcm # cm^-3

        return ng


    def puff_up_layer(self, sig, z, z0, H):
        return sig * np.exp( - (z - z0)**2. / (2.*H*H)) / np.sqrt(2. * np.pi) / H


    def linewidth(self, R, Tg = None, dv_mode = 'thermal', mmol = 2.34):
        # line width
        if dv_mode == 'thermal':
            vth = np.sqrt(2. * kb * Tg / mmol / mH) * 1.e-5 # km/s
            if self.dv > 0.:
                vnth = self.dv * (R / self.r0)**(- self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == 'total':
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv
        else:
            print('ERROR\tbuild_gas_layer: dv_mode must be thermal or total.')
            print("ERROR\tbuild_gas_layer: Ignore the input and assume dv is the total line width.")
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv

        return dv


    def dust_density(self, R,):
        return ssdisk(R, 10.**self.log_tau_dc, self.rc_d, self.gamma_d, beta = None)


    def build(self, R, phi, z, Rmid, 
        dv_mode = 'total', collapse = False, 
        mmol = 30., mu = 2.34, pterm = True,):
        '''
        deproject_grid frist.
        '''
        # gas
        T_g = self.gas_temperature(R)
        n_g = self.gas_density(R, z)
        #vlos = self.gas_velocity(R, phi, z)
        vlos = self.gas_velocity(R, phi, z, T_g, 
            pterm = pterm, mu = mu)
        dv = self.linewidth(R, dv_mode = dv_mode, Tg = T_g, mmol = mmol)

        # dust
        T_d = self.dust_temperature(Rmid)
        tau_d = self.dust_density(Rmid)

        return T_g, n_g, vlos, dv, T_d, tau_d


    def build_cube(self, Tcmb = 2.73, f0 = 230., 
        dist = 140., dv_mode = 'total', pterm = True, 
        contsub = True,
        return_Ttau = False, rin = 0.1):
        T_g, vlos, n_gf, n_gr, T_d, tau_d, dv = self.build(rin = rin, dv_mode = dv_mode, pterm = pterm)

        # dust
        T_d = self.grid2D.collapse(T_d)
        tau_d = self.grid2D.collapse(tau_d)

        # To cube
        #  calculate column density and density-weighted temperature 
        #  of each gas layer at every velocity channel.
        if (self.dv > 0.) | (dv_mode == 'thermal'):
            # line profile function
            lnprofs = spectra.glnprof_series(self.v, vlos, dv)

            # get nv
            #start = time.time()
            nv_cube = linecube.to_xyzv(
                np.array([n_gf, n_gr]), lnprofs)
            #end = time.time()
            #print('to_xyzv takes %.2f'%(end-start))

            # collapse
            #start = time.time()
            T_g = self.grid.collapse(T_g)
            nv_gf = self.grid.high_dimensional_collapse(nv_cube[0,:,:], fill = 'zero')
            nv_gr = self.grid.high_dimensional_collapse(nv_cube[1,:,:], fill = 'zero')
            #end = time.time()
            #print('collapsing takes %.2f'%(end-start))

            # to cube
            Tv_gf, Tv_gr, tau_v_gf, tau_v_gr = np.transpose(
                transfer.Tnv_to_cube(
                T_g, nv_gf, nv_gr,
                self.grid.dz * auTOcm,
                self.freq, self.Aul, self.Eu, self.gu, self.Qgrid),
                (0,1,3,2,))
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


    def side(self):
        return self._side


@dataclass(slots=True)
class MultiLayerRingDisk(MultiLayerDisk):

    # add an axisymmetric over density region
    # described with a Gaussian ring
    r_ring: float = 0. # radius of ring
    log_N_ring: float = 0. # peak surface density of ring
    w_ring: float = 0. # with of ring defined as standard deviation of Gaussian
    # geometrical parameters (over write)
    inc: float = 0.
    pa: float = 0.

    # hidden parameters
    #__inc_rad: float = np.radians(inc)
    #__pa_rad: float = np.radians(pa)
    #__side: int = np.sign(np.cos(__inc_rad)) # cos(-i) = cos(i)


    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_N_gc = 0., rc_g = 100., gamma_g = 1., 
        r_ring = 0., log_N_ring = 0., w_ring = 0.,
        z0 = 0., pz = 1.25, h0 = 0., ph = 0., inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., dv = 0., pdv = 0.25):
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
        self.log_N_gc = log_N_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
        # gas layer ring
        self.r_ring = r_ring
        self.log_N_ring = log_N_ring
        self.w_ring = w_ring
        # gas layer height
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


        # hidden parameters
        self._inc_rad = np.radians(inc)
        self._pa_rad = np.radians(pa)
        self._side = np.sign(np.cos(self._inc_rad)) # cos(-i) = cos(i)


    def gas_density(self, R, z,):
        # surface density
        N_pl = ssdisk(R, 10.**self.log_N_gc, self.rc_g, self.gamma_g, beta = None)
        N_rng = 10.**self.log_N_ring * np.exp(- 0.5 * (R - self.r_ring) ** 2. / (self.w_ring**2.))
        N_g = N_pl + N_rng

        # puff up layer
        # layer height
        zl = self.z0 * (R / self.r0)**(self.pz) # height
        h_out = self.h0 * (R / self.r0)**(self.ph)
        h_in = h_out

        # check which is fore or rear side
        #side = np.sign(np.cos(self.__inc_rad)) # cos(-i) = cos(i)

        # height of fore/rear layers
        z0f = zl * self._side
        z0r = - zl * self._side

        if self._side > 0.:
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

        ng = np.zeros(N_g.shape)
        ng[zout_f] = self.puff_up_layer(N_g[zout_f], z[zout_f], z0f[zout_f], h_out[zout_f]) / auTOcm # cm^-3
        ng[zin_f] = self.puff_up_layer(N_g[zin_f], z[zin_f], z0f[zin_f], h_in[zin_f]) / auTOcm # cm^-3

        # rear layer
        ng[zout_r] = self.puff_up_layer(N_g[zout_r], z[zout_r], z0r[zout_r], h_out[zout_r]) / auTOcm # cm^-3
        ng[zin_r] = self.puff_up_layer(N_g[zin_r], z[zin_r], z0r[zin_r], h_in[zin_r]) / auTOcm # cm^-3

        return ng



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
    '''
    Single-layer Disk Model which consists of a thin gas and dust layer.
    '''

    # params for dust layer
    Td0: float = 300.
    qd: float = 0.5
    log_Sig_dc: float = 0. # Surface density of dust at rc
    rc_d: float = 1.
    gamma_d: float = 1.
    # params for gas layer
    Tg0: float = 300. # Gas temperature at r0
    qg: float = 0.5 # power-law index of temperature distribution
    log_N_gc: float = 0.
    rc_g: float = 1.
    gamma_g: float = 1.
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
    dv: float = 0.
    pdv: float = 0.25


    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad)) # cos(-i) = cos(i)


    def set_params(self, 
        Td0 = 400., qd = 0.5, log_Sig_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_N_gc = 0., rc_g = 100., gamma_g = 1., 
        inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., dv = 0., pdv = 0.25):
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
        self.log_Sig_dc = log_Sig_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0 # gas temperature
        self.qg = qg
        self.log_N_gc = log_N_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
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


        # hidden parameters
        self._inc_rad = np.radians(inc)
        self._pa_rad = np.radians(pa)
        self._side = np.sign(np.cos(self._inc_rad)) # cos(-i) = cos(i)


    @classmethod
    def get_paramkeys(cls):
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != '_']
        return paramkeys


    @classmethod
    def get_annotations(cls):
        d = {}
        for c in cls.mro():
            try:
                d.update(**c.__annotations__)
            except AttributeError:
                # object, at least, has no __annotations__ attribute.
                pass
        return d


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def gas_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Tg0 * (R / self.r0)**(-self.qg)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def dust_temperature(self, R):
        # calculate T(R) & tau(R)
        # temperature
        T = self.Td0 * (R / self.r0)**(-self.qd)
        T[np.isnan(T)] = 1. # to prevent computational errors
        T[T <= 1] = 1. # safty net
        return T


    def gas_speed(self, R, z):
        return vkep(R * auTOcm, self.ms * Msun, z * auTOcm) * 1.e-5 # cm/s --> km/s


    def gas_velocity(self, R, phi, z, T, pterm = True, mu = 2.34):
        '''
        Calculate line-of-sight velocity, i.e., 
        projection of the rotational velocity.

        Parameters
        ----------
        R (array): Cylindarical radius (au)
        phi (array): Azimuthal angle (rad)
        z (array): Height (au)
        T (array): Gas temperature (K)
        pterm (bool): If include the pressure gradient term or not.
        '''
        #vphi = vrot(R * auTOcm, self.ms * Msun, rho, T,
        #    z = z * auTOcm, pterm = pterm, mu = mu,)
        vphi = vrot_ssdisk(R * auTOcm, self.ms * Msun, T, 
            self.rc_g * auTOcm, self.gamma_g, self.qg, 
            z = z * auTOcm, pterm = pterm, mu = mu)
        return vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s


    def speed_to_velocity(self, gs, phi,):
        return gs * np.cos(phi) * np.sin(self._inc_rad)


    def gas_density(self, R, mu = 2.34):
        # surface density
        return ssdisk(R, 10.**self.log_N_gc, self.rc_g, self.gamma_g, beta = None)


    def linewidth(self, R, Tg = None, dv_mode = 'thermal', mmol = 2.34):
        # line width
        if dv_mode == 'thermal':
            vth = np.sqrt(2. * kb * Tg / mmol / mH) * 1.e-5 # km/s
            if self.dv > 0.:
                vnth = self.dv * (R / self.r0)**(- self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == 'total':
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv
        else:
            print('ERROR\tbuild_gas_layer: dv_mode must be thermal or total.')
            print("ERROR\tbuild_gas_layer: Ignore the input and assume dv is the total line width.")
            dv = self.dv * (R / self.r0)**(- self.pdv) if self.dv > 0. else self.dv

        return dv


    def dust_density(self, R,):
        return ssdisk(R, 10.**self.log_Sig_dc, self.rc_d, self.gamma_d, beta = None)


    def build(self, R, phi, 
        z = 0., Rmid = None,
        dv_mode = 'total',
        mmol = 30., mu = 2.34, pterm = False,):
        '''
        deproject_grid frist.
        '''
        if Rmid is None:
            Rmid = R

        # gas
        T_g = self.gas_temperature(R)
        N_g = self.gas_density(R)
        #vlos = self.gas_velocity(R, phi, z)
        vlos = self.gas_velocity(R, phi, z, T_g, 
            pterm = pterm, mu = mu)
        dv = self.linewidth(R, dv_mode = dv_mode, Tg = T_g, mmol = mmol)

        # dust
        T_d = self.dust_temperature(Rmid)
        Sig_d = self.dust_density(Rmid)

        return T_g, N_g, vlos, dv, T_d, Sig_d



@dataclass(slots=True)
class SingleLayerDisk_old:

    # params for the disk
    # whichever gas or dust
    T0: float = 300. # temperature
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

    Ic: float = 1.
    rc: float = 1.
    beta: float = 0.
    gamma: float = 0.
    inc: float = 0.
    pa: float = 0.
    ms: float = 0.
    vsys: float = 0.
    dv: float = 0.
    pdv: float = 0.
    r0: float = 1.
    dx0: float = 0.
    dy0: float = 0.

    __inc_rad: float = np.radians(inc)
    __pa_rad: float = np.radians(pa)


    def set_params(self, Ic = 0, rc = 0, beta = 0, gamma = 0, 
        inc = 0, pa = 0, ms = 0, vsys = 0, dv = 0, pdv = 0., r0 = 1.,
        dx0 = 0., dy0 = 0.):
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
        self.ms = ms
        self.vsys = vsys
        self.dv = dv
        self.pdv = pdv
        self.r0 = r0
        self.dx0 = dx0
        self.dy0 = dy0

        self.__inc_rad = np.radians(inc)
        self.__pa_rad = np.radians(pa)


    def get_paramkeys(self):
        paramkeys = list(self.__annotations__.keys())
        paramkeys = [i for i in paramkeys if i[0] != '_']
        return paramkeys


    def build(self, r, phi):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # intensity
        I_int = ssdisk(r, self.Ic, self.rc, self.gamma, self.beta)
        # velocity
        # take y-axis as the line of sight
        vlos = vkep(r * auTOcm, self.ms * Msun) \
        * np.cos(phi) * np.sin(self.__inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s

        dv = self.dv * (r / self.r0)**(-self.pdv)

        return I_int, vlos, dv



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


def vrot2_r(r, ms, rho, cs2, z = 0.):
    '''
    Calculate vrot^2 / r taking into account height and pressure gradient.

    Parameters
    ----------
    r (float or array): Cylinderical radius (cm)
    ms (float): Stellar mass (g)
    rho (float or array): Surface or volume density (g cm^-3 or g cm^-2).
    '''
    #cs2 = cs**2.
    rho_grad = np.gradient(np.log(rho), r)
    cs_grad = np.gradient(cs2, r)
    return vkep(r, ms, z=z)**2. / r + cs2 * rho_grad + cs_grad


def vrot(r, ms, rho, T, z = 0., pterm = True, mu = 2.34):
    if pterm:
        cs2 = kb * T / mu / mH
        return np.sqrt(vrot2_r(r, ms, rho, cs2, z = z) * r)
    else:
        return vkep(r, ms, z = z)


def vrot_ssdisk(r, ms, T, rc, gamma, q,
    z = 0., pterm = True, mu = 2.34):
    '''
    The pressure gradient term will be analytically calculated
    '''
    if pterm:
        cs2 = kb * T / mu / mH
        vkep2 = vkep(r, ms, z = z)**2.
        vrot2 = vkep2 - cs2 * ((2.-gamma) * (r/rc)**(2.-gamma) + q + gamma)
        #plt.scatter(r.ravel()[::100]/auTOcm, np.sqrt(vkep2.ravel()[::100]) * 1e-5)
        #plt.scatter(r.ravel()[::100]/auTOcm, np.sqrt(vrot2.ravel()[::100]) * 1e-5)
        #plt.show()
        #plt.close()
        return np.sqrt(vrot2)
    else:
        return vkep(r, ms, z = z)


def Tdisk_D20(r, z, T0mid, qmid, T0atm, qatm, 
    z0, alpha, beta, r0 = 100.):
    '''

    r (float or array): Radius in an arbitoral unit
    z (float or array): Height

    '''
    Tatm = T0atm * (r/r0)**(-qatm)
    Tmid = T0mid * (r/r0)**(-qmid)
    zq = z0 * (r/r0)**beta

    tanh = np.tanh((z - alpha * zq) / zq)
    T4 = Tmid**4. + 0.5 * (1. + tanh) * Tatm**4.
    return T4**0.25


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
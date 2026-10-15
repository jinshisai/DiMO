# import modules
import sys
import time
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units

from .funcs import beam_convolution, gaussian2d, glnprof_conv, rotate2d
from .grid import Nested3DGrid, Nested2DGrid, Nested3DObsGrid, SubGrid2D
from .molecule import Molecule
from .libcube import spectra, transfer
from .fastbuild import (
    fastbuild_twocompdisk,
    fastbuild_multilayer,
    fastbuild_twocompdisk_brokendvpower,
)


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


@dataclass(slots=True)
class TwoComponentDisk:
    """
    Two-component disk model.
    The model consists of a thick gas layer and a thin dust layer.

    Parameters
    ----------
    Td0 : float
        Dust temperature at ``r0`` in K.
    qd : float
        Power-law index of the dust temperature profile.
    log_tau_dc : float
        Logarithmic dust optical depth at ``rc_d``.
    rc_d : float
        Characteristic radius of the dust layer in au.
    gamma_d : float
        Exponential taper index of the dust surface density.
    Tg0 : float
        Gas temperature at ``r0`` in K.
    qg : float
        Power-law index of the gas temperature profile.
    log_N_gc : float
        Logarithmic gas surface density at ``rc_g``.
    rc_g : float
        Characteristic radius of the gas layer in au.
    gamma_g : float
        Exponential taper index of the gas surface density.
    inc : float
        Inclination angle in degrees.
    pa : float
        Position angle in degrees.
    ms : float
        Stellar mass in solar masses.
    vsys : float
        Systemic velocity in km/s.
    dx0, dy0 : float
        Positional offsets in au.
    r0 : float
        Reference radius in au.
    dv : float
        Line width at ``r0`` in km/s.
    pdv : float
        Power-law index of the line width profile.
    """

    # params for dust layer
    Td0: float = 400.0
    qd: float = 0.5
    log_tau_dc: float = 0.0
    rc_d: float = 1.0
    gamma_d: float = 1.0
    # params for gas layer
    Tg0: float = 400.0
    qg: float = 0.5
    log_N_gc: float = -10.0
    rc_g: float = 1.0
    gamma_g: float = 1.0
    # geometry & velocity
    inc: float = 0.0
    pa: float = 0.0
    ms: float = 1.0
    vsys: float = 0.0
    # positional offsets
    dx0: float = 0.0
    dy0: float = 0.0
    # reference radius
    r0: float = 1.0
    # line width
    dv: float = 0.0
    pdv: float = 0.25

    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad))    # cos(-i) = cos(i)

    def set_params(
        self,
        Td0: float = 400.0,
        qd: float = 0.5,
        log_tau_dc: float = 0.0,
        rc_d: float = 100.0,
        gamma_d: float = 1.0,
        Tg0: float = 400.0,
        qg: float = 0.5,
        log_N_gc: float = -10.0,
        rc_g: float = 100.0,
        gamma_g: float = 1.0,
        inc: float = 0.0,
        pa: float = 0.0,
        ms: float = 1.0,
        vsys: float = 0,
        dx0: float = 0.0,
        dy0: float = 0.0,
        r0: float = 1.0,
        dv: float = 0.0,
        pdv: float = 0.25,
    ):
        """
        Set parameters.

        Parameters
        ----------
        Td0 : float
            Dust temperature at ``r0`` in K.
        qd : float
            Power-law index of the dust temperature profile.
        log_tau_dc : float
            Logarithmic dust optical depth at ``rc_d``.
        rc_d : float
            Characteristic radius of the dust layer in au.
        gamma_d : float
            Exponential taper index of the dust surface density.
        Tg0 : float
            Gas temperature at ``r0`` in K.
        qg : float
            Power-law index of the gas temperature profile.
        log_N_gc : float
            Logarithmic gas surface density at ``rc_g``.
        rc_g : float
            Characteristic radius of the gas layer in au.
        gamma_g : float
            Exponential taper index of the gas surface density.
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        ms : float
            Stellar mass in solar masses.
        vsys : float
            Systemic velocity in km/s.
        dx0, dy0 : float
            Positional offsets in au.
        r0 : float
            Reference radius in au.
        dv : float
            Line width at ``r0`` in km/s.
        pdv : float
            Power-law index of the line width profile.
        """

        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0
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
        self._side = np.sign(np.cos(self._inc_rad))  # cos(-i) = cos(i)

    @classmethod
    def get_paramkeys(cls):
        """
        Get parameter keys.
        """
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != "_"]
        return paramkeys

    @classmethod
    def get_annotations(cls):
        """
        Get annotations.
        """
        d = {}
        for c in cls.mro():
            try:
                d.update(**c.__annotations__)
            except AttributeError:
                # object, at least, has no __annotations__ attribute.
                pass
        return d

    def print_params(self):
        """
        Print parameters
        """
        fields = dataclasses.fields(self)
        for v in fields:
            print(f"{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}")

    def gas_temperature(self, R: np.ndarray):
        """
        Calculate gas temperature as a function of radius.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        """
        T = self.Tg0 * (R / self.r0) ** (-self.qg)
        T[np.isnan(T)] = 1.0  # to prevent computational errors
        T[T < 1] = 1.0  # safety net
        return T

    def dust_temperature(self, R: np.ndarray):
        """
        Calculate dust temperature as a function or radius.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        """
        T = self.Td0 * (R / self.r0) ** (-self.qd)
        T[np.isnan(T)] = 1.0  # to prevent computational errors
        T[T < 1] = 1.0  # safety net
        return T

    def gas_speed(self, R: np.ndarray, z: np.ndarray):
        """
        Calculate gas speed.
        """
        return vkep(R * auTOcm,
                    self.ms * Msun,
                    z * auTOcm) * 1.0e-5  # cm/s --> km/s

    def gas_velocity(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        T: np.ndarray,
        pterm: bool = True,
        mu: float = 2.34,
    ):
        """
        Calculate line-of-sight velocity, i.e.,
        projection of the rotational velocity.
        No vertical and radial velocities are assumed.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        phi : np.ndarray
            Azimuthal angle in radians.
        z : np.ndarray
            Height in au.
        T : np.ndarray
            Gas temperature in K.
        pterm : bool
            Whether to include the pressure gradient term.
        mu : float
            Mean molecular weight.
        """
        vphi = vrot_ssdisk(
            R * auTOcm,
            self.ms * Msun,
            T,
            self.rc_g * auTOcm,
            self.gamma_g,
            self.qg,
            z=z * auTOcm,
            pterm=pterm,
            mu=mu,
        )
        return (
            vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.0e-5 + self.vsys
        )  # cm/s --> km/s

    def speed_to_velocity(
        self,
        gs: np.ndarray,
        phi: np.ndarray,
    ):
        return gs * np.cos(phi) * np.sin(self._inc_rad)

    def gas_density(self, R: np.ndarray, z: np.ndarray, mu: float = 2.34):
        # surface density
        N_g = ssdisk(R, 10.0**self.log_N_gc, self.rc_g, self.gamma_g, beta=None)

        # scale height at r0
        _cs = np.sqrt(kb * self.Tg0 / mu / mH)
        _Omega = np.sqrt(Ggrav * self.ms * Msun / (self.r0 * auTOcm) ** 3.0)
        h0 = _cs / _Omega / auTOcm  # in au
        # pressure scale height
        ph = -0.5 * self.qg + 1.5
        h = h0 * (R / self.r0) ** (ph)

        # density
        ng = self.puff_up_layer(N_g, R, z, h) / auTOcm  # cm^-3
        return ng

    def _puff_up_layer_old(
        self, sig: np.ndarray, z: np.ndarray, z0: np.ndarray, H: np.ndarray
    ):
        return (
            sig * np.exp(-((z - z0) ** 2.0) / (2.0 * H * H)) / np.sqrt(2.0 * np.pi) / H
        )

    def puff_up_layer(
        self, sig: np.ndarray, R: np.ndarray, z: np.ndarray, H: np.ndarray
    ):
        """
        Without approximation of z<<R.
        """
        rho0 = sig / np.sqrt(2.0 * np.pi) / H
        exp = np.exp(R**2.0 / H**2.0 * ((1.0 + z**2 / R**2) ** (-0.5) - 1.0))
        return rho0 * exp

    def linewidth(
        self,
        R: np.ndarray,
        Tg: np.ndarray | None = None,
        dv_mode: str = "thermal",
        mmol: float = 2.34,
    ):
        """
        Compute local line width as a function of radius.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        Tg : np.ndarray, optional
            Gas temperature used when ``dv_mode`` is ``"thermal"``.
        dv_mode : str
            Line-width mode. Use ``"total"`` or ``"thermal"``.
        mmol : float
            Molecular weight used when ``dv_mode`` is ``"thermal"``.
        """
        # line width
        if dv_mode == "thermal":
            vth = np.sqrt(2.0 * kb * Tg / mmol / mH) * 1.0e-5  # km/s
            if self.dv > 0.0:
                vnth = self.dv * (R / self.r0) ** (-self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == "total":
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv
        else:
            print(
                "ERROR\tbuild_gas_layer: "
                "dv_mode must be thermal or total."
                )
            print(
                "ERROR\tbuild_gas_layer: "
                "Ignore the input and assume dv is the total line width."
                )
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv

        return dv

    def dust_density(
        self,
        R: np.ndarray,
    ):
        """
        Caclulate gas density assuming the SS disk.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        """
        return ssdisk(R, 10.0**self.log_tau_dc, self.rc_d,
                      self.gamma_d, beta=None)

    def build(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        Rmid: np.ndarray,
        dv_mode: str = "total",
        mmol: float = 30.0,
        mu: float = 2.34,
        pterm: bool = True,
    ):
        """
        Build up model and return parameters for solving the radiative transfer.

        Parameters
        ----------
        R : np.ndarray
            Three-dimensional array of cylindrical radii in au.
        phi : np.ndarray
            Three-dimensional array of azimuthal angles in radians.
        z : np.ndarray
            Three-dimensional array of vertical heights in au.
        Rmid : np.ndarray
            Two-dimensional array of cylindrical radii in au.
        dv_mode : str
            Line-width mode. Use ``"total"`` or ``"thermal"``.
        mmol : float
            Molecular weight used when ``dv_mode`` is ``"thermal"``.
        mu : float
            Mean molecular weight.
        pterm : bool
            Whether to include the pressure gradient term.
        """
        return self.fastbuild(
            R, phi, z, Rmid, dv_mode=dv_mode, mmol=mmol, mu=mu, pterm=pterm
        )

    def fastbuild(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        Rmid: np.ndarray,
        dv_mode: str = "total",
        mmol: float = 30.0,
        mu: float = 2.34,
        pterm: bool = True,
    ):
        """
        Faster verion of build.
        """
        shape = R.shape
        shape_d = Rmid.shape

        T_g, n_g, vlos, dv, T_d, tau_d = fastbuild_twocompdisk(
            R.ravel(),
            phi.ravel(),
            z.ravel(),
            Rmid.ravel(),
            self.log_N_gc,
            self.rc_g,
            self.gamma_g,
            self.Tg0,
            self.qg,
            self.log_tau_dc,
            self.rc_d,
            self.gamma_d,
            self.Td0,
            self.qd,
            self.dv,
            self.pdv,
            self.r0,
            self.ms * Ggrav * Msun,
            self._inc_rad,
            self.vsys,
            mu,
            mmol,
            dv_mode,
            pterm,
            kb,
            mH,
            auTOcm,
        )
        return (
            T_g.reshape(shape),
            n_g.reshape(shape),
            vlos.reshape(shape),
            dv.reshape(shape),
            T_d.reshape(shape_d),
            tau_d.reshape(shape_d),
        )

    def side(self):
        return self._side


@dataclass(slots=True)
class MultiLayerDisk:
    """
    Multi-layer disk model.
    The model consists of two gas layers with density peak
    at finite vertical heights and with finite thickness,
    and a geometrically thin dust layer.

    Parameters
    ----------
    Td0 : float
        Dust temperature at ``r0`` in K.
    qd : float
        Power-law index of the dust temperature profile.
    log_tau_dc : float
        Logarithmic dust optical depth at ``rc_d``.
    rc_d : float
        Characteristic radius of the dust layer in au.
    gamma_d : float
        Exponential taper index of the dust surface dnesity.
    Tg0 : float
        Gas temperature at ``r0`` in K.
    qg : float
        Power-law index of the gas temperature profile.
    log_N_gc : float
        Logarithmic gas surface density at ``rc_g``.
    rc_g : float
        Characteristic radius of the gas layer in au.
    gamma_g : float
        Exponential taper index of the gas surface density.
    z0 : float
        Gas-layer height at ``r0`` in au.
    pz : float
        Power-law index of the gas-layer height profile.
    h0 : float
        Gas-layer width at ``r0`` in au.
    ph : float
        Power-law index of the gas-layer width profile.
    inc : float
        Inclination angle in degrees.
    pa : float
        Position angle in degrees.
    ms : float
        Stellar mass in solar masses.
    vsys : float
        Systemic velocity in km/s.
    dx0, dy0 : float
        Positional offsets in au.
    r0 : float
        Reference radius in au.
    dv : float
        Line width at ``r0`` in km/s.
    pdv : float
        Power-law index of the line width profile.
    """

    # params for dust layer
    Td0: float = 400.0
    qd: float = 0.5
    log_tau_dc: float = 0.0
    rc_d: float = 1.0
    gamma_d: float = 1.0
    # params for gas layer
    Tg0: float = 400.0
    qg: float = 0.5
    log_N_gc: float = 0.0
    rc_g: float = 1.0
    gamma_g: float = 1.0
    z0: float = 0.0
    pz: float = 1.25
    # parameters defining thickness of gas layers
    h0: float = 0.0
    ph: float = 1.25
    # geometry & velocity
    inc: float = 0.0
    pa: float = 0.0
    ms: float = 1.0
    vsys: float = 0.0
    # positional offsets
    dx0: float = 0.0
    dy0: float = 0.0
    # reference radius
    r0: float = 1.0
    # line width
    dv: float = 0.0
    pdv: float = 0.25

    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad))  # cos(-i) = cos(i)

    def set_params(
        self,
        Td0: float = 400.0,
        qd: float = 0.5,
        log_tau_dc: float = 0.0,
        rc_d: float = 100.0,
        gamma_d: float = 1.0,
        Tg0: float = 400.0,
        qg: float = 0.5,
        log_N_gc: float = 0.0,
        rc_g: float = 100.0,
        gamma_g: float = 1.0,
        z0: float = 0.0,
        pz: float = 1.25,
        h0: float = 0.0,
        ph: float = 0.0,
        inc: float = 0.0,
        pa: float = 0.0,
        ms: float = 1.0,
        vsys: float = 0,
        dx0: float = 0.0,
        dy0: float = 0.0,
        r0: float = 1.0,
        dv: float = 0.0,
        pdv: float = 0.25,
    ):
        """
        Set parameters.

        Parameters
        ----------
        Td0 : float
            Dust temperature at ``r0`` in K.
        qd : float
            Power-law index of the dust temperature profile.
        log_tau_dc : float
            Logarithmic dust optical depth at ``rc_d``.
        rc_d : float
            Characteristic radius of the dust layer in au.
        gamma_d : float
            Exponential taper index of the dust surface dnesity.
        Tg0 : float
            Gas temperature at ``r0`` in K.
        qg : float
            Power-law index of the gas temperature profile.
        log_N_gc : float
            Logarithmic gas surface density at ``rc_g``.
        rc_g : float
            Characteristic radius of the gas layer in au.
        gamma_g : float
            Exponential taper index of the gas surface density.
        z0 : float
            Gas-layer height at ``r0`` in au.
        pz : float
            Power-law index of the gas-layer height profile.
        h0 : float
            Gas-layer width at ``r0`` in au.
        ph : float
            Power-law index of the gas-layer width profile.
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        ms : float
            Stellar mass in solar masses.
        vsys : float
            Systemic velocity in km/s.
        dx0, dy0 : float
            Positional offsets in au.
        r0 : float
            Reference radius in au.
        dv : float
            Line width at ``r0`` in km/s.
        pdv : float
            Power-law index of the line width profile.
        """
        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0
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
        self._side = np.sign(np.cos(self._inc_rad))  # cos(-i) = cos(i)

    @classmethod
    def get_paramkeys(cls):
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != "_"]
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
            print(f"{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}")

    def gas_temperature(self, R: np.ndarray):
        T = self.Tg0 * (R / self.r0) ** (-self.qg)
        T[np.isnan(T)] = 1.0  # to prevent computational errors
        T[T <= 1] = 1.0  # safety net
        return T

    def dust_temperature(self, R: np.ndarray):
        T = self.Td0 * (R / self.r0) ** (-self.qd)
        T[np.isnan(T)] = 1.0  # to prevent computational errors
        T[T <= 1] = 1.0  # safety net
        return T

    def gas_speed(self, R: np.ndarray, z: np.ndarray):
        return vkep(R * auTOcm, self.ms * Msun,
                    z * auTOcm) * 1.0e-5  # cm/s --> km/s

    def _gas_velocity_old(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray
    ):
        return (
            vkep(R * auTOcm, self.ms * Msun, z * auTOcm)
            * np.cos(phi)
            * np.sin(self._inc_rad)
            * 1.0e-5
            + self.vsys
        )  # cm/s --> km/s

    def gas_velocity(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        T: np.ndarray,
        pterm: bool = True,
        mu: float = 2.34,
    ):
        """
        Calculate line-of-sight velocity, i.e.,
        projection of the rotational velocity.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        phi : np.ndarray
            Azimuthal angle in radians.
        z : np.ndarray
            Height in au.
        T : np.ndarray
            Gas temperature in K.
        pterm : bool
            Whether to include the pressure gradient term.
        mu : float
            Mean molecular weight.
        """
        vphi = vrot_ssdisk(
            R * auTOcm,
            self.ms * Msun,
            T,
            self.rc_g * auTOcm,
            self.gamma_g,
            self.qg,
            z=z * auTOcm,
            pterm=pterm,
            mu=mu,
        )
        return (
            vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.0e-5 + self.vsys
        )  # cm/s --> km/s

    def speed_to_velocity(
        self,
        gs: np.ndarray,
        phi: np.ndarray,
    ):
        return gs * np.cos(phi) * np.sin(self._inc_rad)

    def gas_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
    ):
        # surface density
        N_g = ssdisk(R, 10.0**self.log_N_gc,
                     self.rc_g, self.gamma_g, beta=None)

        # puff up layer
        # layer height
        zl = self.z0 * (R / self.r0) ** (self.pz)
        h_out = self.h0 * (R / self.r0) ** (self.ph)
        h_in = h_out

        # height of fore/rear layers
        z0f = zl * self._side
        z0r = -zl * self._side

        if self._side > 0.0:
            # positive z is fore side
            zout_f = np.where(z - z0f > 0.0)  # outer side
            zin_f = np.where((z <= z0f) * (z > 0.0))  # inner side
            # negative z is rear side
            zout_r = np.where(z - z0r < 0.0)  # outer side
            zin_r = np.where((z >= z0r) * (z < 0.0))  # inner side
        else:
            # positive z is rear side
            zout_r = np.where(z - z0r > 0.0)  # outer side
            zin_r = np.where((z <= z0r) * (z > 0.0))  # inner side
            # negative z is fore side
            zout_f = np.where(z - z0f < 0.0)  # outer side
            zin_f = np.where((z >= z0f) * (z < 0.0))  # inner side

        ng = np.zeros(N_g.shape)
        ng[zout_f] = (
            self.puff_up_layer(
                N_g[zout_f], z[zout_f],
                z0f[zout_f], h_out[zout_f]) / auTOcm
        )  # cm^-3
        ng[zin_f] = (
            self.puff_up_layer(
                N_g[zin_f], z[zin_f],
                z0f[zin_f], h_in[zin_f]) / auTOcm
        )  # cm^-3

        # rear layer
        ng[zout_r] = (
            self.puff_up_layer(
                N_g[zout_r], z[zout_r],
                z0r[zout_r], h_out[zout_r]) / auTOcm
        )  # cm^-3
        ng[zin_r] = (
            self.puff_up_layer(
                N_g[zin_r], z[zin_r],
                z0r[zin_r], h_in[zin_r]) / auTOcm
        )  # cm^-3

        return ng

    def puff_up_layer(
        self, sig: np.ndarray, z: np.ndarray, z0: np.ndarray, H: np.ndarray
    ):
        return (
            sig * np.exp(-((z - z0) ** 2.0) / (2.0 * H * H)) \
            / np.sqrt(2.0 * np.pi) / H
        )

    def linewidth(
        self,
        R: np.ndarray,
        Tg: np.ndarray | None = None,
        dv_mode: str = "thermal",
        mmol: float = 2.34,
    ):
        # line width
        if dv_mode == "thermal":
            vth = np.sqrt(2.0 * kb * Tg / mmol / mH) * 1.0e-5  # km/s
            if self.dv > 0.0:
                vnth = self.dv * (R / self.r0) ** (-self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == "total":
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv
        else:
            print("ERROR\tbuild_gas_layer: "
                  "dv_mode must be thermal or total.")
            print(
                "ERROR\tbuild_gas_layer: "
                "Ignore the input and assume dv is the total line width."
            )
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv

        return dv

    def dust_density(
        self,
        R: np.ndarray,
    ):
        return ssdisk(R, 10.0**self.log_tau_dc,
                      self.rc_d, self.gamma_d, beta=None)

    def build(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        Rmid: np.ndarray,
        dv_mode: str = "total",
        mmol: float = 30.0,
        mu: float = 2.34,
        pterm: bool = True,
    ):
        """
        Build the model. deproject_grid frist.
        """
        return self.fastbuild(
            R,
            phi,
            z,
            Rmid,
            dv_mode=dv_mode,
            mmol=mmol,
            mu=mu,
            pterm=pterm,
        )

    def fastbuild(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        Rmid: np.ndarray,
        dv_mode: str = "total",
        mmol: float = 30.0,
        mu: float = 2.34,
        pterm: bool = True,
    ):
        shape = R.shape
        shape_d = Rmid.shape

        T_g, n_g, vlos, dv, T_d, tau_d = fastbuild_multilayer(
            R.ravel(),
            phi.ravel(),
            z.ravel(),
            Rmid.ravel(),
            self.log_N_gc,
            self.rc_g,
            self.gamma_g,
            self.Tg0,
            self.qg,
            self.z0,
            self.pz,
            self.h0,
            self.ph,
            self.log_tau_dc,
            self.rc_d,
            self.gamma_d,
            self.Td0,
            self.qd,
            self.dv,
            self.pdv,
            self.r0,
            self.ms * Ggrav * Msun,
            self._inc_rad,
            self.vsys,
            mu,
            mmol,
            dv_mode,
            pterm,
            kb,
            mH,
            auTOcm,
        )
        return (
            T_g.reshape(shape),
            n_g.reshape(shape),
            vlos.reshape(shape),
            dv.reshape(shape),
            T_d.reshape(shape_d),
            tau_d.reshape(shape_d),
        )

    def side(self):
        return self._side


@dataclass(slots=True)
class TwoSlabDisk:
    """
    Twp-slab disk model.
    The model consists of a geometrically thin dust layer
    behind a geometrically thin gas layer. The dust and gas
    layers share the same geometry and temperature profile.

    Parameters
    ----------
    T0 : float
        Temperature at ``r0`` in K.
    q : float
        Power-law index of the temperature profile.
    rc : float
        Characteristic radius in au.
    gamma : float
        Exponential taper index.
    log_N_gc : float
        Logarithmic gas surface density at ``rc``.
    log_tau_dc : float
        Logarithmic dust optical depth at ``rc``.
    inc : float
        Inclination angle in degrees.
    pa : float
        Position angle in degrees.
    ms : float
        Stellar mass in solar masses.
    vsys : float
        Systemic velocity in km/s.
    dx0, dy0 : float
        Positional offsets in au.
    r0 : float
        Reference radius in au.
    dv : float
        Line width at ``r0`` in km/s.
    pdv : float
        Power-law index of the line width profile.
    """

    # params for dust layer
    T0: float = 300.0
    q: float = 0.5
    rc: float = 1.0
    gamma: float = 1.0
    log_N_gc: float = 0.0
    log_tau_dc: float = 0.0
    # geometry & velocity
    inc: float = 0.0
    pa: float = 0.0
    ms: float = 1.0
    vsys: float = 0.0
    # positional offsets
    dx0: float = 0.0
    dy0: float = 0.0
    # reference radius
    r0: float = 1.0
    # line width
    dv: float = 0.0
    pdv: float = 0.25

    # hidden parameters
    _inc_rad: float = np.radians(inc)
    _pa_rad: float = np.radians(pa)
    _side: int = np.sign(np.cos(_inc_rad))  # cos(-i) = cos(i)

    def set_params(
        self,
        T0: float = 400.0,
        q: float = 0.5,
        rc: float = 100.0,
        gamma: float = 1.0,
        log_N_gc: float = 0.0,
        log_tau_dc: float = 0.0,
        inc: float = 0.0,
        pa: float = 0.0,
        ms: float = 1.0,
        vsys: float = 0,
        dx0: float = 0.0,
        dy0: float = 0.0,
        r0: float = 1.0,
        dv: float = 0.0,
        pdv: float = 0.25,
    ):
        """
        Set parameters.

        Parameters
        ----------
        T0 : float
            Temperature at ``r0`` in K.
        q : float
            Power-law index of the temperature profile.
        rc : float
            Characteristic radius in au.
        gamma : float
            Exponential taper index.
        log_N_gc : float
            Logarithmic gas surface density at ``rc``.
        log_tau_dc : float
            Logarithmic dust optical depth at ``rc``.
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        ms : float
            Stellar mass in solar masses.
        vsys : float
            Systemic velocity in km/s.
        dx0, dy0 : float
            Positional offsets in au.
        r0 : float
            Reference radius in au.
        dv : float
            Line width at ``r0`` in km/s.
        pdv : float
            Power-law index of the line width profile.
        """
        # initialize parameters
        # dust layer
        self.T0 = T0
        self.q = q
        self.rc = rc
        self.gamma = gamma
        self.log_N_gc = log_N_gc
        self.log_tau_dc = log_tau_dc
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
        self._side = np.sign(np.cos(self._inc_rad))  # cos(-i) = cos(i)

    @classmethod
    def get_paramkeys(cls):
        paramkeys = list(cls.get_annotations().keys())
        paramkeys = [i for i in paramkeys if i[0] != "_"]
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
            print(f"{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}")

    def temperature(self, R: np.ndarray):
        T = self.T0 * (R / self.r0) ** (-self.q)
        T[np.isnan(T)] = 1.0  # to prevent computational errors
        T[T <= 1] = 1.0  # safety net
        return T

    def gas_speed(self, R: np.ndarray, z: np.ndarray):
        return vkep(R * auTOcm,
                    self.ms * Msun, z * auTOcm) * 1.0e-5  # cm/s --> km/s

    def gas_velocity(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        T: np.ndarray,
        pterm: bool = True,
        mu: float = 2.34,
    ):
        """
        Calculate line-of-sight velocity, i.e.,
        projection of the rotational velocity.

        Parameters
        ----------
        R : np.ndarray
            Cylindrical radius in au.
        phi : np.ndarray
            Azimuthal angle in radians.
        z : np.ndarray
            Height in au.
        T : np.ndarray
            Gas temperature in K.
        pterm : bool
            Whether to include the pressure gradient term.
        mu : float
            Mean molecular weight.
        """
        vphi = vrot_ssdisk(
            R * auTOcm,
            self.ms * Msun,
            T,
            self.rc * auTOcm,
            self.gamma,
            self.q,
            z=z * auTOcm,
            pterm=pterm,
            mu=mu,
        )
        return (
            vphi * np.cos(phi) * np.sin(self._inc_rad) * 1.0e-5 + self.vsys
        )  # cm/s --> km/s

    def speed_to_velocity(
        self,
        gs: np.ndarray,
        phi: np.ndarray,
    ):
        return gs * np.cos(phi) * np.sin(self._inc_rad)

    def gas_density(self, R: np.ndarray, mu: float = 2.34):
        # surface density
        return ssdisk(R, 10.0**self.log_N_gc, self.rc, self.gamma, beta=None)

    def linewidth(
        self,
        R: np.ndarray,
        Tg: np.ndarray | None = None,
        dv_mode: str = "thermal",
        mmol: float = 2.34,
    ):
        # line width
        if dv_mode == "thermal":
            vth = np.sqrt(2.0 * kb * Tg / mmol / mH) * 1.0e-5  # km/s
            if self.dv > 0.0:
                vnth = self.dv * (R / self.r0) ** (-self.pdv)
                dv = np.sqrt(vth * vth + vnth * vnth)
            else:
                dv = np.sqrt(vth * vth)
        elif dv_mode == "total":
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv
        else:
            print("ERROR\tbuild_gas_layer: "
                "dv_mode must be thermal or total.")
            print(
                "ERROR\tbuild_gas_layer: "
                "Ignore the input and assume dv is the total line width."
            )
            if self.dv > 0.0:
                dv = self.dv * (R / self.r0) ** (-self.pdv)
            else:
                dv = self.dv

        return dv

    def dust_density(
        self,
        R: np.ndarray,
    ):
        return ssdisk(R, 10.0**self.log_tau_dc,
                      self.rc, self.gamma, beta=None)

    def build(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: float | np.ndarray = 0.0,
        Rmid: np.ndarray | None = None,
        dv_mode: str = "total",
        mmol: float = 30.0,
        mu: float = 2.34,
        pterm: bool = False,
    ):
        """
        deproject_grid frist.
        """
        if Rmid is None:
            Rmid = R

        # gas
        T = self.temperature(R)
        N_g = self.gas_density(R)
        vlos = self.gas_velocity(R, phi, z, T, pterm=pterm, mu=mu)
        dv = self.linewidth(R, dv_mode=dv_mode, Tg=T, mmol=mmol)

        # dust
        tau_d = self.dust_density(Rmid)

        return T, N_g, vlos, dv, tau_d


@dataclass(slots=True)
class SingleLayerDisk:
    """
    Single-layer disk model.
    This model builds sky-plane images and cubes from a flared single layer.

    Parameters
    ----------
    T0 : float
        Temperature at ``r0`` in K.
    q : float
        Power-law index of the temperature profile.
    log_tau_c : float
        Logarithmic optical depth at ``rc``.
    rc : float
        Characteristic radius in au.
    gamma : float
        Exponential taper index.
    z0 : float
        Disk height at ``r0`` in au.
    hp : float
        Power-law index of the disk height.
    inc : float
        Inclination angle in degrees.
    pa : float
        Position angle in degrees.
    ms : float
        Stellar mass in solar masses.
    vsys : float
        Systemic velocity in km/s.
    dx0, dy0 : float
        Positional offsets in au.
    r0 : float
        Reference radius in au.
    delv : float
        Line width in km/s.
    """

    # params for the disk
    # whichever gas or dust
    T0: float = 300.0
    q: float = 0.5
    log_tau_c: float = 0.0
    rc: float = 100.0
    gamma: float = 1.0
    # make it flared if you like
    z0: float = 0.0
    hp: float = 1.25
    # geometry
    inc: float = 0.0
    pa: float = 0.0
    # stellar mass and vsys matter only for line case
    ms: float = 1.0
    vsys: float = 0.0
    # positional offsets
    dx0: float = 0.0
    dy0: float = 0.0
    # reference radius
    r0: float = 1.0
    # line width
    delv: float = 0.0

    def set_params(
        self,
        T0: float = 400.0,
        q: float = 0.5,
        log_tau_c: float = 0.0,
        rc: float = 100.0,
        gamma: float = 1.0,
        z0: float = 0.0,
        hp: float = 1.25,
        inc: float = 0.0,
        pa: float = 0.0,
        ms: float = 1.0,
        vsys: float = 0.0,
        dx0: float = 0.0,
        dy0: float = 0.0,
        r0: float = 1.0,
        delv: float = 0.0,
    ):
        """
        Set parameters.

        Parameters
        ----------
        T0 : float
            Temperature at ``r0`` in K.
        q : float
            Power-law index of the temperature profile.
        log_tau_c : float
            Logarithmic optical depth at ``rc``.
        rc : float
            Characteristic radius in au.
        gamma : float
            Exponential taper index.
        z0 : float
            Disk height at ``r0`` in au.
        hp : float
            Power-law index of the disk height.
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        ms : float
            Stellar mass in solar masses.
        vsys : float
            Systemic velocity in km/s.
        dx0, dy0 : float
            Positional offsets in au.
        r0 : float
            Reference radius in au.
        delv : float
            Line width in km/s.
        """
        # initialize parameters
        # dust layer
        self.T0 = T0
        self.q = q
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
            print(f"{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}")

    def build(
        self,
        xx_sky: np.ndarray,
        yy_sky: np.ndarray,
        rin: float = 0.1):
        """
        Build a model given sky coordinates
        and return a info for making a image cube.
        """
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0 * (r / r0) ** hp
        _dfz = lambda x, y, z0, r0, hp: (
            2.0 * y * 0.5 / np.sqrt(x * x + y * y)
            / r0 * z0 * hp * (np.sqrt(x * x + y * y) / r0)
            * (hp - 1.0)
        )

        # calculate temperature (T), velocity (v) and tau (t)
        def get_Tvt(
            xx_sky: np.ndarray,
            yy_sky: np.ndarray,
            _fz: Callable[..., np.ndarray],
            zargs: list[float],
            inc: float,
            pa: float,
            ms: float,
            T0: float,
            q: float,
            r0: float,
            tau_c: float,
            rc: float,
            gamma: float,
            _dfz: Callable[..., np.ndarray] | None = None,
        ):
            # deprojection
            depr = sky_to_local(
                xx_sky,
                yy_sky,
                inc,
                pa + 0.5 * np.pi,
                _fz,
                zargs,
                _dfz,
                zarg_lims=[[-0.3, 0.3], [0.1, 100.1], [0.0, 2.0]],
            )
            if type(depr) == int:
                T = np.full(
                    xx_sky.shape, 1.0
                )  # 1 instead of zero to prevent numerical errors
                vlos = np.zeros(xx_sky.shape)
                tau = np.zeros(xx_sky.shape)
                return T, vlos, tau
            else:
                xx, yy = depr

            # local coordinates
            rr = np.sqrt(xx * xx + yy * yy)  # radius
            phph = np.arctan2(yy, xx)  # azimuthal angle (rad)
            zz = _fz(rr, *zargs)  # height
            # prevent r=0
            rr[rr == 0.0] = np.nan

            # quantities
            # temperature
            T = T0 * (rr / r0) ** (-q)
            T[np.isnan(T)] = 1.0  # to prevent computational errors
            T[T <= 1.0] = 1.0  # safety net

            # line of sight velocity
            # take y-axis as the line of sight
            vlos = (
                vkep(rr * auTOcm, ms * Msun, zz * auTOcm)
                * np.cos(phph)
                * np.sin(_inc_rad)
                * 1.0e-5
            )  # cm/s --> km/s

            # tau
            tau = ssdisk(rr, tau_c, rc, gamma, beta=None)
            tau[np.isnan(tau)] = 0.0  # to prevent computational errors
            tau[tau < 0.0] = 0.0  # safety net

            T[rr < rin] = 0.0
            vlos[rr < rin] = 0.0
            tau[rr < rin] = 0.0

            return T, vlos, tau

        # for a layer
        T, vlos, tau = get_Tvt(
            xx_sky - self.dx0,
            yy_sky - self.dy0,
            _fz,
            [self.z0, self.r0, self.hp],
            _inc_rad,
            _pa_rad,
            self.ms,
            self.T0,
            self.q,
            self.r0,
            10.0**self.log_tau_c,
            self.rc,
            self.gamma,
        )

        return T, vlos, tau

    def build_cube(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        v: np.ndarray,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
        return_tau: bool = False,
    ):
        # get quantities
        T, vlos, tau = self.build(xx, yy)
        vlos += self.vsys

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0, 1] - xx[0, 0]
        dy = yy[1, 0] - yy[0, 0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # making a cube
        _tau = np.zeros((nv, ny, nx))
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist=dist, au=True)
        # calculate tau_v
        for i in range(nv):
            _tau[i, :, :] = np.where(
                (ve[i] <= vlos) & (vlos < ve[i + 1]),
                tau,
                0.0
                )
        # line width
        if self.delv > 0.0:
            _tau = glnprof_conv(_tau, v, self.delv)

        if return_tau:
            return _tau

        # radiative transfer
        _Bv_bg = np.tile(_Bv(Tcmb, f0), (nv, 1, 1))
        _Bv_T = np.tile(_Bv(T, f0), (nv, 1, 1))
        I_cube = (_Bv_T - _Bv_bg) * (1.0 - np.exp(-_tau))

        # Convolve beam if given
        if beam is not None:
            I_cube = beam_convolution(
                xx, yy, I_cube, [beam[0], beam[1], beam[2]]
            )

        # return intensity
        return I_cube

    def build_nested_cube(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        nnest: int,
        xlim: float | None = None,
        ylim: float | None = None,
        reslim: float = 10.,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
        return_tau: bool = False,
    ):
        # original grid
        nx = len(x)
        ny = len(y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # nested grid
        nstgrid = Nested2DGrid(
            x, y, nsub = nnest,
            xlim = xlim, ylim = ylim, reslim = reslim
        )

        # get quantities
        T, vlos, tau = self.build(nstgrid.xnest, nstgrid.ynest)
        vlos += self.vsys

        # making a cube
        _tau = np.zeros((nv, nstgrid.xnest.size,))
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist=dist, au=True)

        # calculate tau_v
        for i in range(nv):
            _tau[i, :] = np.where(
                (ve[i] <= vlos) & (vlos < ve[i + 1]),
                tau,
                0.0
                )

        # line width
        if self.delv > 0.0:
            _tau = glnprof_conv(_tau, v, self.delv)

        if return_tau:
            return nstgrid.high_dimensional_collapse(_tau)

        # radiative transfer
        _Bv_bg = np.tile(_Bv(Tcmb, f0), (nv, 1))
        _Bv_T = np.tile(_Bv(T, f0), (nv, 1))
        I_cube = (_Bv_T - _Bv_bg) * (1.0 - np.exp(-_tau))

        # collapse
        I_cube = nstgrid.high_dimensional_collapse(I_cube)

        # Convolve beam if given
        if beam is not None:
            I_cube = beam_convolution(
                nstgrid.xx, nstgrid.yy,
                I_cube, [beam[0], beam[1], beam[2]]
            )

        return I_cube

    def build_cube_subgrid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        nsub: int = 2,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
    ):
        if nsub < 1:
            print("ERROR\tbuild_cube_subgrid: nsub must be >= 2.")
            return 0
        subgrid = SubGrid2D(x, y, nsub=nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub
        I_cube_sub = self.build_cube(_xx, _yy, v, beam, dist, Tcmb, f0)

        return subgrid.binning_onsubgrid_layered(I_cube_sub)

    def build_cont(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
    ):
        # get quantities
        T, _, tau = self.build(xx, yy)

        # grid
        ny, nx = xx.shape
        dx = xx[0, 1] - xx[0, 0]
        dy = yy[1, 0] - yy[0, 0]

        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist=dist, au=True)
        Iv = (_Bv(T, f0) - _Bv(Tcmb, f0)) * (1.0 - np.exp(-tau))

        # Convolve beam if given
        if beam is not None:
            Iv = beam_convolution(xx, yy, Iv, [beam[0], beam[1], beam[2]])

        # return intensity
        return Iv

    def build_cont_subgrid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
        nsub: int = 2,
    ):
        # subgrid
        subgrid = SubGrid2D(x, y, nsub=nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub

        # cube on the original grid
        Iv = self.build_cont(
            _xx,
            _yy,
            beam=beam,
            dist=dist,
            Tcmb=Tcmb,
            f0=f0,
        )

        # return intensity
        return subgrid.binning_onsubgrid_layered(Iv)

    def build_nested_cont(
        self,
        x: np.ndarray,
        y: np.ndarray,
        nnest: int,
        xlim: float | None = None,
        ylim: float | None = None,
        reslim: float = 10.,
        beam: list[float] | None = None,
        dist: float = 140.0,
        Tcmb: float = 2.73,
        f0: float = 230.0,
    ):
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # nested grid
        nstgrid = Nested2DGrid(
            x, y, nsub = nnest,
            xlim = xlim, ylim = ylim, reslim = reslim)
        xnest = nstgrid.xnest
        ynest = nstgrid.ynest

        # build model on the nested grid
        T, _, tau = self.build(xnest, ynest)

        # radiative transfer
        _Bv = lambda T, v: Bvppx(
            T, v,
            dx, dy,
            dist=dist, au=True
        )  # in unit of Jy/original_pixel
        Iv = (_Bv(T, f0) - _Bv(Tcmb, f0)) * (1.0 - np.exp(-tau))

        # back to original grid
        Iv = nstgrid.collapse(Iv)

        # Convolve beam if given
        if beam is not None:
            Iv = beam_convolution(
                nstgrid.xx, nstgrid.yy,
                Iv, [beam[0], beam[1], beam[2]]
            )

        return Iv


@dataclass(slots=True)
class SSDisk:
    """
    Self-similar disk model.
    This model returns intensity, line-of-sight velocity, and local line width
    on a cylindrical grid.

    Parameters
    ----------
    Ic : float
        Intensity at ``rc``.
    rc : float
        Characteristic radius in au.
    beta : float
        Surface-brightness power-law index.
    gamma : float
        Exponential taper index.
    inc : float
        Inclination angle in degrees.
    pa : float
        Position angle in degrees.
    ms : float
        Stellar mass in solar masses.
    vsys : float
        Systemic velocity in km/s.
    dv : float
        Line width in km/s.
    pdv : float
        Power-law index of the line width.
    r0 : float
        Reference radius in au.
    dx0, dy0 : float
        Positional offsets in au.
    """

    Ic: float = 1.0
    rc: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    inc: float = 0.0
    pa: float = 0.0
    ms: float = 0.0
    vsys: float = 0.0
    dv: float = 0.0
    pdv: float = 0.0
    r0: float = 1.0
    dx0: float = 0.0
    dy0: float = 0.0

    __inc_rad: float = np.radians(inc)
    __pa_rad: float = np.radians(pa)

    def set_params(
        self,
        Ic: float = 0,
        rc: float = 0,
        beta: float = 0,
        gamma: float = 0,
        inc: float = 0,
        pa: float = 0,
        ms: float = 0,
        vsys: float = 0,
        dv: float = 0,
        pdv: float = 0.0,
        r0: float = 1.0,
        dx0: float = 0.0,
        dy0: float = 0.0,
    ):
        """
        Set parameters.

        Parameters
        ----------
        Ic : float
            Intensity at ``rc``.
        rc : float
            Characteristic radius in au.
        beta : float
            Surface-brightness power-law index.
        gamma : float
            Exponential taper index.
        inc : float
            Inclination angle in degrees.
        pa : float
            Position angle in degrees.
        ms : float
            Stellar mass in solar masses.
        vsys : float
            Systemic velocity in km/s.
        dv : float
            Line width in km/s.
        pdv : float
            Power-law index of the line width.
        r0 : float
            Reference radius in au.
        dx0, dy0 : float
            Positional offsets in au.
        """
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
        paramkeys = [i for i in paramkeys if i[0] != "_"]
        return paramkeys

    def build(self, r: np.ndarray, phi: np.ndarray):
        """
        Build a model given sky coordinates
        and return a info for making a image cube.
        """
        # intensity
        I_int = ssdisk(r, self.Ic, self.rc, self.gamma, self.beta)
        # velocity
        # take y-axis as the line of sight
        vlos = (
            vkep(r * auTOcm, self.ms * Msun)
            * np.cos(phi)
            * np.sin(self.__inc_rad)
            * 1.0e-5
            + self.vsys
        )  # cm/s --> km/s

        dv = self.dv * (r / self.r0) ** (-self.pdv)

        return I_int, vlos, dv


# Radial profiles
def powerlaw_profile(r: np.ndarray, p: float, I0: float, r0: float = 1.0):
    return I0 * (r / r0) ** (-p)


def mdpowerlaw_profile(r: np.ndarray, a0: float, alpha: float, p: float):
    return a0 * (1.0 + alpha * r) ** (-p)


def nuker_profile(
    r: np.ndarray, rt: float, It: float,
    alpha: float, beta: float, gamma: float
):
    return (
        It
        * (r / rt) ** (-gamma)
        * (1.0 + (r / rt) ** alpha) ** ((gamma - beta) / alpha)
    )


def ssdisk(
    r: np.ndarray, Ic: float, rc: float,
    gamma: float, beta: float | None = None
):
    beta_p = gamma if beta is None else beta  # - beta = - gamma - q
    return Ic * (r / rc) ** (-beta_p) * np.exp(-((r / rc) ** (2.0 - gamma)))


def gaussian_profile(r: np.ndarray, I0: float, sigr: float):
    return I0 * np.exp(-(r**2.0) / (2.0 * sigr**2))


def vkep(r: np.ndarray, ms: float, z: float | np.ndarray = 0.0):
    return np.sqrt(Ggrav * ms * r * r / (r * r + z * z) ** (1.5))


def vrot2_r(
    r: np.ndarray,
    ms: float,
    rho: np.ndarray,
    cs2: np.ndarray,
    z: float | np.ndarray = 0.0,
):
    """
    Calculate vrot^2 / r taking into account height and pressure gradient.

    Parameters
    ----------
    r : np.ndarray
        Cylindrical radius in cm.
    ms : float
        Stellar mass in g.
    rho : np.ndarray
        Surface or volume density in g cm^-2 or g cm^-3.
    cs2 : np.ndarray
        Squared sound speed.
    z : float or np.ndarray
        Height in cm.
    """
    # cs2 = cs**2.
    rho_grad = np.gradient(np.log(rho), r)
    cs_grad = np.gradient(cs2, r)
    return vkep(r, ms, z=z) ** 2.0 / r + cs2 * rho_grad + cs_grad


def vrot(
    r: np.ndarray,
    ms: float,
    rho: np.ndarray,
    T: np.ndarray,
    z: float | np.ndarray = 0.0,
    pterm: bool = True,
    mu: float = 2.34,
):
    if pterm:
        cs2 = kb * T / mu / mH
        return np.sqrt(vrot2_r(r, ms, rho, cs2, z=z) * r)
    else:
        return vkep(r, ms, z=z)


def vrot_ssdisk(
    r: np.ndarray,
    ms: float,
    T: np.ndarray,
    rc: float,
    gamma: float,
    q: float,
    z: float | np.ndarray = 0.0,
    pterm: bool = True,
    mu: float = 2.34,
):
    """
    The pressure gradient term will be analytically calculated
    """
    if pterm:
        cs2 = kb * T / mu / mH
        vkep2 = vkep(r, ms, z=z) ** 2.0
        vrot2 = vkep2 - cs2 * (
            (2.0 - gamma) * (r / rc) ** (2.0 - gamma) + q + gamma)
        return np.sqrt(vrot2)
    else:
        return vkep(r, ms, z=z)


def Tdisk_D20(
    r: np.ndarray,
    z: np.ndarray,
    T0mid: float,
    qmid: float,
    T0atm: float,
    qatm: float,
    z0: float,
    alpha: float,
    beta: float,
    r0: float = 100.0,
):
    """

    Parameters
    ----------
    r : np.ndarray
        Radius in an arbitrary unit.
    z : np.ndarray
        Height in the same unit as ``r``.

    """
    Tatm = T0atm * (r / r0) ** (-qatm)
    Tmid = T0mid * (r / r0) ** (-qmid)
    zq = z0 * (r / r0) ** beta

    tanh = np.tanh((z - alpha * zq) / zq)
    T4 = Tmid**4.0 + 0.5 * (1.0 + tanh) * Tatm**4.0
    return T4**0.25


# sky deprojection
def sky_to_local(
    x: np.ndarray,
    y: np.ndarray,
    inc: float,
    pa: float,
    fz, zargs, dfz = None,
    inc_max = 85., zarg_lims = None, method = 'krylov'):
    '''
    Conversion from the plane-of-sky coordinate to the local coordinate.

    Parameters
    ----------
    x, y : ndarray
        Plane-of-sky coordinates.
    inc : float
        Inclination angle in radians.
    pa : float
        Position angle in radians.
    method : {'krylov', 'anderson'}
        Method
    '''
    _inc_max_rad = np.radians(inc_max)
    cosi = np.cos(inc)
    cosimax = np.cos(_inc_max_rad)
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
            return (y * np.cos(inc)
                    + fz(np.sqrt(xp*xp + y*y), *zargs) * np.sin(inc) - yp
                    )
        else:
            _jac = 1. / (np.cos(inc) + dfz(xp, y, *zargs) * np.sin(inc))
            return (y * np.cos(inc)
                    + fz(np.sqrt(xp*xp + y*y), *zargs)
                    * np.sin(inc) - yp, np.diag(_jac)
                    )

    # rotation
    if pa != 0.:
        x, y = rotate2d(x, y, pa, coords=False, deg=False)

    # check inclination limit
    if inc < 0.:
        return 0
    if np.abs(cosi) < cosimax:
        return 0

    # deprojection limit
    # only valid for power-law flaring case
    z0, r0, hp = zargs
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

    return x, ydep


# Planck function
def Bv(T: float | np.ndarray, v: float | np.ndarray):
    """
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in K.
    v : float or np.ndarray
        Frequency in GHz.
    """
    v = v * 1.0e9  # GHz --> Hz
    exp = np.exp((hp * v) / (kb * T)) - 1.0
    fterm = (2.0 * hp * v * v * v) / (clight * clight)
    Bv = fterm / exp
    return Bv


# Planck function
def Bvppx(
    T: float | np.ndarray,
    v: float | np.ndarray,
    px: float,
    py: float,
    dist: float = 140.0,
    au: bool = True,
):
    """
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in K.
    v : float or np.ndarray
        Frequency in GHz.
    px, py : float
        Pixel size.
    dist : float
        Distance in pc when ``au`` is True.
    au : bool
        Whether ``px`` and ``py`` are given in au.
    """
    # unit
    v = v * 1.0e9  # GHz --> Hz

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


# Jy/beam
def Bv_Jybeam(T: float | np.ndarray, v: float | np.ndarray, bmaj: float, bmin: float):
    """
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    Parameters
    ----------
    T : float or np.ndarray
        Temperature in K.
    v : float or np.ndarray
        Frequency in GHz.
    bmaj, bmin : float
        Beam size in arcsec.
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

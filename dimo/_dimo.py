# import modules
import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import emcee
import numpy as np
from astropy import constants, units

from .builder import Builder, Builder_SSDisk
from .grid import SubGrid2D
from .mpe import BayesEstimator


# np.random.seed(42); for debug

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


# Disk Model Optimization
class DiMO(object):
    """Fit disk models to image and spectral-cube data.

    Parameters
    ----------
    model : callable
        Model class or factory providing ``get_paramkeys``.
    params_free, params_fixed : dict
        Initial values of free parameters and values of fixed parameters.
    beam : sequence of float, optional
        Beam as [major FWHM, minor FWHM, position angle] in au, au, degrees.
    width : float
        Spectral smoothing FWHM. A negative value disables smoothing.
    dist : float
        Distance to the object in pc.
    dv_mode : {'total', 'thermal'}
        Interpretation of the model line-width profile.
    line : str, optional
        Molecular line identifier.
    iline : int, optional
        Transition selector.
    ilinemode, database : str
        Transition selection mode and molecular database.
    build_args : dict, optional
        Additional keyword arguments passed to thin-model builders.
    sampling : bool
        Whether to sample pixels at approximately half a beam width.
    n_subgrid : int
        Uniform subgrid factor; one disables subgridding.
    n_nest : list of int, optional
        Subdivision factor at each nested-grid level; None disables nesting.
    zstrech : list of float, optional
        Vertical stretch factors for 3D nested grids.
    x_nestlim, y_nestlim, z_nestlim : list, optional
        Explicit nested-grid boundaries.
    xscale, yscale, zscale : float
        Scale factors for nested-grid boundaries.
    rin : float
        Inner radius in au.
    reslim : float
        Default nested-region half-width in pixels.
    cosi_lim : float
        Inclination threshold for adaptive deprojection.
    f_nvbin : float
        Velocity sub-binning factor.
    """

    def __init__(
        self,
        model: Callable[..., Any],
        params_free: dict[str, float],
        params_fixed: dict[str, float],
        beam: Sequence[float] | None = None,
        width: float = -1,
        dist: float = 140.0,
        dv_mode: str = "total",
        line: str | None = None,
        iline: int | None = None,
        ilinemode: str = "index",
        database: str = "lamda",
        build_args: Mapping[str, Any] | None = None,
        sampling: bool = False,
        n_subgrid: int = 1,
        n_nest: list[int] | None = None,
        zstrech: list[float] | None = None,
        x_nestlim: list[list[float]] | None = None,
        y_nestlim: list[list[float]] | None = None,
        z_nestlim: list[list[float]] | None = None,
        xscale: float = 0.5,
        yscale: float = 0.5,
        zscale: float = 0.5,
        rin: float = 1.0,
        reslim: float = 10.0,
        cosi_lim: float = 0.5,
        f_nvbin: float = 0.33,
    ):
        """Initialize model parameters and fitting options."""

        # parameter checks
        try:
            _model = model()
        except Exception:
            _model = model(
                np.arange(0, 3, 1),
                np.arange(0, 3, 1),
                np.arange(0, 3, 1),
                np.arange(0, 3, 1),
            )
        _model_keys = _model.get_paramkeys()
        _input_keys = list(params_free.keys()) + list(params_fixed.keys())
        if sorted(_model_keys) != sorted(_input_keys):
            print("ERROR\tDiMO: input keys do not match model input parameters.")
            print("ERROR\tDiMO: input keys: ")
            print(sorted(_input_keys))
            print("ERROR\tDiMO: input parameters must be as follows:")
            print(sorted(_model_keys))
            return 0

        # set model
        self.model = model

        # set model parameters
        self.model_keys = _model_keys
        self.params_free = params_free
        self.pfree_keys = list(params_free.keys())
        self.params_fixed = params_fixed
        self.pfixed_keys = list(params_fixed.keys())
        _params = merge_dictionaries(params_free, params_fixed)
        self._params_ini = _params
        self.params_ini = list(
            {k: _params[k] for k in _model_keys}.values()
        )  # re-ordered elements

        # beam, etc.
        self.beam = beam
        self.width = width
        self.dist = dist
        self.build_args = {"beam": beam, "dist": dist}
        if build_args is not None:
            self.build_args.update(build_args)

        # gridding
        self.sampling = sampling
        self.n_subgrid = n_subgrid
        self.n_nest = n_nest
        self.zstrech = zstrech
        self.x_nestlim, self.y_nestlim, self.z_nestlim = x_nestlim, y_nestlim, z_nestlim
        self.reslim = reslim
        self.xscale, self.yscale = xscale, yscale

        # molecular line
        self.line = line
        self.iline = iline
        self.ilinemode = ilinemode
        self.database = database

        # optional setting
        self.rin = rin
        self.cosi_lim = cosi_lim
        self.f_nvbin = f_nvbin

    def fit_cube(
        self,
        params: dict[str, float],
        pranges: np.ndarray,
        d: np.ndarray,
        derr: float | np.ndarray,
        axes: Sequence[np.ndarray],
        outname: str = "modelfitter_results",
        nwalkers: int | None = None,
        nrun: int = 2000,
        nburn: int = 1000,
        labels: list[str] = [],
        show_progress: bool = True,
        optimize_ini: bool = False,
        moves: emcee.moves.Move = emcee.moves.WalkMove(),
        symmetric_error: bool = False,
        npool: int = 1,
        f_rand_init: float = 0.1,
    ):
        """Dispatch a cube fit according to the number of coordinate axes.

        Parameters
        ----------
        params : dict
            Parameter values used to identify fitted parameters.
        pranges : np.ndarray
            Lower and upper bounds for each free parameter.
        d : np.ndarray
            Observed spectral cube with axes (velocity, y, x).
        derr : float or np.ndarray
            Uncertainty of the observed data.
        axes : sequence of np.ndarray
            Coordinate axes [x, y, v] or [x, y, z, v].
        outname : str
            Prefix for fit output files.
        nwalkers, nrun, nburn : int, optional
            Number of walkers, sampling steps, and burn-in steps.
        labels : list of str
            Labels for free parameters.
        show_progress, optimize_ini, symmetric_error : bool
            Options forwarded to the sampler.
        moves : emcee.moves.Move
            Proposal move for the sampler.
        npool : int
            Number of worker processes.
        f_rand_init : float
            Scale of the random initial walker positions.
        """

        if len(axes) == 3:
            x, y, v = axes
            self.fit_cube_thin(
                params,
                pranges,
                d,
                derr,
                x,
                y,
                v,
                outname=outname,
                nwalkers=nwalkers,
                nrun=nrun,
                nburn=nburn,
                labels=labels,
                show_progress=show_progress,
                optimize_ini=optimize_ini,
                moves=moves,
                symmetric_error=symmetric_error,
                npool=npool,
                f_rand_init=f_rand_init,
            )
        elif len(axes) == 4:
            x, y, z, v = axes
            self.fit_cube_3Dmodel(
                params,
                pranges,
                d,
                derr,
                x,
                y,
                z,
                v,
                outname=outname,
                nwalkers=nwalkers,
                nrun=nrun,
                nburn=nburn,
                labels=labels,
                show_progress=show_progress,
                optimize_ini=optimize_ini,
                moves=moves,
                symmetric_error=symmetric_error,
                npool=npool,
                f_rand_init=f_rand_init,
            )
        else:
            print("ERROR\tfit_cube: axes must consist of three or four axes.")
            return 0

    # define fitting function
    def fit_cube_thin(
        self,
        params: dict[str, float],
        pranges: np.ndarray,
        d: np.ndarray,
        derr: float | np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        outname: str = "modelfitter_results",
        nwalkers: int | None = None,
        nrun: int = 2000,
        nburn: int = 1000,
        labels: list[str] = [],
        show_progress: bool = True,
        optimize_ini: bool = False,
        moves: emcee.moves.Move = emcee.moves.WalkMove(),
        symmetric_error: bool = False,
        npool: int = 1,
        f_rand_init: float = 0.1,
    ):
        """Fit a single-layer model to a spectral cube.

        Parameters
        ----------
        params : dict
            Parameter values used to identify fitted parameters.
        pranges : np.ndarray
            Lower and upper bounds for free parameters.
        d : np.ndarray
            Observed cube with axes (velocity, y, x).
        derr : float or np.ndarray
            Uncertainty of the observed cube.
        x, y : np.ndarray
            Sky-plane coordinate axes in au.
        v : np.ndarray
            Velocity channels in km/s.
        outname : str
            Prefix for fit output files.
        nwalkers, nrun, nburn : int, optional
            Number of walkers, sampling steps, and burn-in steps.
        labels : list of str
            Labels for free parameters.
        show_progress, optimize_ini, symmetric_error : bool
            Options forwarded to the sampler.
        moves : emcee.moves.Move
            Proposal move for the sampler.
        npool : int
            Number of worker processes.
        f_rand_init : float
            Scale of the random initial walker positions.

        Returns
        -------
        np.ndarray
            Best-fit cube with axes (velocity, y, x).
        """
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        delx = -(x[1] - x[0])  # au
        dely = y[1] - y[0]  # au
        nx = len(x)
        ny = len(y)
        nv = len(v)
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[::smpl_y, ::smpl_x]
                Rbeam_pix = 1.0
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = (
                    np.pi
                    / (4.0 * np.log(2.0))
                    * self.beam[1]
                    * self.beam[0]
                    / delx
                    / dely
                )
                d_smpld = d.copy()
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = 1
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = (
                -0.5
                * np.nansum(
                    (d - model) ** 2 / (derr * derr) + np.log(2.0 * np.pi * derr * derr)
                )
                / Rbeam_pix
            )
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # labels
        if len(labels) != len(params):
            labels = self.pfree_keys

        # gridding and fitting function
        if self.n_subgrid > 1:
            # fitting function
            def fitfunc(x, y, v, *params):
                # safety net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros((nv, ny, nx))[
                        :, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x
                    ]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                )  # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcube = model.build_cube_subgrid(
                    x, y, v, self.n_subgrid, **self.build_args
                )

                return modelcube[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]
        else:
            # fitting function
            def fitfunc(x, y, v, *params):
                # safety net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros((nv, ny, nx))[
                        :, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x
                    ]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                )  # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcube = model.build_nested_cube(
                    x,
                    y,
                    v,
                    nnest=self.n_nest,
                    xlim=self.x_nestlim,
                    ylim=self.y_nestlim,
                    reslim=self.reslim,
                    **self.build_args,
                )

                return modelcube[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]

        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([x, y, v], d_smpld, derr, fitfunc, lnlike=lnlike)
        BE.run_mcmc(
            p0,
            pranges,
            outname=outname,
            nwalkers=nwalkers,
            nrun=nrun,
            nburn=nburn,
            labels=labels,
            show_progress=show_progress,
            optimize_ini=optimize_ini,
            moves=moves,
            symmetric_error=symmetric_error,
            npool=npool,
            f_rand_init=0.1,
        )
        self.pfit = BE.pfit.copy()
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]
        self.writeout_fitres(outname, BE.criterion)

        # best solution
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(x, y, v, *self.popt)
        self.modelcube = modelcube

        return modelcube

    def fit_cube_ssdisk(
        self,
        params: dict[str, float],
        pranges: np.ndarray,
        d: np.ndarray,
        derr: float | np.ndarray,
        r: np.ndarray,
        phi: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        v: np.ndarray,
        outname: str = "fit_SSDisk",
        nwalkers: int | None = None,
        nrun: int = 2000,
        nburn: int = 1000,
        labels: list[str] = [],
        show_progress: bool = True,
        optimize_ini: bool = False,
        moves: emcee.moves.Move = emcee.moves.StretchMove(),
        symmetric_error: bool = False,
        npool: int = 1,
        f_rand_init: float = 1.0,
        show_results: bool = True,
    ):
        """Fit a self-similar disk model to a spectral cube.

        Parameters
        ----------
        params : dict
            Parameter values used to identify fitted parameters.
        pranges : np.ndarray
            Lower and upper bounds for free parameters.
        d : np.ndarray
            Observed cube with axes (velocity, y, x).
        derr : float or np.ndarray
            Uncertainty of the observed cube.
        r, phi : np.ndarray
            Radial and azimuthal axes for the model grid.
        x, y : np.ndarray
            Sky-plane coordinate axes in au.
        v : np.ndarray
            Velocity channels in km/s.
        outname : str
            Prefix for fit output files.
        nwalkers, nrun, nburn : int, optional
            Number of walkers, sampling steps, and burn-in steps.
        labels : list of str
            Labels for free parameters.
        show_progress, optimize_ini, symmetric_error, show_results : bool
            Options forwarded to the sampler.
        moves : emcee.moves.Move
            Proposal move for the sampler.
        npool : int
            Number of worker processes.
        f_rand_init : float
            Scale of the random initial walker positions.

        Returns
        -------
        np.ndarray
            Best-fit cube with axes (velocity, y, x).
        """

        axes_model = [r, phi]
        axes_sky = [x, y, v]
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny = len(x), len(y)
        # sampling step
        delx = -(x[1] - x[0])  # au
        dely = y[1] - y[0]  # au
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]
                Rbeam_pix = 1.0
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = (
                    np.pi
                    / (4.0 * np.log(2.0))
                    * self.beam[1]
                    * self.beam[0]
                    / delx
                    / dely
                )
                d_smpld = d.copy()
        else:
            Rbeam_pix = 1.0
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            mdl = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = (
                -0.5
                * np.nansum(
                    (d - mdl) ** 2 / (derr * derr) + np.log(2.0 * np.pi * derr * derr)
                )
                / Rbeam_pix
            )

            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # labels
        if len(labels) != len(params):
            labels = self.pfree_keys

        # setup model
        model = Builder_SSDisk(
            self.model,
            axes_model,
            axes_sky,
            xlim=None,
            ylim=None,
            nsub=self.n_nest,
            reslim=self.reslim,
            beam=self.beam,
            coordinate_type="polar",
        )
        model.skygrid_info()

        # make grid
        model.set_model(self._params_ini)
        model.visualize_grid(
            showfig=False, savefig=True, cmap="coolwarm", outname="ssdisk_grid"
        )

        # define fitting function
        axes = [x, y, v]

        def fitfunc(x, y, v, *params):
            # safty net
            if not np.all(
                (pranges[0] < np.array([*params])) * (np.array([*params]) < pranges[1])
            ):
                return np.zeros((len(v), ny, nx))[
                    :, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x
                ]

            # merge free parameters to fixed parameters
            params_free = dict(zip(self.pfree_keys, [*params]))
            _params_full = merge_dictionaries(params_free, self.params_fixed)

            # update parameters
            _model = copy.deepcopy(model)  # to make sure parallel calculations go well
            _model.set_model(_params_full)

            # cube on the original grid
            modelcube = _model.build_cube()

            return modelcube[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]

        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator(axes, d_smpld, derr, fitfunc, lnlike=lnlike)
        BE.run_mcmc(
            p0,
            pranges,
            outname=outname,
            nwalkers=nwalkers,
            nrun=nrun,
            nburn=nburn,
            labels=labels,
            show_progress=show_progress,
            optimize_ini=optimize_ini,
            moves=moves,
            symmetric_error=symmetric_error,
            npool=npool,
            f_rand_init=f_rand_init,
            show_results=show_results,
        )
        self.pfit = BE.pfit.copy()
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]

        # best solution
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(x, y, v, *self.popt)
        self.modelcube = modelcube

        self.writeout_fitres(outname, BE.criterion)

        return modelcube

    # define fitting function
    def fit_cube_3Dmodel(
        self,
        params: dict[str, float],
        pranges: np.ndarray,
        d: np.ndarray,
        derr: float | np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        Tcmb: float = 2.73,
        f0: float = 230.0,
        dv_mode: str = "total",
        pterm: bool = False,
        builder: type[Builder] = Builder,
        outname: str = "modelfitter_results",
        nwalkers: int | None = None,
        nrun: int = 2000,
        nburn: int = 1000,
        labels: list[str] = [],
        show_progress: bool = True,
        optimize_ini: bool = False,
        moves: emcee.moves.Move = emcee.moves.StretchMove(),
        symmetric_error: bool = False,
        npool: int = 1,
        f_rand_init: float = 1.0,
        show_results: bool = True,
        restart: bool = False,
    ):
        """Fit a three-dimensional model to a spectral cube.

        Parameters
        ----------
        params : dict
            Parameter values used to identify fitted parameters.
        pranges : np.ndarray
            Lower and upper bounds for free parameters.
        d : np.ndarray
            Observed cube with axes (velocity, y, x).
        derr : float or np.ndarray
            Uncertainty of the observed cube.
        x, y, z : np.ndarray
            Coordinate axes of the model grid in au.
        v : np.ndarray
            Velocity channels in km/s.
        Tcmb : float
            Background temperature in K.
        f0 : float
            Rest frequency in GHz.
        dv_mode : str
            Line-width mode forwarded to the builder.
        pterm : bool
            Whether to include the pressure-gradient term.
        builder : type[Builder]
            Builder class for the three-dimensional model.
        outname : str
            Prefix for fit output files.
        nwalkers, nrun, nburn : int, optional
            Number of walkers, sampling steps, and burn-in steps.
        labels : list of str
            Labels for free parameters.
        show_progress, optimize_ini, symmetric_error, show_results : bool
            Options forwarded to the sampler.
        moves : emcee.moves.Move
            Proposal move for the sampler.
        npool : int
            Number of worker processes.
        f_rand_init : float
            Scale of the random initial walker positions.
        restart : bool
            Whether to resume a saved sampling run.

        Returns
        -------
        np.ndarray
            Best-fit cube with axes (velocity, y, x).
        """
        axes = [x, y, z, v]
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny = len(x), len(y)
        # sampling step
        delx = -(x[1] - x[0])  # au
        dely = y[1] - y[0]  # au
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]
                Rbeam_pix = 1.0
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = (
                    np.pi
                    / (4.0 * np.log(2.0))
                    * self.beam[1]
                    * self.beam[0]
                    / delx
                    / dely
                )
                d_smpld = d.copy()
        else:
            Rbeam_pix = 1.0
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            mdl = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = (
                -0.5
                * np.nansum(
                    (d - mdl) ** 2 / (derr * derr) + np.log(2.0 * np.pi * derr * derr)
                )
                / Rbeam_pix
            )

            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # fitting function
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(x, y, nsub=self.n_subgrid)
            _x, _y = subgrid.x_sub, subgrid.y_sub

            # define fitting function
            def fitfunc(x, y, z, v, *params):
                # safty net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros((len(v), ny, nx))[
                        :, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x
                    ]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)

                # update parameters
                _model = copy.deepcopy(
                    model
                )  # to make sure parallel calculations go well
                _model.set_model(_params_full)

                # renew grid
                if renew_grid:
                    _model.renest()
                    _model.deproject_grid()

                # cube on the original grid
                modelcube = _model.build_cube(
                    Tcmb=Tcmb, f0=f0, dist=self.dist, dv_mode=dv_mode, pterm=pterm
                )
                modelcube = subgrid.binning_onsubgrid_layered(modelcube)

                return modelcube[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]
        else:
            _x = x
            _y = y

            # define fitting function
            def fitfunc(x, y, z, v, *params):
                # safty net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros((len(v), ny, nx))[
                        :, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x
                    ]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)

                # update parameters
                _model = copy.deepcopy(
                    model
                )  # to make sure parallel calculations go well
                _model.set_model(_params_full)

                # renew grid
                if renew_grid:
                    _model.renest()
                    _model.deproject_grid()

                # cube on the original grid
                modelcube = _model.build_cube(
                    Tcmb=Tcmb, f0=f0, dist=self.dist, dv_mode=dv_mode, pterm=pterm
                )

                return modelcube[:, smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]

        # labels
        if len(labels) != len(params):
            labels = self.pfree_keys

        # setup model
        model = builder(
            _x,
            _y,
            z,
            v,
            self.model,
            nsub=self.n_nest,
            zstrech=self.zstrech,
            reslim=self.reslim,
            beam=self.beam,
            width=self.width,
            f_nvbin=self.f_nvbin,
            line=self.line,
            iline=self.iline,
            ilinemode=self.ilinemode,
            database=self.database,
            rin=self.rin,
            adoptive_zaxis=True,
            cosi_lim=self.cosi_lim,
        )
        model.grid.gridinfo()

        # renew grid every fit or not
        if any([i in self.pfree_keys for i in ["dx0", "dy0", "inc", "pa"]]):
            renew_grid = True
        else:
            print("Grid-saved mode")
            renew_grid = False

        # make grid
        model.set_model(self._params_ini)
        model.renest()
        model.deproject_grid()
        model.show_model_sideview(
            showfig=False,
            savefig=True,
            cmap="coolwarm",
            dv_mode=dv_mode,
            pterm=pterm,
            outname="model_sideview_ini",
        )

        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator(axes, d_smpld, derr, fitfunc, lnlike=lnlike)
        BE.run_mcmc(
            p0,
            pranges,
            outname=outname,
            nwalkers=nwalkers,
            nrun=nrun,
            nburn=nburn,
            labels=labels,
            show_progress=show_progress,
            optimize_ini=optimize_ini,
            moves=moves,
            symmetric_error=symmetric_error,
            npool=npool,
            f_rand_init=f_rand_init,
            show_results=show_results,
            restart=restart,
        )
        self.pfit = BE.pfit.copy()
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]

        # best solution
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(_x, _y, z, v, *self.popt)
        self.modelcube = modelcube
        self.writeout_fitres(outname, BE.criterion)

        # make a side view
        _params_full = merge_dictionaries(
            dict(zip(self.pfree_keys, [*self.popt])), self.params_fixed
        )
        model.set_model(_params_full)
        model.renest()
        model.deproject_grid()
        model.show_model_sideview(
            showfig=False,
            savefig=True,
            cmap="coolwarm",
            dv_mode=dv_mode,
            pterm=pterm,
            outname="model_sideview_fit",
        )

        return modelcube

    # define fitting function
    def fit_cont(
        self,
        params: dict[str, float],
        pranges: np.ndarray,
        d: np.ndarray,
        derr: float | np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        outname: str = "modelfitter_results",
        nwalkers: int | None = None,
        nrun: int = 2000,
        nburn: int = 1000,
        labels: list[str] = [],
        show_progress: bool = True,
        optimize_ini: bool = False,
        moves: emcee.moves.Move = emcee.moves.WalkMove(),
        symmetric_error: bool = False,
        npool: int = 1,
    ):
        """Fit a single-layer model to a continuum image.

        Parameters
        ----------
        params : dict
            Parameter values used to identify fitted parameters.
        pranges : np.ndarray
            Lower and upper bounds for free parameters.
        d : np.ndarray
            Observed image with axes (y, x).
        derr : float or np.ndarray
            Uncertainty of the observed image.
        x, y : np.ndarray
            Sky-plane coordinate axes in au.
        outname : str
            Prefix for fit output files.
        nwalkers, nrun, nburn : int, optional
            Number of walkers, sampling steps, and burn-in steps.
        labels : list of str
            Labels for free parameters.
        show_progress, optimize_ini, symmetric_error : bool
            Options forwarded to the sampler.
        moves : emcee.moves.Move
            Proposal move for the sampler.
        npool : int
            Number of worker processes.

        Returns
        -------
        np.ndarray
            Best-fit continuum image with axes (y, x).
        """
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        delx = -(x[1] - x[0])  # au
        dely = y[1] - y[0]  # au
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[::smpl_y, ::smpl_x]
                Rbeam_pix = 1.0
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = (
                    np.pi
                    / (4.0 * np.log(2.0))
                    * self.beam[1]
                    * self.beam[0]
                    / delx
                    / dely
                )
                d_smpld = d.copy()
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = 1
            d_smpld = d.copy()

        # labels
        if len(labels) != len(params):
            labels = self.pfree_keys

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = (
                -0.5
                * np.nansum(
                    (d - model) ** 2 / (derr * derr)
                    + np.log(2.0 * np.pi * derr * derr)
                )
                / Rbeam_pix
            )
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # nested grid
        if self.n_subgrid > 1:
            # fitting function
            def fitfunc(x, y, *params):
                # safty net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros_like(d_smpld)

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                )  # reordered elements

                # build model cube
                model = self.model(*params_full)
                modelcont = model.build_cont_subgrid(
                    x, y, nsub=self.n_subgrid, **self.build_args
                )
                return modelcont[smpl_y // 2 :: smpl_y, smpl_x // 2 :: smpl_x]
        else:

            def fitfunc(x, y, *params):
                # safty net
                if not np.all(
                    (pranges[0] < np.array([*params]))
                    * (np.array([*params]) < pranges[1])
                ):
                    return np.zeros_like(d_smpld)

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                )  # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelim = model.build_nested_cont(
                    x,
                    y,
                    nnest=self.n_nest,
                    xlim=self.x_nestlim,
                    ylim=self.y_nestlim,
                    reslim=self.reslim,
                    **self.build_args,
                )

                return modelim[::smpl_y, ::smpl_x]

        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([x, y], d_smpld, derr, fitfunc, lnlike=lnlike)
        BE.run_mcmc(
            p0,
            pranges,
            outname=outname,
            nwalkers=nwalkers,
            nrun=nrun,
            nburn=nburn,
            labels=labels,
            show_progress=show_progress,
            optimize_ini=optimize_ini,
            moves=moves,
            symmetric_error=symmetric_error,
            npool=npool,
            f_rand_init=0.1,
        )
        self.pfit = BE.pfit
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]
        self.writeout_fitres(outname, BE.criterion)

        # best solution
        smpl_y, smpl_x = 1, 1
        modelim = fitfunc(x, y, *self.popt)
        self.modelim = modelim
        return modelim

    def writeout_fitres(
        self,
        outname: str,
        criterion: Mapping[str, float] | None = None,
        credible_interval: float = 0.68,
    ):
        """Write fitted parameter values and fit criteria to a text file.

        Parameters
        ----------
        outname : str
            Prefix of the ``_results.txt`` output file.
        criterion : mapping of str to float, optional
            Fit criteria to append to the results.
        credible_interval : float
            Probability mass of the reported credible interval.
        """
        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list({k: _params_full[k] for k in self.model_keys}.values())

        # overwrite default output
        outtxtfile = outname + "_results.txt"
        ne, _ = self.pfit.shape
        labels = self.model_keys
        perr_indx = dict(
            zip(self.pfree_keys, [*np.arange(0, len(self.pfree_keys))])
        )
        with open(outtxtfile, "+w") as f:
            if ne == 2:
                f.write("# param mean sigma\n")
                # make a full perr list
                for i, k in enumerate(self.model_keys):
                    if k in self.pfree_keys:
                        indx = perr_indx[k]
                        f.write(
                            "%s %13.6e %13.6e\n"
                            % (labels[i], params_full[i], self.perr[indx])
                        )
                    else:
                        f.write(
                            "%s %13.6e %13.6e\n"
                            % (labels[i], params_full[i], 0.0))
            elif ne == 3:
                f.write(
                    "# param 50th %.fth %.fth\n"
                    % (50 * (1.0 - credible_interval), 50 * (1.0 + credible_interval))
                )
                for i, k in enumerate(self.model_keys):
                    if k in self.pfree_keys:
                        indx = perr_indx[k]
                        f.write(
                            "%s %13.6e %13.6e %13.6e\n"
                            % (
                                labels[i],
                                params_full[i],
                                self.perr[0, indx],
                                self.perr[1, indx],
                            )
                        )
                    else:
                        f.write(
                            "%s %13.6e %13.6e %13.6e\n"
                            % (labels[i], params_full[i], 0.0, 0.0)
                        )
            if criterion is not None:
                f.write("# criterion")
                for k in criterion.keys():
                    f.write("\n# %s %.4f" % (k, criterion[k]))


def merge_dictionaries(
    dict1: dict[str, Any], dict2: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge two dictionaries, with entries from ``dict2`` taking precedence.

    Parameters
    ----------
    dict1 : dict
        Base values.
    dict2 : mapping
        Values to overlay.

    Returns
    -------
    dict
        A new dictionary containing both sets of values.
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def mathlabels(pylabels: Iterable[str]) -> list[str]:
    """Convert known parameter names to mathematical plot labels.

    Parameters
    ----------
    pylabels : iterable of str
        Parameter names to format.

    Returns
    -------
    list of str
        Formatted labels, or original names when no label is defined.
    """
    labels = {
        "Td0": r"$T_{\mathrm{d},0}$",
        "qd": r"$q_\mathrm{d}$",
        "log_tau_dc": r"$\log \tau_\mathrm{c,d}$",
        "rc_d": r"$R_\mathrm{c,d}$",
        "gamma_d": r"$\gamma_\mathrm{d}$",
        "Tg0": r"$T_{\mathrm{g},0}$",
        "qg": r"$q_\mathrm{g}$",
        "log_N_gc": r"$\log N_\mathrm{c,g}$",
        "log_tau_gc": r"$\log \tau_\mathrm{c,g}$",
        "Mg": r"$M_\mathrm{g}$",
        "log_Xmol": r"$\log~X_\mathrm{mol}$",
        "rc_g": r"$R_\mathrm{c,g}$",
        "gamma_g": r"$\gamma_\mathrm{g}$",
        "z0": r"$z_0$",
        "pz": r"$p_z$",
        "h0": r"$H_0$",
        "ph": r"$p_H$",
        "z0l": r"$z_{\mathrm{l},0}$",
        "z0u": r"$z_{\mathrm{u},0}$",
        "pzl": r"$p_{z,\mathrm{l}}$",
        "pzu": r"$p_{z,\mathrm{u}}$",
        "inc": r"$i$",
        "pa": r"$PA$",
        "ms": r"$M_\ast$",
        "vsys": r"$v_\mathrm{sys}$",
        "dx0": r"$\delta x_0$",
        "dy0": r"$\delta y_0$",
        "r0": r"$R_0$",
        "dv": r"$\Delta v$",
        "pdv": r"$p_{\Delta v}$",
        "delv": r"$\Delta v$",
    }
    keys = labels.keys()

    return [labels[i] if i in keys else i for i in pylabels]

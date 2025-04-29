# import modules
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units
import emcee
from datetime import datetime
from dataclasses import dataclass
from typing import Callable

from .models import MultiLayerDisk, ThreeLayerDisk, SingleLayerDisk, SSDisk
from .grid import Nested2DGrid, SubGrid2D, Nested3DObsGrid
from .mpe import BayesEstimator
from .builder import Builder, Builder_SSDisk


np.random.seed(42)


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


# Fitters
class FitThinModel(object):
    """docstring for Fitter

    Args
    ----
     model: Model you select. Must be an object.
     params_free (dict): free parameters.
     params_fixed (dict): Fixed parameters
    """
    def __init__(self, model, params_free, params_fixed, 
        beam = None, dist = 140., build_args = None, 
        sampling = False, n_subgrid = 3,
        n_nstgrid = 1, xscale = 0.5, yscale = 0.5):
        super(FitThinModel, self).__init__()
        # parameter checks
        _model = model()
        _model_keys = _model.get_paramkeys()
        _input_keys = list(params_free.keys()) + list(params_fixed.keys())
        if sorted(_model_keys) != sorted(_input_keys):
            print('ERROR\tModelFitter: input keys do not match model input parameters.')
            print('ERROR\tModelFitter: input parameters must be as follows:')
            print(_model_keys)
            return 0

        # set model
        self.model_keys = _model_keys
        self.params_free = params_free
        self.pfree_keys = list(params_free.keys())
        self.params_fixed = params_fixed
        self.pfixed_keys = list(params_fixed.keys())
        _params = merge_dictionaries(params_free, params_fixed)
        self.params_ini = list(
            {k: _params[k] for k in _model_keys}.values()
            ) # re-ordered elements
        self.model = model #(*self.params_ini)
        self.beam = beam
        self.dist = dist
        self.build_args = [beam, dist] + build_args if build_args is not None\
        else [beam, dist]
        self.sampling = sampling
        self.n_subgrid = n_subgrid
        self.n_nstgrid = n_nstgrid
        self.xscale, self.yscale = xscale, yscale


    # define fitting function
    def fit_cube(self, params, pranges, d, derr, xx, yy, v,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1,):
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        delx = - (xx[0,1] - xx[0,0]) / self.dist # arcsec
        dely = (yy[1,0] - yy[0,0]) / self.dist   # arcsec
        if self.sampling:
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
            Rbeam_pix = 1.
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-model)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # gridding
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(xx, yy)
            xx_sub, yy_sub = subgrid.xx_sub, subgrid.yy_sub

            # fitting function
            def fitfuc(xx, yy, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), xx.shape[1], xx.shape[0])
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcube = model.build_cube_subgrid(
                    xx, yy, v, self.n_subgrid, 
                    *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        elif self.n_nstgrid > 1:
            nstgrid = Nested2DGrid(xx, yy)
            xlim = [-np.nanmax(xx) * self.xscale, np.nanmax(xx) * self.xscale]
            ylim = [-np.nanmax(yy) * self.yscale, np.nanmax(yy) * self.yscale]
            xx_sub, yy_sub = nstgrid.nest(xlim, ylim, self.n_subgrid)
            if self.beam is not None:
                xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
                    self.beam[0] * self.dist * 1.3, self.beam[0] * self.dist * 1.3)
                # fitting function
                def fitfuc(xx, yy, v, *params):
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(
                            (len(v), xx.shape[1], xx.shape[0])
                            )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelcube = model.build_cube(xx, yy, v, *self.build_args)
                    # cube on the nested grid
                    modelcube_sub = model.build_cube(xx_sub, yy_sub, 
                        v, *self.build_args)[:, yi:-yi, xi:-xi]
                    # replace
                    for i in range(len(v)):
                        modelcube[i, yi0:-yi0, xi0:-xi0] = \
                        nstgrid.binning_onsubgrid(modelcube_sub[i,:,:])
                    return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
            else:
                # fitting function
                def fitfuc(xx, yy, v, *params):
                    where_sub = nstgrid.where_subgrid()
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(
                            (len(v), xx.shape[1], xx.shape[0])
                            )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements
                    #del params_free, _params_full # release memory

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelcube = model.build_cube(xx, yy, v, *self.build_args)
                    # cube on the nested grid
                    modelcube_sub = model.build_cube(xx_sub, yy_sub, 
                        v, *self.build_args)
                    # replace
                    for i in range(len(v)):
                        modelcube[i,where_sub[0], where_sub[1]] = \
                        nstgrid.binning_onsubgrid(modelcube_sub[i,:,:]).ravel()
                    return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        else:
            def fitfuc(xx, yy, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), xx.shape[1], xx.shape[0])
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelcube = model.build_cube(xx, yy, v, *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([xx, yy, v], d_smpld, derr, fitfuc,
            lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init=0.1)
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelcube = fitfuc(xx, yy, v, *self.popt)
        self.modelcube = modelcube
        return modelcube



    # define fitting function
    def fit_cont(self, params, pranges, d, derr, xx, yy,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1,):
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        delx = - (xx[0,1] - xx[0,0]) / self.dist # arcsec
        dely = (yy[1,0] - yy[0,0]) / self.dist # arcsec
        if self.sampling:
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[::smpl_y, ::smpl_x]
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
            d_smpld = d.copy()

        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-model)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # nested grid
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(xx, yy)
            xx_sub, yy_sub = subgrid.xx_sub, subgrid.yy_sub

            # fitting function
            def fitfuc(xx, yy, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(xx.shape)[smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcont = model.build_cont_subgrid(xx, yy, *self.build_args)
                return modelcont[smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        elif self.n_nstgrid > 1:
            nstgrid = Nested2DGrid(xx, yy)
            xlim = [-np.nanmax(xx) * self.xscale, np.nanmax(xx) * self.xscale]
            ylim = [-np.nanmax(yy) * self.yscale, np.nanmax(yy) * self.yscale]
            xx_sub, yy_sub = nstgrid.nest(xlim, ylim, self.n_subgrid)
            if self.beam is not None:
                xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
                    self.beam[0] * self.dist * 1.3, self.beam[0] * self.dist * 1.3)
                # fitting function
                def fitfuc(xx, yy, *params):
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelim = model.build_cont(xx, yy, *self.build_args)
                    # cube on the nested grid
                    modelim_sub = model.build_cont(xx_sub, yy_sub, 
                        *self.build_args)[yi:-yi, xi:-xi]
                    # replace
                    modelim[yi0:-yi0, xi0:-xi0] = \
                    nstgrid.binning_onsubgrid(modelim_sub)
                    return modelim[::smpl_y, ::smpl_x]
            else:
                # fitting function
                def fitfuc(xx, yy, *params):
                    where_sub = nstgrid.where_subgrid()
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements
                    #del params_free, _params_full # release memory

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelim = model.build_cont(xx, yy, *self.build_args)
                    # cube on the nested grid
                    modelim_sub = model.build_cont(xx_sub, yy_sub, 
                        *self.build_args)
                    # replace
                    modelim[where_sub] = \
                    nstgrid.binning_onsubgrid(modelim_sub).ravel()
                    return modelcube[::smpl_y, ::smpl_x]
        else:
            def fitfuc(xx, yy, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelim = model.build_cont(xx, yy, *self.build_args)
                return modelim[::smpl_y, ::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([xx, yy], d_smpld, derr, fitfuc,
            lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init=0.1)
        self.pfit = BE.pfit
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelim = fitfuc(xx, yy, *self.popt)
        self.modelim = modelim
        return modelim



# Disk Fitter
class Fit3DModel(object):
    """docstring for Fitter

    Args
    ----
     model: Model you select. Must be an object.
     params_free (dict): free parameters.
     params_fixed (dict): Fixed parameters
    """
    def __init__(self, model, params_free, params_fixed, 
        beam = None, dist = 140., build_args = None, 
        sampling = False, n_subgrid = 1,
        n_nstgrid = 1, xscale = 0.5, yscale = 0.5, zscale = 0.5):
        super(DiMO, self).__init__()
        '''
        model, params_free, params_fixed, 
            beam = beam, dist = dist, build_args = build_args, 
            sampling = sampling, n_subgrid = n_subgrid, n_nstgrid = n_nstgrid, 
            xscale = xscale, yscale = yscale, zscale = zscale
        '''
        #super(Fit3DModel, self).__init__()


    # define fitting function
    def fit_cube(self, params, pranges, d, derr, x, y, z, v,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1,):
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny, nz = len(x), len(y), len(z)
        # sampling step
        delx = - (x[1] - x[0]) / self.dist # arcsec
        dely = (y[1] - y[0]) / self.dist   # arcsec
        if self.sampling:
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
            Rbeam_pix = 1.
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-model)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # gridding
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(x, y)
            x_sub, y_sub = subgrid.x_sub, subgrid.y_sub
            xx, yy, zz = np.meshgrid(x_sub, y_sub, z, indexing = 'ij')

            # fitting function
            def fitfuc(xx, yy, zz, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), ny, nx)
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*_params_full)
                # cube on the original grid
                modelcube = model.build_cube(
                    xx, yy, zz, v, *self.build_args)
                modelcube = subgrid.binning_onsubgrid_layered(modelcube)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        else:
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
            def fitfuc(xx, yy, zz, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), ny, nx)
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelcube = model.build_cube(xx, yy, zz, v, *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([xx, yy, zz, v], d_smpld, derr, fitfuc,
            lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init=0.1)
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelcube = fitfuc(xx, yy, zz, v, *self.popt)
        self.modelcube = modelcube
        return modelcube


# Disk Model Optimization
class DiMO(object):#, FitThinModel):
    """docstring for Fitter

    Args
    ----
     model: Model you select. Must be an object.
     params_free (dict): free parameters.
     params_fixed (dict): Fixed parameters
    """
    def __init__(self, model, params_free, params_fixed, 
        beam = None, dist = 140., line = None, iline = None, dv_mode = 'total',
        build_args = None, sampling = False, n_subgrid = 1,
        n_nest = None, zstrech = None, x_nestlim = None, y_nestlim = None, z_nestlim = None,
        xscale = 0.5, yscale = 0.5, zscale = 0.5, rin = 1., reslim = 10.):

        # parameter checks
        try:
            _model = model()
        except:
            _model = model(np.arange(0,3,1),np.arange(0,3,1),np.arange(0,3,1),np.arange(0,3,1))
        _model_keys = _model.get_paramkeys()
        _input_keys = list(params_free.keys()) + list(params_fixed.keys())
        if sorted(_model_keys) != sorted(_input_keys):
            print('ERROR\tModelFitter: input keys do not match model input parameters.')
            print('ERROR\tModelFitter: input parameters must be as follows:')
            print(_model_keys)
            return 0

        # set model
        self.model_keys = _model_keys
        self.params_free = params_free
        self.pfree_keys = list(params_free.keys())
        self.params_fixed = params_fixed
        self.pfixed_keys = list(params_fixed.keys())
        _params = merge_dictionaries(params_free, params_fixed)
        self._params_ini = _params
        self.params_ini = list(
            {k: _params[k] for k in _model_keys}.values()
            ) # re-ordered elements
        self.model = model #(*self.params_ini)
        self.beam = beam
        self.dist = dist
        self.build_args = [beam, dist] + build_args if build_args is not None\
        else [beam, dist]
        self.sampling = sampling
        self.n_subgrid = n_subgrid
        self.n_nest = n_nest
        self.zstrech = zstrech
        self.x_nestlim, self.y_nestlim, self.z_nestlim = x_nestlim, y_nestlim, z_nestlim
        self.xscale, self.yscale = xscale, yscale
        self.line = line
        self.iline = iline
        self.rin = rin
        self.reslim = reslim


    def fit_cube(self, params: dict, pranges:list, 
        d: np.ndarray, derr: float or np.ndarray, axes: list,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1, f_rand_init = 0.1):
        '''

        Parameters
        ----------
        params (dict):
        pranges (list):
        d (array):
        derr (float or array):
        axes (list): A list containing all axes. Must be [x, y, v] or [x, y, z, v].
        '''

        if len(axes) == 3:
            x, y, v = axes
        elif len(axes) == 4:
            x, y, z, v = axes
            self.fit_cube_3Dmodel(params, pranges, d, derr, x, y, z, v,
                outname = outname, nwalkers = nwalkers, nrun = nrun, nburn = nburn,
                labels = [], show_progress = show_progress, optimize_ini = optimize_ini, 
                moves = moves, symmetric_error = symmetric_error, npool = npool,
                f_rand_init = f_rand_init)
        else:
            print('ERROR\tfit_cube: axes must consist of three or four axes.')
            return 0


    def fit_cube_ssdisk(self, 
        params, pranges, 
        d, derr, 
        r, phi, x, y, v,
        outname = 'fit_SSDisk', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves = emcee.moves.StretchMove(), 
        symmetric_error=False, npool = 1, f_rand_init = 1.,
        show_results = True):

        axes_model = [r, phi]
        axes_sky = [x, y, v]
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny, nv = len(x), len(y), len(v)
        # sampling step
        delx = - (x[1] - x[0]) # au / self.dist # arcsec
        dely = (y[1] - y[0]) # au / self.dist   # arcsec
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                Rbeam_pix = 1.
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
                d_smpld = d.copy()
        else:
            Rbeam_pix = 1.
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            mdl = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-mdl)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix

            if np.isnan(exp):
                return -np.inf
            else:
                return exp


        # labels
        if len(labels) != len(params): labels = self.pfree_keys


        # setup model
        model = Builder_SSDisk(self.model,
            axes_model, axes_sky, 
            xlim = None, ylim = None,
            nsub = self.n_nest, reslim = self.reslim,
            beam = self.beam, coordinate_type = 'polar')
        model.skygrid_info()


        # make grid
        model.set_model(self._params_ini)
        model.visualize_grid(
            showfig = False, savefig = True, 
            cmap = 'coolwarm', outname = 'ssdisk_grid')


        # define fitting function
        axes = [x, y, v]
        def fitfunc(x, y, v, *params):
            # safty net
            if np.all((pranges[0] < np.array([*params])) \
                * (np.array([*params]) < pranges[1])) == False:
                return np.zeros(
                    (len(v), ny, nx)
                    )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


            # merge free parameters to fixed parameters
            params_free = dict(zip(self.pfree_keys, [*params]))
            _params_full = merge_dictionaries(params_free, self.params_fixed)

            # update parameters
            _model = copy.deepcopy(model) # to make sure parallel calculations go well
            _model.set_model(_params_full)

            # cube on the original grid
            modelcube = _model.build_cube()

            return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator(axes, d_smpld, derr, fitfunc, lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init = f_rand_init,
            show_results = show_results)
        self.pfit = BE.pfit.copy()
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(x, y, v, *self.popt)
        #modelcube = fitfunc(x, y, z, v, *list(self.params_free.values()))
        self.modelcube = modelcube

        self.writeout_fitres(outname, BE.criterion)

        return modelcube


    # define fitting function
    def fit_cube_3Dmodel(self, params, pranges, d, derr, x, y, z, v,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1, f_rand_init = 1.):
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny, nz = len(x), len(y), len(z)
        # sampling step
        delx = - (x[1] - x[0]) / self.dist # arcsec
        dely = (y[1] - y[0]) / self.dist   # arcsec
        if self.sampling:
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
            Rbeam_pix = 1.
        else:
            smpl_x, smpl_y = 1, 1
            Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            model = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-model)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix
            if np.isnan(exp):
                return -np.inf
            else:
                return exp

        print('Rbeam_pix %.2f'%Rbeam_pix)

        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # gridding
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(x, y)
            x_sub, y_sub = subgrid.x_sub, subgrid.y_sub
            xx, yy, zz = np.meshgrid(x_sub, y_sub, z, indexing = 'ij')
            axes = [xx, yy, zz, v]
            # fitting function
            def fitfunc(xx, yy, zz, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), ny, nx)
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*_params_full)
                # cube on the original grid
                modelcube = model.build_cube(
                    xx, yy, zz, v, *self.build_args)
                modelcube = subgrid.binning_onsubgrid_layered(modelcube)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        elif self.n_nest is not None:
            axes = [x, y, z, v]
            def fitfunc(x, y, z, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), ny, nx)
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcube = model.build_nested_cube(
                    x.copy(), y.copy(), z.copy(), v.copy(), 
                    self.x_nestlim.copy(), self.y_nestlim.copy(), 
                    self.z_nestlim.copy(), self.n_nest,
                    *self.build_args)

                print('Tg %.f K, Mean resid %.2f'%(model.Tg0,
                    np.sqrt(np.nanmean((d_smpld - modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x])**2.))/derr))
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        else:
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
            axes = [xx, yy, zz, v]
            def fitfunc(xx, yy, zz, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), ny, nx)
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelcube = model.build_cube(xx, yy, zz, v, *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator(axes, d, derr, fitfunc, lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init = f_rand_init)
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(xx, yy, zz, v, *self.popt) if self.n_nest is None else fitfunc(x, y, z, v, *self.popt)
        self.modelcube = modelcube

        self.writeout_fitres(outname, BE.criterion)

        return modelcube



    def writeout_fitres(self, outname, criterion = None,
        credible_interval = 0.68):
        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )

        # overwrite default output
        outtxtfile = outname + '_results.txt'
        dt = datetime.now()
        dtstr = dt.strftime('%Y-%m-%d %H:%M:%S')
        ne, _ = self.pfit.shape
        labels = self.model_keys
        perr_indx = dict(zip(self.pfree_keys, 
            [*np.arange(0,len(self.pfree_keys))]))
        with open(outtxtfile, '+w') as f:
            if ne == 2:
                f.write('# param mean sigma\n')
                # make a full perr list
                for i, k in enumerate(self.model_keys):
                    if k in self.pfree_keys:
                        indx = perr_indx[k]
                        f.write(
                            '%s %13.6e %13.6e\n'%(labels[i], params_full[i], self.perr[indx])
                            )
                    else:
                        f.write(
                            '%s %13.6e %13.6e\n'%(labels[i], params_full[i], 0.)
                            )
            elif ne == 3:
                f.write('# param 50th %.fth %.fth\n'%(50*(1. - credible_interval), 50*(1. + credible_interval)))
                for i, k in enumerate(self.model_keys):
                    if k in self.pfree_keys:
                        indx = perr_indx[k]
                        f.write(
                            '%s %13.6e %13.6e %13.6e\n'%(labels[i], 
                                params_full[i], self.perr[0, indx], self.perr[1, indx])
                            )
                    else:
                        f.write(
                            '%s %13.6e %13.6e %13.6e\n'%(labels[i], params_full[i], 0., 0.)
                            )
            if criterion is not None:
                f.write('# criterion')
                for k in criterion.keys():
                    f.write('\n# %s %.4f'%(k, criterion[k]))


    # define fitting function
    def fit_multilayer_model(self, params, pranges, 
        d, derr, x, y, z, v,
        Tcmb = 2.73, f0 = 230., dv_mode = 'total', 
        pterm = False, outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves = emcee.moves.StretchMove(), 
        symmetric_error=False, npool = 1, f_rand_init = 1.,
        show_results = True):
        axes = [x, y, z, v]
        # drop unecessary axis
        d = np.squeeze(d)
        # dimentions
        nx, ny, nz = len(x), len(y), len(z)
        # sampling step
        delx = - (x[1] - x[0]) # au / self.dist # arcsec
        dely = (y[1] - y[0]) # au / self.dist   # arcsec
        if self.beam is not None:
            if self.sampling:
                smpl_x = int(self.beam[1] * 0.5 / delx)
                smpl_y = int(self.beam[1] * 0.5 / dely)
                d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                Rbeam_pix = 1.
            else:
                smpl_x, smpl_y = 1, 1
                Rbeam_pix = np.pi/(4.*np.log(2.)) * self.beam[1] * self.beam[0] / delx / dely
                d_smpld = d.copy()
        else:
            Rbeam_pix = 1.
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()

        # log likelihood
        def lnlike(params, d, derr, fmodel, *x):
            mdl = fmodel(*x, *params)

            # Likelihood function (in log)
            exp = -0.5 * np.nansum((d-mdl)**2/(derr*derr) 
                + np.log(2.*np.pi*derr*derr)) / Rbeam_pix

            if np.isnan(exp):
                return -np.inf
            else:
                return exp


        # labels
        if len(labels) != len(params): labels = self.pfree_keys


        # setup model
        model = Builder(x, y, z, v, 
            self.model, nsub = self.n_nest, zstrech = self.zstrech, 
            reslim = self.reslim, beam = self.beam,
            line = self.line, iline = self.iline, rin = self.rin,
            adoptive_zaxis = True, cosi_lim = 0.5,)
        #model = self.model(x, y, z, v,
        #    xlim = None, ylim = None, zlim = None,
        #    nsub = self.n_nest, reslim = self.reslim,
        #    line = self.line, iline = self.iline,
        #    adoptive_zaxis = True, cosi_lim = 0.5, beam = self.beam,)
        model.grid.gridinfo()

        # renew grid every fit or not
        if any([i in self.pfree_keys for i in ['dx0', 'dy0', 'inc', 'pa']]):
            renew_grid = True
        else:
            print('Grid-saved mode')
            renew_grid = False

        # make grid
        model.set_model(self._params_ini)
        model.deproject_grid()
        model.show_model_sideview(showfig = False, savefig = True, cmap = 'coolwarm')


        # define fitting function
        def fitfunc(x, y, z, v, *params):
            # safty net
            if np.all((pranges[0] < np.array([*params])) \
                * (np.array([*params]) < pranges[1])) == False:
                return np.zeros(
                    (len(v), ny, nx)
                    )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


            # merge free parameters to fixed parameters
            params_free = dict(zip(self.pfree_keys, [*params]))
            #print('p input')
            #print(params_free)
            _params_full = merge_dictionaries(params_free, self.params_fixed)
            #params_full = list(
            #    {k: _params_full[k] for k in self.model_keys}.values()
            #    ) # reordered elements

            # update parameters
            _model = copy.deepcopy(model) # to make sure parallel calculations go well
            _model.set_model(_params_full)

            # renew grid
            if renew_grid:
                _model.deproject_grid()

            # cube on the original grid
            modelcube = _model.build_cube(
                Tcmb = Tcmb, f0 = f0, dist = self.dist, 
                dv_mode = dv_mode, pterm = pterm)

            return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator(axes, d_smpld, derr, fitfunc, lnlike = lnlike)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init = f_rand_init,
            show_results = show_results)
        self.pfit = BE.pfit.copy()
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        smpl_y, smpl_x = 1, 1
        modelcube = fitfunc(x, y, z, v, *self.popt)
        #modelcube = fitfunc(x, y, z, v, *list(self.params_free.values()))
        self.modelcube = modelcube

        self.writeout_fitres(outname, BE.criterion)

        return modelcube


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def mathlabels(pylabels):
    labels = {
    'Td0': r'$T_{\mathrm{d},0}$',
    'qd': r'$q_\mathrm{d}$',
    'log_tau_dc': r'$\log \tau_\mathrm{c,d}$',
    'rc_d': r'$R_\mathrm{c,d}$',
    'gamma_d': r'$\gamma_\mathrm{d}$',
    'Tg0': r'$T_{\mathrm{g},0}$',
    'qg': r'$q_\mathrm{g}$',
    'log_N_gc': r'$\log N_\mathrm{c,g}$',
    'log_tau_gc': r'$\log \tau_\mathrm{c,g}$',
    'rc_g': r'$R_\mathrm{c,g}$',
    'gamma_g': r'$\gamma_\mathrm{g}$',
    'z0': r'$z_0$',
    'pz': r'$p_z$',
    'h0': r'$H_0$',
    'ph': r'$p_H$',
    'inc': r'$i$',
    'pa': r'$PA$',
    'ms': r'$M_\ast$',
    'vsys': r'$v_\mathrm{sys}$',
    'dx0': r'$\delta x_0$',
    'dy0': r'$\delta y_0$',
    'r0': r'$R_0$',
    'dv': r'$\Delta v$',
    'pdv': r'$p_{\Delta v}$',
    'delv': r'$\Delta v$'
    }
    keys = labels.keys()

    return [labels[i] if i in keys else i for i in pylabels]




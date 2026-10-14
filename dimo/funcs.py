# modules
import numpy as np
from scipy.signal import convolve


# Gaussians
# 1D
def gauss1d(x, amp, mx, sig):
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
    if deg: angle = np.radians(angle)
    if coords: angle = -angle

    cos = np.cos(angle)
    sin = np.sin(angle)

    xrot = x*cos - y*sin
    yrot = x*sin + y*cos

    return xrot, yrot



def beam_convolution(xx, yy, image, beam, beam_image = None):
    '''
    Perform beam convolution. The input image must be in 
    a unit of an arbitral intensity per pixel, and then the output image
    will be in a unit of the intensity per beam.

    Parameters
    ----------
    xx, yy (array): 2D grid arrays.
    image (array): Input image in dimension of two to four.
    beam (list): Observing beam. Must be given as [bmaj, bmin bpa]
    beam_image (ndarry): 2D array of defined beam that will be convolved.
    '''
    # grid info
    ny, nx = xx.shape
    dx = xx[0,1] - xx[0,0]
    dy = yy[1,0] - yy[0,0]

    # beam to be convolved
    if beam_image is None:
        # define Gaussian beam
        gaussbeam = gaussian2d(xx, yy, 1., 
            xx[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            yy[ny//2 - 1 + ny%2, nx//2 - 1 + nx%2],
            beam[1] / 2.35, beam[0] / 2.35, beam[2], peak=True)
        gaussbeam /= np.sum(gaussbeam)
    else:
        gaussbeam = beam_image.copy()

    # dimension check
    ndim = len(image.shape)
    if ndim == 2:
        pass
    elif ndim == 3:
        gaussbeam = np.array([gaussbeam])
    elif ndim == 4:
        gaussbeam = np.array([[gaussbeam]])
    else:
        print('WARNING\tbeam_convolution: dimension of the image may be too large.')
        print('WARNING\tbeam_convolution: ndim > 4 may cause a wrong result.')

    # convolution
    Iv = np.where(np.isnan(image), 0., image)
    Iv = convolve(Iv, gaussbeam, mode='same')

    # unit
    Iv /= np.abs(dx * dy) # per pixel to per user-defined length unit^2
    Iv *= np.pi/(4.*np.log(2.)) * beam[0] * beam[1] # to per beam

    return Iv


def glnprof(t0, v, v0, delv, fn = 1.):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (ndarray): Total optical depth or integrated intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. t0 will be in units of velocity if fn is not given.
    '''

    return t0 / np.sqrt(np.pi) / delv * np.exp( - (v - v0)**2. / delv**2.) * fn


def glnprof_conv(t0, v, delv, fn = 1.):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (ndarray): Peak optical depth or peak intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. 
                 t0 will be dimensionless and regarded as tau_v if fn is not given.
    '''
    # kernel
    nv = len(v)
    dv = v[1] - v[0]
    if dv < 0: dv *= -1.
    gauss = np.exp(-(v - v[nv//2 - 1 + nv%2]) **2. / delv**2)
    gauss /= np.sum(gauss)

    # convolution
    ndim = len(t0.shape)
    if ndim == 1:
        pass
    elif ndim == 2:
        gauss = np.array([gauss]).T
    elif ndim == 3:
        gauss = np.array([[gauss]]).T
    elif ndim == 4:
        gauss = np.array([[[gauss]]]).T
    else:
        print('WARNING\tglnprof_conv: dimension of the image may be too large.')
        print('WARNING\tglnprof_conv: ndim > 4 may cause a wrong result.')

    tv = t0 / dv * fn * np.sqrt(np.pi) * delv # per pixel with scaling
    tv = convolve(tv, gauss, mode='same')

    return tv


def main():
    # ------ for debug -------
    # model
    tau0 = 1. # tau_v (dimensionless)
    delv = 0.5 # line broadening (km/s)

    # resolutions/ranges
    dvs = np.array([0.08, 0.16])
    lv = 3.
    # ---------------------



    # ----- calculations ------
    # import
    import matplotlib.pyplot as plt

    # figure
    fig, axes = plt.subplots(1,2)
    axes = axes.ravel()
    #ax = fig.add_subplot(111)

    for dv, ax in zip(dvs, axes):
        # velocity grid
        nv = int(lv * 2 / dv)
        ve = np.arange(-nv//2, nv//2 +1, 1) * dv
        v = 0.5 * (ve[:-1] + ve[1:])

        # model
        model = np.zeros(len(v))
        tau_v = tau0
        model[nv//2] = tau_v

        # convolution
        model_convolved = glnprof_conv(model, v, delv)

        # check
        print('dv = %.2f'%dv)
        print('Integrated tau0: %.2f, %.2f'%(
            np.sum(model*dv), np.sum(model_convolved*dv), )) #np.sum(model_convolved_as*dv)
        print('Peak tau_v: %.4f'%np.max(model_convolved))
        #print('Jy per beam: %.4f'%(np.max(model_convolved) * np.sqrt(2. * np.pi * sigma**2.) / dv))

        # plot
        ax.step(v, model, where='mid', color='k', ls='-', lw=1.)
        ax.step(v, model_convolved, where='mid', color='r', ls='-', lw=1.)

        ax.set_ylim(-0.1, 10.)

    plt.show()
    # ---------------------



if __name__ == '__main__':
    main()

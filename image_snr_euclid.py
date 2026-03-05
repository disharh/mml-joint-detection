import matplotlib.pyplot as plt
from scipy.special import gammaincinv, gamma
import numpy as np
from functools import lru_cache
import os

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource

# data specifics
imager = os.environ.get("SH_IMAGER", "euclid")
print("Assuming imaging by", imager)
if imager == 'euclid':
    exp_time = 3*565  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    background_rms = 9.494*np.sqrt(3)/exp_time  # background noise per pixel
    numPix = int(os.environ.get("SH_NUMPIX", 50)) # cutout pixel size
    deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
    fwhm = 0.17  # full width half max of PSF
    ZP = 25.1209 # Euclid photometric zeropoint

    # These are derived in lensing_statistics_analysis_Copy1.ipynb
    threshold_theta_ein=0.33
    threshold_score = 70
    threshold_variation = 0.56
elif imager == 'euclid_opt':
    exp_time = 3*565  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    background_rms = 9.494*np.sqrt(3)/exp_time  # background noise per pixel
    numPix = int(os.environ.get("SH_NUMPIX", 50)) # cutout pixel size
    deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
    fwhm = 0.17  # full width half max of PSF
    ZP = 25.1209 # Euclid photometric zeropoint

    # These are derived in lensing_statistics_analysis_Copy1.ipynb
    threshold_theta_ein=0.33
    threshold_score = 45.37
    threshold_variation = 0.885
elif imager == 'subaruhsc': # Like the "UltraDeep" survey from 1702.08449. Table 2: r-band is taken. Bit pessimistic.
    exp_time=70*60
    background_rms = 356.399/exp_time  # background noise per pixel
    numPix = int(os.environ.get("SH_NUMPIX", 50)) # cutout pixel size
    deltaPix = 0.17  # pixel size in arcsec (area per pixel = deltaPix**2)
    fwhm = float(os.environ.get("SH_SEEING", 0.4))  # full width half max of PSF
    ZP=28.9145 # Euclid photometric zeropoint

    # These are derived in lensing_statistics_analysis.ipynb
    if fwhm==0.4:
        #threshold_theta_ein=0.663
        #threshold_score=89.15
        #threshold_variation=0.6855
        threshold_theta_ein=0.5286664
        threshold_score=111.51991461
        threshold_variation=0.87403949
    elif fwhm==0.6:
        threshold_theta_ein = 0.719987  
        threshold_score = 171.66531407
        threshold_variation=0.6504596
    else:
        raise ValueError("Undetermined thresholds for given $SH_SEEING")
else:
    raise ValueError("Unknown imager")

# PSF specification
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 5}

kwargs_numerics = {'supersampling_factor': 5, 'supersampling_convolution': True}


def normalize(kwargs_sersic, reverse=False, inplace=True):
    """
    Really unnormalize: go from normalized amplitude to lenstronomy amplitude.
    :param kwargs_sersic: lenstronomy ofrmat sersic kwargs
    :param reverse: if True, go from lenstronomy to normalised amplitude (e.g. to calculate magnitudes)
    :param inplace: Edit the kwargs in place.
    :return:
    """
    ns = kwargs_sersic['n_sersic']
    rs = kwargs_sersic['R_sersic']
    bn = gammaincinv(2*ns, 0.5)
    fact = rs**2*2*np.pi*ns*np.exp(bn)/bn**(2*ns)*gamma(2*ns)

    if 'e1' in kwargs_sersic:
        phi, q = param_util.ellipticity2phi_q(kwargs_sersic['e1'], kwargs_sersic['e2'])
        q = max(q, 1/q)
        fact /= q
    if reverse:
        fact = 1/fact
    if inplace:
        kwargs_sersic['amp'] /= fact
        # After this, amp is such that instead of amp being the sbr at Re, amp is the integrated flux.
    else:
        return fact



def clear_cache_after_use(func):
    def inner(*args, **kwargs):
        imagemodel = func(*args, **kwargs)
        imagemodel.reset_point_source_cache()
        return imagemodel
    return inner

#@clear_cache_after_use
@lru_cache
def get_ImageModel(lens_model_list, lens_light_model_list=None, source_model_list=None, point_source_model_list=None):
    """
    A cached function so that the image model object does not have to be remade at each likelihood call. Be sure to clear the caches!
    :param lens_model_list:
    :param lens_light_model_list:
    :param source_model_list:
    :param point_source_model_list:
    :return:
    """
    data_class = ImageData(**kwargs_data)
    psf_class = PSF(**kwargs_psf)
    source_model_class = LightModel(light_model_list=source_model_list) if source_model_list is not None else None
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list) if lens_light_model_list is not None else None
    point_source_model_class = PointSource(point_source_model_list, fixed_magnification_list=[True], kwargs_lens_eqn_solver={'solver': 'analytical'}) if point_source_model_list is not None else None
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    imageMod = ImageModel(data_class, psf_class, lens_model_class=lens_model_class, lens_light_model_class=lens_light_model_class,
                          point_source_class=point_source_model_class, source_model_class=source_model_class, kwargs_numerics=kwargs_numerics)
    return imageMod



def calcStoN(mag_lens, lens_reff, pos_lens, lens_model_class, mag_source, kwargs_lens, reff_source, ell_source,
             phi_ellsource, pos_source, lens_nsersic=4., source_nsersic=1., plot=False, return_images=False, approx_ps=None, verbose=False,
             elliptic_lensgal=False, lens_light_theta_ell_ell=None, require_score=False):
    """
    Calcuates the images and total signal to noise of some lens system.
    :param mag_lens:
    :param lens_reff:
    :param pos_lens:
    :param lens_model_class:
    :param mag_source:
    :param kwargs_lens:
    :param reff_source:
    :param ell_source:
    :param phi_ellsource:
    :param pos_source:
    :param plot: do some diagnostic plots
    :param return_images: return images too
    :param approx_ps: approximate the source as a point source (for small sersic radii)
    :param verbose: some more output
    :param elliptic_lensgal: whether the lens galaxy is elliptic (True) or circular (False)
    :param require_score: require the background pixel number criterion.
    :return:
    """

    #cond_resolve = kwargs_lens[0]['theta_E']**2 > (fwhm/2)**2 + reff_source**2
    cond_resolve = kwargs_lens[0]['theta_E'] > threshold_theta_ein # >0.3
    if not cond_resolve and not return_images:
        if verbose: print("too smalll einstein radius")
        return -2., -500.-100*(-kwargs_lens[0]['theta_E'] + threshold_theta_ein)**2, (-42., -42.)
        #return -2., -50.-10000*(-kwargs_lens[0]['theta_E']**2 + (fwhm/2)**2 + reff_source**2), (-42., -42.)
    cond_smallenough = reff_source*(1-ell_source) < threshold_variation*kwargs_lens[0]['theta_E']
    if not cond_smallenough and not return_images:
        if verbose: print("too smalll einstein radius w.r.t. source reff")
        return -2., -500.-100*(-reff_source*(1-ell_source)/kwargs_lens[0]['theta_E'] + threshold_variation)**2, (-42., -42.)
        #return -2., -50.-10000*(-kwargs_lens[0]['theta_E']**2 + (fwhm/2)**2 + reff_source**2), (-42., -42.)

    # Lens light
    lens_light_model_list = ('SERSIC_ELLIPSE' if elliptic_lensgal else 'SERSIC',)
    kwargs_sersic = {'amp': 10 ** (-(mag_lens - ZP) / 2.5), 'R_sersic': lens_reff, 'n_sersic': lens_nsersic, 'center_x': pos_lens[0], 'center_y': pos_lens[1]}
    if elliptic_lensgal:
        if lens_light_theta_ell_ell is None:
            e1_light, e2_light = kwargs_lens[0]['e1'], kwargs_lens[0]['e2']
        else:
            theta_ell, ell = lens_light_theta_ell_ell
            e1_light, e2_light = param_util.phi_q2_ellipticity(theta_ell, 1-ell)
        kwargs_sersic = {**kwargs_sersic, 'e1': e1_light, 'e2': e2_light}
    normalize(kwargs_sersic)
    kwargs_lens_light = [kwargs_sersic]

    lens_model_list = tuple(lens_model_class.lens_model_list)

    imgMod_nosrc = get_ImageModel(lens_model_list=lens_model_list, lens_light_model_list=lens_light_model_list)
    imgMod_nosrc.reset_point_source_cache()
    image_nosrc = imgMod_nosrc.image(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light)

    # Source light
    if approx_ps is None:
        approx_ps = reff_source < deltaPix / 5 # Maybe a better calibration could be done. But this seems reasonable (at the 5% level)
        # Todo: test if this is also true up to q=0.1

    if not approx_ps:
        e1, e2 = param_util.phi_q2_ellipticity(phi=phi_ellsource, q=1-ell_source)
        kwargs_sersic_ellipse = {'amp': 10 ** (-(mag_source - ZP) / 2.5), 'R_sersic': reff_source, 'n_sersic': source_nsersic, 'center_x': pos_source[0], 'center_y': pos_source[1], 'e1': e1, 'e2': e2}
        normalize(kwargs_sersic_ellipse)
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_list = ('SERSIC_ELLIPSE',)
        imgMod_nolens = get_ImageModel(lens_model_list=lens_model_list, source_model_list=source_model_list)
        imgMod_nolens.reset_point_source_cache()
        image_nolens = imgMod_nolens.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)
    else:
        kwargs_ps = [{'ra_source': pos_source[0], 'dec_source': pos_source[1], 'source_amp': 10**(-(mag_source-ZP)/2.5)}]
        point_source_model_list = ('SOURCE_POSITION',)
        imgMod_nolens = get_ImageModel(lens_model_list=lens_model_list, point_source_model_list=point_source_model_list)
        imgMod_nolens.reset_point_source_cache()
        image_nolens = imgMod_nolens.image(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps)
    if verbose:
        from pprint import pprint
        pprint({'kwargs_lens': kwargs_lens, 'kwargs_lens_light': kwargs_lens_light, 'kwargs_source': kwargs_source if not approx_ps else None, 'kwargs_ps': kwargs_ps if approx_ps else None})

    image_tot = image_nolens + image_nosrc
    mag_tot = image_nolens.sum() / 10**((mag_source-ZP)/-2.5)

    SNlens = np.sqrt(((image_nosrc*exp_time)**2/((background_rms*exp_time)**2+image_tot*exp_time)).sum())
    SNsource = np.sqrt(((image_nolens*exp_time)**2/((background_rms*exp_time)**2+image_tot*exp_time)).sum())


    if plot:
        print('mag_tot, 1-q', mag_tot, ell_source)
        print("Snl,s", SNlens, SNsource)
        print("ml, ms, rl, rs", mag_lens, mag_source, lens_reff, reff_source)
        #plt.pcolormesh(image_tot + image_util.add_background(image_tot, background_rms) + image_util.add_poisson(image_tot, exp_time=exp_time))
        plt.pcolormesh(image_tot)
        plt.show()

    """
    #Turned off by default: implements some of the Collett2015 observability requirements
    cond_shear = mag_tot*reff_source > fwhm/2
    cond_mag = mag_tot > 3
        if require_shear:
            if not cond_shear and not cond_mag and not return_images:
                return -5., -50.+1000*(mag_tot*reff_source-fwhm/2) +100*(mag_tot-3), (mag_tot/3, mag_tot*reff_source/(fwhm/2))
            if not cond_shear and not return_images:
                return -5., -50.+1000*(mag_tot*reff_source-fwhm/2), (mag_tot/3, mag_tot*reff_source/(fwhm/2))
            if not cond_mag and not return_images:
                return -10., -50.+1000*(mag_tot-3), (mag_tot/3, mag_tot*reff_source/(fwhm/2))
    """
    threshold = 1.5
    noise = np.sqrt(background_rms**2 + image_tot/exp_time)
    score = np.sum(image_nolens/noise>threshold)
    cond_highenough_score = score > threshold_score
    if require_score and not return_images:
        if not cond_highenough_score:
            if verbose: print("too small score")
            return -10., -50. - 1000*(score-threshold_score)**2, (mag_tot/3, score)

    observable = cond_highenough_score and cond_resolve and cond_smallenough
    if return_images:
        img_nonoise = image_tot
        img_plusnoise = image_tot + image_util.add_background(image_tot, background_rms) + image_util.add_poisson(image_tot, exp_time=exp_time)
        return SNlens, SNsource, mag_tot, img_nonoise, img_plusnoise, image_nolens, observable
    else:
        return SNlens, SNsource, (mag_tot/3, mag_tot*reff_source/(fwhm/2))

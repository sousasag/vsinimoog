# adapted from vsini_mpfit_stars from Ana Rita Silva and Vardan
# using vmac from formula and vrot from mpfit

from multiprocessing.pool import RUN
from ssl import get_server_certificate
import time
import os
import numpy as np
from PyAstronomy import pyasl
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(repr(__file__).replace("'",""))+'mpfit/')
from mpfit.mpfit import mpfit
from matplotlib import pyplot as plt
import pickle

LOC_FLAG      = "WORK"
LOC_FLAG      = "HOME"
RUN_PATH      = 'running_dir/'
#MOOG_PATH    = ""  # if you have MOOGSILENT in your path use this.
#MODELS_PATH  = ""
if LOC_FLAG == "WORK":
    MOOG_PATH     = "/home/sousasag/Programas/GIT_projects/SPECPAR3/codes/MOOG2019/./"  # otherwise write your full path to MOOGSILENT here
    MODELS_PATH   = "/home/sousasag/Programas/GIT_projects/SPECPAR3/codes/interpol_models/./"
else:
    MOOG_PATH     = "/home/sousasag/Programs/MOOG2019/./"  # otherwise write your full path to MOOGSILENT here
    MODELS_PATH   = "/home/sousasag/Programs/interpol_models/./"


LINELIST_PATH = 'linelist/'

def norm(obs_array_complete, snr):
    """
    Function to normalise a given interval of data points (average flux of continuum).
    :param obs_array_complete: list of floats of data points (e.g. flux)
    :param snr: float, signal to noise ratio of given star
    :return: m, float, mean value of data points
    """
    obs_array_complete = list(obs_array_complete)
    obs_array = obs_array_complete.copy()
    m = np.mean(obs_array_complete)
    sigma = np.std(obs_array_complete)

    # define maximum iterations as desired
    max_it = 20 if snr <= 200.0 else 10

    it = 0
    while it < max_it:
        for data_value in obs_array:
            if data_value < m-sigma or data_value > m+2*sigma:
                obs_array.remove(data_value)
            m = np.mean(obs_array)
            sigma = np.std(obs_array)
        it = it + 1

    return m

def moog_fe(star, p, vmac, lambda_i, lambda_f, ldc, CDELT1, instr_broad):
    """
    This function outputs a parameter file with the given details, outputs a text file and calls MOOGSILENT to read
    these files. In turn, MOOGSILENT will output an ascii file of a synthetic spectrum for the given star properties.
    The function then reads the data/wavelength in the ascii file.
    :param star: string, star
    :param p: list of floats, starting values of parameters (in this case, only vrot)
    :param vmac: float, macroturbulence
    :param lambda_i: float, starting wavelength of synthesis
    :param lambda_f: float, ending wavelength of synthesis
    :param ldc: float, limb darkening coefficient
    :param CDELT1: float, delta lambda in the observed spectrum
    :param instr_broad: float, instrumental broadening
    :return: .par file, .txt file, runs them through MOOGSILENT and returns a message of completion
    """

    print (star, p, vmac, lambda_i, lambda_f, ldc, CDELT1, instr_broad)

    with open(RUN_PATH+'synth_fe.par', 'w') as par:
        par.write('synth \n')
        par.write('model_in       \'' + star + '.atm\' \n')
        par.write('summary_out    \'out1\' \n')
        par.write('smoothed_out   \'synth_fe.asc\' \n')
        par.write('standard_out   \'out2\' \n')
#        par.write('lines_in       \'../linelist/iron_vrot_moog.list\' \n')
        par.write('lines_in       \'../linelist/iron_vrot_moog_2.list\' \n')
        par.write('abundances     1    1\n')
        par.write('        26     0.00 \n')
        par.write('plot           1 \n')
        par.write('synlimits \n')
        par.write(str(lambda_i) + '  ' + str(lambda_f) + '   ' + str(round(CDELT1, 3)) + '  1.0 \n')
        par.write('plotpars       1 \n')
        par.write(str(lambda_i) + '   ' + str(lambda_f) + '  0.80   1.05 \n')
        par.write('0.0   0.0   0.0   1.0  \n')
        par.write('r  ' + str(round(instr_broad, 3)) + '  ' + str(round(p[0], 3)) + '  ' + str(round(ldc, 3)) + '  ' + str(round(vmac, 3)) + '  0.0 \n')
        par.write('damping        0 \n')
        par.write('atmosphere     1 \n')
        par.write('molecules      2 \n')
        par.write('trudamp        1 \n')
        par.write('lines          1 \n')
        par.write('strong         0 \n')
        par.write('flux/int       0 \n')
        par.write('units          0 \n')
        par.write('opacit         0 \n')
        par.write('obspectrum     0 \n')

    with open(RUN_PATH+'synth_fe.txt', 'w') as txt:
        txt.write('synth_fe.par \n')
        txt.write('f \n')
        #txt.write(star + '_fe.ps \n')
        txt.write('q \n')

    os.system('rm '+RUN_PATH+'batch.par')
    os.system('cd '+RUN_PATH+' && '+MOOG_PATH+'MOOGSILENT < synth_fe.txt')

    return 'Finished MOOG synthesis for ' + star + ' in range ' + str(lambda_i) + ' to ' + str(lambda_f) + '.'

def minimize_synth(p, star, vmac, fe_intervals, obs_lambda, obs_flux, ldc, CDELT1, instr_broad, **kwargs):
    """
    Function to minimize a model to observational data.
    :param p: list, initial values of parameters
    :param star: string, star name
    :param vmac: float, value of macroturbulence of star
    :param fe_lines_intervals: intervals where iron lines are present
    :param fe_intervals_lambda: fe intervals that contain the start and end point of the synthesis
    :param obs_flux: list of observational flux points
    :param ldc: float, limb darkening coefficient
    :param CDELT1: float, delta lambda in the observed spectrum
    :param kwargs
    :return: best values of parameters
    """


    def myfunct(p, star=None, vmac=None, fe_intervals=None, obs_lambda=None,
                obs_flux=None, flux_err=0.01, **kwargs):
        """
        User supllied function that contains the model to be tested. Calculates the synthetic points at the same
        wavelength of the observational points (this means inside the iron lines regions).
        Returns an integer as status of the calculation and an array of deviates between the data points and the model
        points, normalized by the error in the observation (here, an arbitrary value is given as HARPS did not provide
        errors).
        :param p: list, initial values of parameters
        :param star: string, star name
        :param vmac: float, macroturbulence of star
        :param fe_lines_intervals: intervals where iron lines are present
        :param obs_flux: list of observational flux points
        :param flux_err: float, error in observational flux points (set to 0.01 here)
        :param kwargs
        :return: integer (status of operations), array of deviates
        """

        #print round(p[0], 3)

        #lambda_i_values = [round(fe_intervals_lambda[0][0], 3), round(fe_intervals_lambda[0][0], 3) +2.01]
        #lambda_f_values = [round(fe_intervals_lambda[0][0], 3) + 2.00, round(fe_intervals_lambda[-1][1], 3)]

        gap = obs_lambda[-1] - obs_lambda[0]
        if gap <= 450:
            lambda_i_values = [round(obs_lambda[0], 3)]
            lambda_f_values = [round(obs_lambda[-1], 3)]
        elif gap > 450 and gap <= 900:
            lambda_i_values = [round(obs_lambda[0], 3), round(obs_lambda[int(len(obs_lambda)/2)], 3)]
            lambda_f_values = [round(obs_lambda[int(len(obs_lambda)/2)-1], 3), round(obs_lambda[-1], 3)]
        elif gap > 900:
            lambda_i_values = [round(obs_lambda[0], 3), round(obs_lambda[int(len(obs_lambda)/3)], 3), round(obs_lambda[int(len(obs_lambda)/3*2)], 3)]
            lambda_f_values = [round(obs_lambda[int(len(obs_lambda)/3)-1], 3), round(obs_lambda[int(len(obs_lambda)/3*2)-1], 3), round(obs_lambda[-1], 3)]

        synth_data = []  # all flux values from model
        synth_lambda = []  # all wavelength points from model

        for lambda_i, lambda_f in zip(lambda_i_values, lambda_f_values):
            moog_fe(star, p, vmac, lambda_i, lambda_f, ldc, CDELT1, instr_broad)
            with open(RUN_PATH+'synth_fe.asc') as asc:
                for x in asc:
                    if x[0] == ' ':
                        entry = x.rstrip().split()
                        synth_lambda.append(float(entry[0]))
                        synth_data.append(float(entry[1]))

        synth_data_fe = []
        synth_lambda_fe = []

        fe_intervals_list = [row for row in fe_intervals[['ll_li', 'll_lf','ll_si','ll_sf']].to_numpy()]

        for i,(ll_li, ll_lf, ll_si, ll_sf) in enumerate(fe_intervals_list):
            select_sll = np.where((synth_lambda >= ll_si) & (synth_lambda <= ll_sf))[0]
            synth_data_fe.extend(list(np.array(synth_data)[select_sll]))
            synth_lambda_fe.extend(list(np.array(synth_lambda)[select_sll]))

        obs_flux = np.array(obs_flux)
        synth_data_fe = np.array(synth_data_fe)

        err = np.zeros(len(obs_flux)) + flux_err
        status = 0

        return [status, (obs_flux - synth_data_fe)/err]

    def convergence_info(res, parinfo, dof):
        """
        Function that returns the best parameter values and errors.
        :param res: output object resultant from mpfit function
        :param parinfo: list of dictionaries of parameter information
        :param dof: float, degrees of freedom
        :return: best parameter values and errors
        """

        if res.status == -16:
            print('status = {0:4}: A parameter or function value has become infinite or an undefined number.'
                    .format(res.status))
        elif -15 <= res.status <= -1:
            print('status = {0:4}: MYFUNCT or iterfunct functions return to terminate the fitting process.'
                    .format(res.status))
        elif res.status == 0:
            print('status = {0:4}: Improper input parameters.'.format(res.status))
        elif res.status == 1:
            print('status = {0:4}: Both actual and predicted relative reductions in the sum of squares are at most ftol'
                    '.'.format(res.status))
        elif res.status == 2:
            print('status = {0:4}: Relative error between two consecutive iterates is at most xtol.'.format(res.status))
        elif res.status == 3:
            print('status = {0:4}: Conditions for status = 1 and status = 2 both hold.'.format(res.status))
        elif res.status == 4:
            print('status = {0:4}: The cosine of the angle between fvec and any column of the jacobian is at most gtol '
                    'in absolute value.'.format(res.status))
        elif res.status == 5:
            print('status = {0:4}: The maximum number of iterations has been reached.'.format(res.status))
        elif res.status == 6:
            print('status = {0:4}: ftol is too small'.format(res.status))
        elif res.status == 7:
            print('status = {0:4}: xtol is too small.'.format(res.status))
        elif res.status == 8:
            print('status = {0:4}: gtol is too small'.format(res.status))

        # res.niter = number of iterations
        # res.fnorm = summed square residuals
        chi_reduced = round((res.fnorm/dof), 4)
        vrot = round(float(res.params[0]), 3)
        vrot_err = round(float(res.perror[0]), 3)
        vrot_parameters = [vrot, vrot_err, res.niter, round(float(res.fnorm), 3), chi_reduced, res.status]

        print (star, ('%s: %s +- %s' % (parinfo[0]['parname'], vrot, vrot_err)))

        return vrot_parameters

    # define parameters for minimization
#    vrot_info = {'parname': 'vrot', 'value': 15, 'fixed': 0, 'limited': [1, 1], 'limits': [1, 20], 'mpside': 2,
#                    'step': 0.001}
    vrot_info = {'parname': 'vrot', 'value': 5, 'fixed': 0, 'limited': [1, 1], 'limits': [0.1, 30], 'mpside': 2,
                    'step': 0.001}
    parinfo = [vrot_info]

    fa = {'star': star, 'vmac': vmac, 'fe_intervals': fe_intervals, 'obs_lambda':
            obs_lambda, 'obs_flux': obs_flux}
    # call for minimization

    m = mpfit(myfunct, parinfo=parinfo, functkw=fa, ftol=1e-5, xtol=1e-5, gtol=1e-5, maxiter=20)

    dof = len(obs_flux) - len(m.params)
    parameters = convergence_info(m, parinfo, dof)

    return parameters

def creating_final_synth_spectra(vsini, star, spectrum, teff, feh, vtur, logg, snr, fe_intervals, ldc, instr_broad):
    obs_lambda, obs_flux, synth_data_fe = create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vsini)
    flux_ratio = (obs_flux / synth_data_fe)
    flux_diff = (obs_flux - synth_data_fe)
    synth_normalized_spectra = pd.DataFrame(data=np.column_stack((obs_lambda, synth_data_fe, flux_diff, flux_ratio)),columns=['wl','flux', 'flux_diff', 'flux_ratio'])
    synth_normalized_spectra.to_csv('running_dir/%s_synth_normalized_spectra.rdb' % star, index = False, sep = '\t')

def fit_lmfit_gauss(x, y):
  """
  This is my simple function to fit a Gaussian to data (x,y)
  using lmfit package
  """
  from lmfit.models import GaussianModel, ConstantModel

  gmodel = GaussianModel(prefix='g_')
  cmodel = ConstantModel(prefix='c_')
  model = gmodel + cmodel

  params = model.make_params()
  params['g_center'].set(value=x[np.argmax(y)], min=x[0], max=x[-1])
  params['g_sigma'].set(value=1.0, min=0.01, max=10.0)
  params['g_amplitude'].set(value=np.max(y), min=0.01)
  params['c_c'].set(value=np.min(y))

  result = model.fit(y, params, x=x)
  return result


def get_mask_spectra(li=6050, lf=6730):
    maskdata = fits.getdata('Data/ESPRESSO_G2.fits')
    tw = np.linspace(li, lf, 20000)
    ind = np.where( (maskdata['lambda'] > li) & (maskdata['lambda'] <  lf))
    tf = np.ones(tw.shape)
    for lr in maskdata['lambda'][ind]:
      tf -= np.exp(-(tw-lr)**2/(2.*0.1**2))
    return tw, tf

def get_rv(waves, flux, li=6050, lf=6730):
    tw, tf = get_mask_spectra(li=li, lf=lf)
    ind_s = np.where( (waves > li) & (waves <  lf))
    dw = waves[ind_s]
    df = flux[ind_s] / np.max(flux[ind_s]) # with first order normalization
    #plt.plot(dw, df)
    #plt.plot(tw, tf)
    #plt.show()
    rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -100., 100., 0.05, skipedge=500)
    maxind = np.argmax(cc)
    result = fit_lmfit_gauss(rv, cc)
    #print(result.fit_report())
    #print(result.params['g_center'].value)
    #plt.plot(rv, cc, 'bp-')
    #plt.plot(rv, result.best_fit, 'r-')
    #plt.axvline(x=rv[maxind], color='k', ls='--')
    #plt.show()  
  
    return rv[maxind]

def correct_rv(waves, rv):
    """
    This is my simple function to correct the radial velocity
    """
    c = 299792.458 # speed of light in km/s
    waves_corr = waves * (1 + rv*(-1)/c)
    return waves_corr


def get_spectra(fitsfile, li=6050, lf=6730):
    img_data, img_header = fits.getdata(fitsfile, header=True)
    cdelta1 = img_header['CDELT1']
    crval1  = img_header['CRVAL1']
    npoints = img_header['NAXIS1']
    ll = np.arange(0,npoints)*cdelta1+crval1
    rv = get_rv(ll, img_data, li=li, lf=lf)
    ll_cor = correct_rv(ll, rv)
    #plt.plot(ll_cor, img_data)
    #plt.show()
    return ll_cor, img_data, cdelta1

def get_intervals_normalized_spectra(ll, flux, fe_intervals, snr):
    obs_data_norm = []
    obs_lambda = []
    fe_intervals_list = [row for row in fe_intervals[['ll_li', 'll_lf','ll_si','ll_sf']].to_numpy()]
    for i,(ll_li, ll_lf, ll_si, ll_sf) in enumerate(fe_intervals_list):
        # get data from large intervals to do normalisation in each region
        select_ll = np.where((ll >= ll_li) & (ll <= ll_lf))[0]
        obs_data_one_large_interval = flux[select_ll]

        # m : mean value of larger interval to divide the smaller interval by
        m = norm(obs_data_one_large_interval, snr)

        # get data from smaller intervals and normalising it
        select_sll = np.where((ll >= ll_si) & (ll <= ll_sf))[0]
        obs_data_one_small_interval = flux[select_sll]/m
        obs_lambda_one_small_interval = ll[select_sll]

        #print 'obs_data_one_small_interval',  obs_data_one_small_interval
        #obs_data_norm.append(obs_data_one_small_interval)
        #obs_lambda.append(obs_lambda_one_small_interval)
        obs_data_norm.extend(list(obs_data_one_small_interval))
        obs_lambda.extend(list(obs_lambda_one_small_interval))

    # single list with data from all fe regions
    
    #print ('finished')
    #print 'obs_data', obs_data_norm
    #obs_data_norm_flat = [float(value) for sublist in obs_data_norm for value in sublist]
    #print 'obs_data_norm', obs_data_norm
    # single list with wavelength points from all fe regions
    #obs_lambda_flat = [float(value) for sublist in obs_lambda for value in sublist]
    #print ('obs_data all together', obs_data_norm_flat)
    #print "obs_lambda", np.array(obs_lambda)
    #return obs_lambda_flat, obs_data_norm_flat
    return obs_lambda,obs_data_norm


def create_atm_model(teff, log_g, feh, vtur, star):
    owd = os.getcwd()
    os.chdir(RUN_PATH)
    ## Modify to use functions in run_programs
    os.system('echo %s %s %s | ' % (teff, log_g, feh) + MODELS_PATH + 'intermod.e' )
    os.system('echo %s | ' % vtur + MODELS_PATH + 'transform.e' )
    os.system('mv out.atm %s.atm' %star)
    os.system('rm mod* for*')
    os.chdir(owd)


def get_vmac(teff, log_g):
    if teff <= 5000.0:
        vmac_funct = 2.0
    elif 6500.0 > teff > 5000.0:
        vmac_funct = 3.21 + 2.33e-3*(teff-5777) + 2.00e-6*(teff-5777)**2-2.00*(log_g-4.44)
        #macroturbulence calibration by Doyle et al. 2014
    elif teff >= 6500.0:
        vmac_funct = 5.5
        # code was run with slight changes to exception stars: HD82342, HD55, HD108564, HD145417, HD134440, HD23249,
        # HD40105, HD92588, HD31128, HD52449
    return vmac_funct


def get_vsini(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals):
    create_atm_model(teff, logg, feh, vtur, star)
    vmac = round(float(get_vmac(teff, logg)), 3)
#    vmac = 0.00 ### CAREFULL HERE <<<<<<<<<<<<<<<<<<<<<--------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<------------
    # read observational spectra
    obs_lambda_full_spectrum, obs_data_full_spectrum, delta_lambda =  get_spectra(spectrum)
    interp_function = interp1d(obs_lambda_full_spectrum, obs_data_full_spectrum)
    # create wavelength array equal to that of the synthetic models
    # starts at the first value of the first fe region and ends at the end value of the last region
    obs_lambda_interp = np.arange(np.min(fe_intervals['ll_li']),np.max(fe_intervals['ll_lf'])+round(float(delta_lambda), 3), round(float(delta_lambda), 3))
    obs_lambda_interp = np.round(obs_lambda_interp,3)
    # Why do we need to interpolate??? To get round values for the synthesis calculation

    obs_data_interp = interp_function(obs_lambda_interp)
    #print 'obs lambda interpolated', obs_lambda_interp
    #print 'obs data interpolated' , obs_data_interp

    # get wavelength points and flux data for Fe lines in interpolated rounded wavelenghts

    obs_lambda_flat, obs_data_norm_flat = get_intervals_normalized_spectra(obs_lambda_interp, obs_data_interp, fe_intervals, snr)

    obs_normalized_spectra = pd.DataFrame(data=np.column_stack((obs_lambda_flat,obs_data_norm_flat)),columns=['wl','flux'])
    obs_normalized_spectra.to_csv(RUN_PATH+'/%s_obs_normalized_spectra.rdb' % star, index = False, sep = '\t')

    par_list = [0.5]

    final_vrot  = minimize_synth(p=par_list, star=star, vmac=vmac, fe_intervals=fe_intervals,
                                obs_lambda=obs_lambda_flat, obs_flux=obs_data_norm_flat, ldc = ldc, CDELT1 = delta_lambda, instr_broad = instr_broad)

    vrot = final_vrot[0]
    vrot_err = final_vrot[1]
    status = final_vrot[5]

    #creating_final_synth_spectra(vsini = final_vrot, star = star, vmac = vmac, fe_intervals=fe_intervals, obs_lambda=obs_lambda_flat, obs_flux=obs_data_norm_flat, ldc = ldc, CDELT1 = delta_lambda, instr_broad = instr_broad, flux_err=1)

    print ('results', star, teff, logg, feh, snr, spectrum, final_vrot)

    #stars_par.write('{:10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} \n'.format(star, teff, log_g, feh,
    #                                                                                    final_vrot[0], final_vrot[1],
    #                                                                                    round(vmac_funct,3), round(ldc,3), instr_broad, snr))

    return vrot, vrot_err, vmac, status

def get_vsini_error(star, spectrum, teff, eteff, feh, efeh, vtur, logg, snr, ldc, instr_broad, fe_intervals):
    #parameters = ['teff', 'feh']
    vrot, vrot_err, vmac, status = get_vsini(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    vrot_tm, vrot_err_tm, vmac_tm, status_tm = get_vsini(star, spectrum, teff-eteff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    vrot_tp, vrot_err_tp, vmac_tp, status_tp = get_vsini(star, spectrum, teff+eteff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    vrot_fm, vrot_err_fm, vmac_fm, status_fm = get_vsini(star, spectrum, teff, feh-efeh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    vrot_fp, vrot_err_fp, vmac_fp, status_fp = get_vsini(star, spectrum, teff, feh+efeh, vtur, logg, snr, ldc, instr_broad, fe_intervals)

    vsini_final_err = np.sqrt( np.abs( (vrot_tm - vrot) - (vrot_tp - vrot) )**2. + np.abs( (vrot_fm - vrot) - (vrot_fp - vrot) )**2. + vrot_err**2. )


    print(vrot, vrot_err, vmac, status)
    print(vrot_tm, vrot_err_tm, vmac_tm, status_tm)
    print(vrot_tp, vrot_err_tp, vmac_tp, status_tp)
    print(vrot_fm, vrot_err_fm, vmac_fm, status_fm)
    print(vrot_fp, vrot_err_fp, vmac_fp, status_fp)
    return vrot, vrot_err, vmac, status, vsini_final_err


def create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test):
    # read observational spectra
    obs_lambda_full_spectrum, obs_data_full_spectrum, delta_lambda =  get_spectra(spectrum)
    interp_function = interp1d(obs_lambda_full_spectrum, obs_data_full_spectrum)
    obs_lambda_interp = np.arange(np.min(fe_intervals['ll_li']),np.max(fe_intervals['ll_lf'])+round(float(delta_lambda), 3), round(float(delta_lambda), 3))
    print(obs_lambda_interp)
    obs_lambda_interp = np.arange(np.min(fe_intervals['ll_li']),np.max(fe_intervals['ll_lf'])-round(float(delta_lambda), 3), round(float(delta_lambda), 3))
    print(obs_lambda_interp)
    obs_lambda_interp = np.round(obs_lambda_interp,3)
    print(obs_lambda_interp)
    print(obs_lambda_full_spectrum)
    obs_data_interp = interp_function(obs_lambda_interp)
    obs_lambda_flat, obs_data_norm_flat = get_intervals_normalized_spectra(obs_lambda_interp, obs_data_interp, fe_intervals, snr)
    obs_normalized_spectra = pd.DataFrame(data=np.column_stack((obs_lambda_flat,obs_data_norm_flat)),columns=['wl','flux'])


    create_atm_model(teff, logg, feh, vtur, star)
    vmac = round(float(get_vmac(teff, logg)), 3)
#    vmac = round(float(0.0), 3)
    print(teff, logg, feh, vtur, vmac)

    obs_lambda = np.array(obs_normalized_spectra['wl'])
    CDELT1 = delta_lambda
    p = [vrot_test]

    
    gap = obs_lambda[-1] - obs_lambda[0]
    if gap <= 450:
        lambda_i_values = [round(obs_lambda[0], 3)]
        lambda_f_values = [round(obs_lambda[-1], 3)]
    elif gap > 450 and gap <= 900:
        lambda_i_values = [round(obs_lambda[0], 3), round(obs_lambda[int(len(obs_lambda)/2)], 3)]
        lambda_f_values = [round(obs_lambda[int(len(obs_lambda)/2)-1], 3), round(obs_lambda[-1], 3)]
    elif gap > 900:
        lambda_i_values = [round(obs_lambda[0], 3), round(obs_lambda[int(len(obs_lambda)/3)], 3), round(obs_lambda[int(len(obs_lambda)/3*2)], 3)]
        lambda_f_values = [round(obs_lambda[int(len(obs_lambda)/3)-1], 3), round(obs_lambda[int(len(obs_lambda)/3*2)-1], 3), round(obs_lambda[-1], 3)]

    synth_data = []  # all flux values from model
    synth_lambda = []  # all wavelength points from model

    for lambda_i, lambda_f in zip(lambda_i_values, lambda_f_values):
        moog_fe(star, p, vmac, lambda_i, lambda_f, ldc, CDELT1, instr_broad)
        with open(RUN_PATH+'synth_fe.asc') as asc:
            for x in asc:
                if x[0] == ' ':
                    entry = x.rstrip().split()
                    synth_lambda.append(float(entry[0]))
                    synth_data.append(float(entry[1]))

    synth_data_fe = []
    synth_lambda_fe = []

    fe_intervals_list = [row for row in fe_intervals[['ll_li', 'll_lf','ll_si','ll_sf']].to_numpy()]

    for i,(ll_li, ll_lf, ll_si, ll_sf) in enumerate(fe_intervals_list):
        select_sll = np.where((synth_lambda >= ll_si) & (synth_lambda <= ll_sf))[0]
        synth_data_fe.extend(list(np.array(synth_data)[select_sll]))
        synth_lambda_fe.extend(list(np.array(synth_lambda)[select_sll]))

    obs_flux = np.array(obs_normalized_spectra['flux'])
    synth_data_fe = np.array(synth_data_fe)

    return obs_lambda, obs_flux, synth_data_fe


def manual_test(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test):
    obs_lambda, obs_flux, synth_data_fe = create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test)
    plt.plot(obs_lambda, obs_flux)
    plt.plot(obs_lambda, synth_data_fe)
    plt.show()



def manual_test2(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test):
    obs_lambda, obs_flux, synth_data_fe = create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test)
#    plot_line_profile(5522.450, obs_lambda, obs_flux, synth_data_fe, space = 4)
#    plot_line_profile(5560.220, obs_lambda, obs_flux, synth_data_fe, space = 4)
    plot_line_profile(5633.950, obs_lambda, obs_flux, synth_data_fe, space = 4)


def find_vsini_mine(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vi=0, vf=8, vstep=0.2):
    lines = np.loadtxt('linelist/linesfitted.ares', usecols= (0,), unpack=True)
    vrot_vec = np.arange(vi,vf,vstep)
    eval1_mat = []
    eval2_mat = []
    for vrot_test in vrot_vec:
        obs_lambda, obs_flux, synth_data_fe = create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test)
        eval1_vec = []
        eval2_vec = []
        for l in lines:
            eval1 , eval2 = plot_line_profile(l, obs_lambda, obs_flux, synth_data_fe, space = 4, plot_flag=False)
            print('Eval line profile fit:', vrot_test, eval1, eval2)
            eval1_vec.append(eval1)
            eval2_vec.append(eval2)
        eval1_mat.append(eval1_vec)
        eval2_mat.append(eval2_vec)

    eval1_mat = np.array(eval1_mat)
    eval2_mat = np.array(eval2_mat)
    a = (eval1_mat, eval2_mat, vrot_vec, lines)
    with open('fitsun2.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(len(lines)):
        print('Line', i, lines[i])
        eval1_vec = np.array(eval1_mat[:,i])
        eval2_vec = np.array(eval2_mat[:,i])
        plt.plot(vrot_vec, eval1_vec, label='Flux')
        plt.plot(vrot_vec, eval2_vec, label='Derivative')
        plt.show()

def test_load(line = -1):
    with open('fit2.pickle', 'rb') as handle:
        eval1_mat, eval2_mat, vrot_vec, lines = pickle.load(handle)
    if line == -1:
        for i in range(len(lines)):
            print('Line', i, lines[i])
            eval1_vec = np.array(eval1_mat[:,i])
            eval2_vec = np.array(eval2_mat[:,i])
            plt.plot(vrot_vec, eval1_vec/np.max(eval1_vec), label='Flux')
            plt.plot(vrot_vec, eval2_vec/np.max(eval2_vec), label='Derivative')
            plt.title('Line at %s A' % lines[i])
            plt.show()
    else:
        i = line
        print('Line', i, lines[i])
        eval1_vec = np.array(eval1_mat[:,i])
        eval2_vec = np.array(eval2_mat[:,i])
        plt.plot(vrot_vec, eval1_vec/np.max(eval1_vec), label='Flux')
        plt.plot(vrot_vec, eval2_vec/np.max(eval2_vec), label='Derivative')
        plt.title('Line at %s A' % lines[i])
        plt.show()

def get_ifit(npf, eval_vec, vrot_vec):
    ifit = np.arange(npf) + np.argmin(eval_vec) - int(npf/2)
    if ifit[0] < 0:
        ifit = np.arange(npf)
    if ifit[-1] >= len(vrot_vec):
        ifit = np.arange(npf) + len(vrot_vec) - npf            
    return ifit


def test_global(itest = -1, plot_flag=True):
    with open('fitsun2.pickle', 'rb') as handle:
        eval1_mat, eval2_mat, vrot_vec, lines = pickle.load(handle)
    if itest == -1:
        ivals = range(len(lines))
    else:
        ivals = [itest]
    vrot_lines1 = []
    vrot_lines2 = []
    for i in ivals:
        eval1_vec = np.array(eval1_mat[:,i])
        eval2_vec = np.array(eval2_mat[:,i])
        ifit1 = get_ifit(7, eval1_vec, vrot_vec)
        ifit2 = get_ifit(7, eval2_vec, vrot_vec)
        p1 = np.polyfit(vrot_vec[ifit1], eval1_vec[ifit1], 2)
        p2 = np.polyfit(vrot_vec[ifit2], eval2_vec[ifit2], 2)
        min1 = -p1[1]/(2.*p1[0])
        min2 = -p2[1]/(2.*p2[0])
        vrot_lines1.append((min1,np.polyval(p1, min1)))
        vrot_lines2.append((min2,np.polyval(p2, min2)))
        if plot_flag and False:
            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex = ax1)
            ax1.plot(vrot_vec, eval1_vec, label='Flux')
            ax1.plot(vrot_vec[ifit1], eval1_vec[ifit1], marker='o', c='r', ls='None')
            ax1.plot(vrot_vec, np.polyval(p1, vrot_vec), label='Polyfit')
            ax1.axvline(x=min1, color='k', ls='--')
            ax2.plot(vrot_vec, eval2_vec, label='Derivative')
            ax2.plot(vrot_vec[ifit2], eval2_vec[ifit2], marker='o', c='r', ls='None')
            ax2.plot(vrot_vec, np.polyval(p2, vrot_vec), label='Polyfit')
            ax2.axvline(x=min2, color='k', ls='--')
            ax1.set_title('Line at %s A' % lines[i])
            plt.show()
    if plot_flag and itest == -1:
        vrot_lines1 = np.array(vrot_lines1)
        vrot_lines2 = np.array(vrot_lines2)
        print('Final results:')
        print(np.median(vrot_lines1[:,0]), np.std(vrot_lines1[:,0]))
        print(np.median(vrot_lines2[:,0]), np.std(vrot_lines2[:,0]))
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex = ax1)
        ax1.plot(lines, vrot_lines1[:,0], marker='o')
        ax1.plot(lines, vrot_lines2[:,0], marker='o')
        ax1.axhline(y=np.median(vrot_lines1[:,0]), color='b', ls='--')
        ax1.axhline(y=np.median(vrot_lines2[:,0]), color='orange', ls='--')
        ax2.plot(lines, vrot_lines1[:,1]/np.median(vrot_lines1[:,1]), marker='o')
        ax2.plot(lines, vrot_lines2[:,1]/np.median(vrot_lines2[:,1]), marker='o')
        plt.show()



from scipy import interpolate
from scipy.optimize import root
def plot_line_profile(line, obs_lambda, obs_flux, synth_data_fe, space = 4, plot_flag=True):

    np_rv=10
    np_fit=20
    ind = np.where( (obs_lambda >= line - space) & (obs_lambda <=  line + space))

    obs_lambda = obs_lambda[ind]
    obs_flux = obs_flux[ind]
    synth_data_fe = synth_data_fe[ind]
    ind_c = np.where(obs_lambda >= line)[0][0]
    ind_l = np.arange(np_rv)+ind_c-int(np_rv/2)

    dobs_flux = np.gradient(obs_flux, obs_lambda)
    dsyn_flux = np.gradient(synth_data_fe, obs_lambda)
    pobs = interpolate.interp1d(obs_lambda[ind_l], dobs_flux[ind_l])
    psyn = interpolate.interp1d(obs_lambda[ind_l], dsyn_flux[ind_l])
    obs_zero = root(pobs, x0=line).x
    syn_zero = root(psyn, x0=line).x
    rv_corr = (obs_zero - syn_zero)/syn_zero * 299792.458
    print(obs_zero, syn_zero, rv_corr)

    obs_lambda_c = correct_rv(obs_lambda, rv_corr)
    flux_int = interpolate.interp1d(obs_lambda_c, obs_flux, fill_value='extrapolate', bounds_error=False)
    obs_flux = flux_int(obs_lambda)

    ind_c = np.where(obs_lambda >= line)[0][0]
    ind_l = np.arange(np_rv)+ind_c-int(np_rv/2)
    ind_l_fit = np.arange(np_fit)+ind_c-int(np_fit/2)
    dobs_flux = np.gradient(obs_flux, obs_lambda)
    dsyn_flux = np.gradient(synth_data_fe, obs_lambda)

    pobs = interpolate.interp1d(obs_lambda[ind_l], dobs_flux[ind_l])
    psyn = interpolate.interp1d(obs_lambda[ind_l], dsyn_flux[ind_l])
    obs_zero = root(pobs, x0=line).x
    syn_zero = root(psyn, x0=line).x
    rv_corr = (obs_zero - syn_zero)/syn_zero * 299792.458
    print(obs_zero, syn_zero, rv_corr)

    if plot_flag:
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex = ax1)
        ax1.plot(obs_lambda, obs_flux, label='Observed')
        ax1.plot(obs_lambda, synth_data_fe, label='Synthetic')
        ax2.plot(obs_lambda, dobs_flux, label='Observed', linestyle='--')
        ax2.plot(obs_lambda[ind_l_fit], dobs_flux[ind_l_fit], marker='o', c='r', ls='None')
        ax2.plot(obs_lambda[ind_l], dobs_flux[ind_l], marker='x', c='k', ls='None')
        ax2.plot(obs_lambda, dsyn_flux, label='Synthetic', linestyle='--', marker='o')
        ax2.plot(obs_lambda[ind_l_fit], dsyn_flux[ind_l_fit], marker='o', c='r', ls='None')
        ax2.plot(obs_lambda[ind_l], dsyn_flux[ind_l], marker='x', c='k', ls='None')
        ax2.plot(obs_lambda[ind_l], pobs(obs_lambda[ind_l]), label='Observed', linestyle='--', color='k')
        ax2.plot(obs_lambda[ind_l], psyn(obs_lambda[ind_l]), label='Synthetic', linestyle='--', color='k')
        ax2.axhline(y=0, color='k', ls='--')
        ax2.axvline(x=obs_zero, color='b', ls='--', label='Obs zero')
        ax2.axvline(x=syn_zero, color='r', ls='--', label='Syn zero')
        ax1.legend()
        ax2.legend()
        plt.show()

    eval1 = np.sum((obs_flux[ind_l_fit] - synth_data_fe[ind_l_fit])**2)
    eval2 = np.sum((dobs_flux[ind_l_fit] - dsyn_flux[ind_l_fit])**2)
    return eval1, eval2

def correct_obs_flux(obs_lambda, obs_flux, synth_lambda, synth_flux):

    pass


def test_line_fit(line, vrot_test, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, spectrum, star):
    obs_lambda, obs_flux, synth_data_fe = create_obs_synth_spec(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, vrot_test)
    plot_line_profile(line, obs_lambda, obs_flux, synth_data_fe, space = 4, plot_flag=True)
    pass


### Main program:
def main():

    print("Running Dir:", RUN_PATH)
    #return
    #
    # ACTUAL CODE STARTS HERE
    #

    start_time = time.time()

    #
    # READ INTERVALS OF FE LINES
    #
    fe_intervals = pd.read_csv(LINELIST_PATH+'vsini_intervals.list', sep='\t')

    star = "WASP-34_ESPRESSO"
    teff = 5684
    eteff = 50
    logg = 4.36
    feh  = 0.03
    efeh = 0.04
    vtur = 0.916
    snr  = 550
    ldc  = 0.61
    instr_broad = 0.042
    #spectrum = "/home/sousasag/Programas/Vsini/Vsini2_Vardan/myVsini/spectra/WASP-34_ESPRESSO.fits"


    star = "TIC61024636_ESPRESSO"
    teff = 5261
    eteff = 84
    logg = 4.23
    feh  = 0.318
    efeh = 0.054
    vtur = 0.945
    snr  = 300
    ldc  = 0.58    #https://exoctk.stsci.edu/limb_darkening
    instr_broad = 0.042
    spectrum = "/home/sousasag/Data/spectra_solene_jupiterHosts/TIC61024636/TIC61024636_waveair_rv.fits"


    star = "TOI_5624_HARPSN"
    teff = 5327
    eteff = 64
    logg = 4.453
    feh  = -0.02
    efeh = 0.04
    vtur = 0.88
    snr  = 300
    ldc  = 0.6    #https://exoctk.stsci.edu/limb_darkening
    instr_broad = 0.055
    if LOC_FLAG == "WORK":
        spectrum = "/home/sousasag/Investigador/spectra/CHEOPS_TS3/SPEC/Axis1/TOI_5624_HARPS_N_CoAdded.fits"
    else:
        spectrum = "Data/TOI_5624_HARPS_N_CoAdded.fits"

    star = "Sun_HARPS"
    teff = 5777
    eteff = 80
    logg = 4.44
    feh  = 0.00
    efeh = 0.05
    vtur = 1.00
    snr  = 250
    ldc  = 0.65   #https://exoctk.stsci.edu/limb_darkening
    instr_broad = 0.055
    spectrum = "/home/sousasag/Data/spectra/sun_harps_ganymede.fits"


    test_line_fit(6698.67, 1, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals, spectrum, star)    
#    test_load()
#    test_global(-1)

    return

#    manual_test2(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals,4)
    find_vsini_mine(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    print (star, teff, logg, feh, vtur, snr, ldc, instr_broad, spectrum)
    return


    print (star, teff, logg, feh, vtur, snr, ldc, instr_broad, spectrum)


#    vrot, vrot_err, vmac, status = get_vsini(star, spectrum, teff, feh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
#    creating_final_synth_spectra(vrot, star, spectrum, teff, feh, vtur, logg, snr, fe_intervals, ldc, instr_broad)
#    print ('results', star, teff, logg, feh, snr, spectrum, vrot, vrot_err, vmac, status)

#With Error propagation
    vrot, vrot_err, vmac, status, vsini_final_err = get_vsini_error(star, spectrum, teff, eteff, feh, efeh, vtur, logg, snr, ldc, instr_broad, fe_intervals)
    creating_final_synth_spectra(vrot, star, spectrum, teff, feh, vtur, logg, snr, fe_intervals, ldc, instr_broad)
    print ('results', star, teff, logg, feh, snr, spectrum, vrot, vrot_err, vmac, status, vsini_final_err)


if __name__ == "__main__":
    main()



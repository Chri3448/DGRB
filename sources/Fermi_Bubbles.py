import numpy as np
import pdb
import scipy
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import astropy.units as units
from astropy.io import fits
from astropy import wcs
import healpy as hp

class Fermi_Bubbles:
    
    def __init__(self, data_path):
        #directory that holds Fermi data
        self.path = data_path
        spectrum_data = np.loadtxt(self.path + '/data/Fermi_Bubbles/1005_5480_fig14_whole_bubble.csv', delimiter=',', skiprows=1)
        self.spectrum = self.get_interpolated_spectrum(spectrum_data[:,0], spectrum_data[:,1])

    def get_interpolated_spectrum(self, data_x, data_y):
        dNdE = data_y/data_x**2/1e3 #flux/Mev
        f = interpolate.interp1d(np.log10(data_x*1e3), np.log10(dNdE), fill_value = 'extrapolate')
        return lambda e: 10**f(np.log10(e))
        
    def get_partial_map(self, angular_cut, Emin, Emax, N_Ebins, N_side = 64):
        file = self.path + '/data/Fermi_Bubbles/template_bub.npy'
        data = np.load(file)
        center = hp.ang2vec(np.pi/2, 0)
        close_pix_i = hp.query_disc(N_side, center, angular_cut)
        npix = int(hp.nside2npix(N_side))
        angs = hp.pix2ang(N_side, close_pix_i)
        raw_map = hp.get_interp_val(data, angs[0], angs[1])
        output_energies = np.linspace(Emin, Emax, N_Ebins + 1)
        vals = np.zeros((N_Ebins + 1, close_pix_i.size))
        for ei, e in enumerate(output_energies):
            normalization = self.spectrum(e)/(4*np.pi/data.size)/np.sum(data)
            vals[ei, :] = raw_map*normalization
        return vals, output_energies, close_pix_i
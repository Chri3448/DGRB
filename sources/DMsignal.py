import torch
import numpy as np
import scipy.interpolate as interp
import pdb

class DMsignal:
    def __init__(self, directory, channel):
        self.channel = channel
        self.backFile = directory + 'AtProduction_gammas.dat'
        self.all_data = np.genfromtxt(self.backFile, names = True)
        self.mass_MeV = 0.
        
    def set_spectrum_interpolator(self, channel, mass_MeV):
        # Create an interpolation object (RectBivariateSpline) that
        # interpolates the DM annihilation table in mass and energy
        
        # Restrict table to a range around the mass of the dark matter particle
        # all this may not be necessary - could maybe just use the full table
        # when constructing the interpolator.  
        mass_GeV = mass_MeV/1000.
        if type(mass_GeV) == type(torch.tensor([])):
            near_desired_mass = np.where((self.all_data['mDM'] > 0.5*mass_GeV.numpy()) & (self.all_data['mDM'] < 2.0*mass_GeV.numpy()))[0]
        else:
            near_desired_mass = np.where((self.all_data['mDM'] > 0.5*mass_GeV) & (self.all_data['mDM'] < 2.0*mass_GeV))[0]
        mass_inrange = self.all_data['mDM'][near_desired_mass]
        log10x_inrange = self.all_data['Log10x'][near_desired_mass]
        dNdlog10x_inrange = self.all_data[channel][near_desired_mass]
        num_unique_mass = len(np.unique(mass_inrange))
        num_unique_x =  int(len(dNdlog10x_inrange)/num_unique_mass)
        log10x_table = np.reshape(log10x_inrange, (num_unique_mass,num_unique_x))
        mass_table = np.reshape(mass_inrange, (num_unique_mass,num_unique_x))
        dNdlog10x_table = np.reshape(dNdlog10x_inrange, (num_unique_mass,num_unique_x))
        interp_func = interp.RectBivariateSpline(mass_table[:,0], log10x_table[0,:], dNdlog10x_table)
        self.spectrum_interp = interp_func
        self.mass_MeV = mass_MeV
        
    def get_raw_data(self, channel, mass_MeV):
        mass_GeV = mass_MeV/1000.
        i = np.where(self.all_data['mDM'] == mass_GeV)[0]
        return mass_MeV*10**self.all_data['Log10x'][i], self.all_data[channel][i]/(10**self.all_data['Log10x'][i]*np.log(10.))/mass_MeV

    def get_dNdE(self, desired_E_MeV, channel, mass_MeV):
        #If the mass or channel is different from what we have stored
        #then we need to re-initialize the spectrum interpolator
        if (self.channel != channel or self.mass_MeV != mass_MeV):
            self.set_spectrum_interpolator(channel, mass_MeV)
            
        x_input = desired_E_MeV/mass_MeV
        mass_GeV = mass_MeV/1000.
        
        desired_x = desired_E_MeV/mass_MeV
        desired_log10x = np.log10(desired_x)
        dNdlog10x = self.spectrum_interp.ev(mass_GeV, desired_log10x)
        dNdx = dNdlog10x/(desired_x*np.log(10.))
        dNdE_MeV = dNdx/mass_MeV
        return np.where(desired_E_MeV > mass_MeV, 0, dNdE_MeV)

    def get_default_model(self):
        if (self.channel == 'Tau'):
            A_DM = 200.#2000.0
            A_BG = 1.0
            mass_DM_MeV = 200000.
        if (self.channel == 'b'):
            A_DM = 5.0
            A_BG = 1.0
            mass_DM_MeV = 50000.
        return {'A_DM':A_DM, 'A_BG':A_BG, 'mass_DM_MeV':mass_DM_MeV}

import numpy as np
import healpy as hp

class smoothDM():

    def __init__(self, **kwargs):
        self.N_side = kwargs['N_side']
        self.N_pix = 12*self.N_side**2
        self.Omega_pixel = 4*np.pi/self.N_pix
        self.theta_cutoff = kwargs['theta_cutoff']
        self.halo_dist = kwargs['halo_dist']
        self.Rs = kwargs['Rs']
        self.set_mass_func(kwargs['mass_func'])

    def get_pixels(self):
        pix_i = np.linspace(0, self.N_pix-1, self.N_pix, dtype = 'int')
        center = hp.ang2vec(np.pi/2, 0)
        close_pix_i = hp.query_disc(self.N_side, center, self.theta_cutoff)

        return close_pix_i

    def J_factor(self, pix, mass_func_params):
        theta = hp.rotator.angdist(hp.pix2ang(self.N_side, pix), (np.pi/2, 0))
        l = np.concatenate((np.flip(np.exp(np.linspace(np.log(self.halo_dist + 1), 0, 1000))) - 1, np.exp(np.linspace(np.log(self.halo_dist), np.log(self.halo_dist + 2*self.Rs), 1000))[1:]))
        dl = l[1:] - l[:-1]
        r = np.sqrt(l**2 + self.halo_dist**2 - 2*np.tile(l, (np.size(theta), 1))*self.halo_dist*np.cos(np.tile(theta, (np.size(l),1)).T))
        J = np.sum(self.mass_func(r[:,:-1], *mass_func_params)**2*np.tile(dl, (np.size(theta), 1)), axis = 1)
        
        return J

    def get_map(self, DM_mass, cross_sec, dNdE, mass_func_params):
        pix = self.get_pixels()
        J = self.J_factor(pix, mass_func_params)
        flux = cross_sec/(8*np.pi*DM_mass**2)*np.tile(dNdE, (np.size(pix), 1))*(np.tile(J, (np.size(dNdE), 1)).T)

        return flux, pix
    
    '''
    Spacial distribution functions
    '''
    def set_mass_func(self, func_name):
        if func_name == 'gNFW':
            self.mass_func = self.gNFW
    
    def gNFW(self, r, r_s, rho_s, gamma):
        
        return rho_s * (r/r_s)**-gamma * (1+(r/r_s))**(-3+gamma)

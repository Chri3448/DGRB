import numpy as np
import pdb
import scipy
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import astropy.units as units
from astropy.io import fits
from astropy import wcs
import healpy as hp
import csv

class FermiBackgrounds:
    
    def __init__(self, fermi_data_path):
        #directory that holds Fermi data
        self.path = fermi_data_path
                   
    def get_isotropic_background_spectrum(self):
        # cm−2 s−1 sr−1 MeV−1
        backFile = self.path + '/data/FermiData/iso_P8R3_SOURCE_V3_v1.txt'
        with open(backFile, "r") as archive:
            reader = csv.reader(archive, delimiter=' ')
            e = []
            dnde = []
            for row in reader:
                e.append(float(row[0]))
                dnde.append(float(row[1]))
        return np.array(e), np.array(dnde)        

    def get_isotropic_background_spectrum_func(self):
        ee, dnde = self.get_isotropic_background_spectrum()
        #should maybe use logarithmic interpolation here?
        return interpolate.interp1d(ee, dnde)

    def get_mean_isotropic_flux(self, Emin, Emax):
        # Do integral over energy to get total flux in cm-2 s-2 sr-1
        dnde_func = self.get_isotropic_background_spectrum_func()
        e = np.geomspace(Emin, Emax, 200)
        de = e[1:] - e[:-1]
        dnde = dnde_func(e)
        total_flux = np.sum(0.5*de*(dnde[1:] + dnde[:-1]))
        return total_flux        
    
    def get_nonistropic_background(self, N_side = 64):
        bg_file = self.path + '/data/FermiData/gll_iem_v07.fits'
        bg_data = fits.open(self.path + bg_file)
        w = wcs.WCS(bg_data[0].header)
        dims = bg_data[0].shape
        N_energy = dims[0]
        N_pix = 12*N_side**2
        map_all = np.zeros((N_energy, N_pix))
        for ri in range(0,dims[1]):
            cols = np.arange(0,dims[2])
            coord = w.pixel_to_world(cols, ri,0)
            l, b = coord[0].l.deg, coord[0].b.deg
            good = np.where(np.isfinite(l))
            vals_good = bg_data[0].data[:,ri,good]
            pix_indices = hp.ang2pix(N_side, l[good], b[good], lonlat = True)
            unique_pix = np.unique(pix_indices)
            for ui in range(0,len(unique_pix)):
                match = np.where(pix_indices == unique_pix[ui])[0]
                #average over all pixels falling into this healpix pixel        
                map_all[:,unique_pix[ui]] = np.mean((vals_good[:,0,:])[:,match], axis = 1)
        return {'galactic_bg':map_all, 'energies_MeV':np.copy(bg_data[1].data['energy'])}
    
    def get_partial_nonistropic_background(self, angular_cut, Emin, Emax, N_Ebins, N_side = 64):
        file = self.path + '/data/FermiData/gll_iem_v07.fits'
        hdul = fits.open(file)
        energy = np.concatenate(hdul['ENERGIES'].data)
        data = np.concatenate((hdul['PRIMARY'].data[:,:,1440:], hdul['PRIMARY'].data[:,:,:-1440]), axis=2)
        data = np.flip(data, axis = 2)
        data = np.concatenate((data, data[:,:, 0:1]), axis=2)
        data = data.T
        center = hp.ang2vec(np.pi/2, 0)
        close_pix_i = hp.query_disc(N_side, center, angular_cut)
        lon = np.linspace(0, 360, 2881)
        lat = np.linspace(-90, 90, 1441)
        npix = int(hp.nside2npix(N_side))
        angs_lon, angs_lat = hp.pix2ang(N_side, close_pix_i, lonlat=True)
        output_energies = np.linspace(Emin, Emax, N_Ebins + 1)
        vals = np.zeros((N_Ebins + 1, close_pix_i.size))
        for ei, e in enumerate(output_energies):
            angs_engs = np.array([angs_lon, angs_lat, e*np.ones(close_pix_i.size)]).T
            vals[ei, :] = interpolate.interpn((lon, lat, energy), data, angs_engs)
        return vals, output_energies, close_pix_i

    def get_masked_isotropic_flux(self, galactic_bg_file, gal_lat_cut, gal_cent_cut, Emin, Emax):

        # Load galactic background model
        import pickle as pk
        gal_bg_data = pk.load(open(galactic_bg_file,'rb'))

        # Galactic maps for each energy
        map_all = gal_bg_data['galactic_bg']
        N_pix = map_all.shape[1]
        N_side = hp.npix2nside(N_pix)
        
        # Sum galactic model over energies
        E_cents = gal_bg_data['energies_MeV']
        E_fine = np.exp(np.linspace(np.log(Emin), np.log(Emax), num = 200))
        dE_fine = E_fine[1:] - E_fine[:-1]

        gal_bg_map = np.zeros(N_pix)
        for ii in range(0,N_pix):
            #if (ii % 10000 == 0):
            #    print(ii)
            ff = interpolate.interp1d(E_cents, map_all[:,ii])
            to_integrate = ff(E_fine)
            gal_bg_map[ii] = np.sum(0.5*dE_fine*(to_integrate[1:] + to_integrate[:-1]))

        # Isotropic background model
        mean_iso_bg_flux = self.get_mean_isotropic_flux(Emin, Emax)

        # Total background model
        total_model = gal_bg_map + mean_iso_bg_flux

        # Get mask
        mask = self.get_mask(N_side, gal_lat_cut, gal_cent_cut)
        
        # Our final estimate is mean over unmasked region
        in_mask = np.where(mask == 1)[0]
        flux_estimate = np.mean(total_model[in_mask])

        return flux_estimate

    
    def get_mask(self, N_side, gal_lat_cut, gal_cent_cut):
        N_pix = hp.nside2npix(N_side)
    
        # Mask that combines galactic latitude cut and galactic center cut
        lon, lat = hp.pix2ang(N_side, np.arange(N_pix), lonlat = True)
        mask = np.zeros(N_pix)
        good = np.where(np.abs(lat) > gal_lat_cut)[0]
        mask[good] = 1.0

        # Get angles with respect to galactic center
        def hav(theta):
            return np.sin(theta/2.)**2.

        def ang_sep(lon1, lat1, lon2, lat2):
            # Haversine formula
            theta = 2.*np.arcsin(np.sqrt(hav(lat1 - lat2) + np.cos(lat1)*np.cos(lat2)*hav(lon1-lon2)))
            return theta
        
        # Mask the galactic center
        angle_to_galcenter = (180./np.pi)*ang_sep(0., 0., lon*np.pi/180., lat*np.pi/180.)
        near_center = np.where(angle_to_galcenter < gal_cent_cut)[0]
        mask[near_center] = 0.

        return mask

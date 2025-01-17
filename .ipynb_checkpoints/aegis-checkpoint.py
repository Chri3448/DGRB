import numpy as np
import pdb
import scipy as sp
import scipy.interpolate
import scipy.integrate as integrate
import astropy.units as units
import astropy.cosmology as cosmo
import healpy as hp
import torch
from astropy.io import fits
import copy

'''
Astrophysical Event Generator for Integration with Simulation-based inference
The class is used to generate simulations of photon maps.
'''

class aegis():

    def __init__(self, abundance_luminosity_and_spectrum_list, source_class_list, parameter_range, energy_range, luminosity_range, max_radius, exposure, angular_cut = np.pi, lat_cut = 0, flux_cut = np.inf, cosmology = None, verbose = False):
        #super().__init__(parameter_range)

        self.cosmology = cosmology
        if self.cosmology:
            if self.cosmology not in cosmo.realizations.available and type(self.cosmology) != cosmo.FlatLambdaCDM:
                raise Exception('No valid cosmology given. Try one of these preloaded cosmologies: ' + ', '.join(cosmo.realizations.available) + '. Alternatively, give a custom cosmology of the astropy.cosmology.FlatLambdaCDM class.')
            if self.cosmology in cosmo.realizations.available:
                self.cosmology = getattr(cosmo, self.cosmology)
        
        self.GC_to_earth = 8.5 #kpc
        
        #PDF_model_list contains the PDF functions and spectrum functions
        #that represent our model
        self.abun_lum_spec = abundance_luminosity_and_spectrum_list
        
        #how should each source be handeled?
        self.source_class_list = source_class_list
        
        #allowed parameter range
        self.parameter_range = parameter_range

        #allowed mass range
        self.Lmin = luminosity_range[0]
        self.Lmax = luminosity_range[1]
        
        #allowed maximum distance from galactic center
        self.Rmax = max_radius
        
        #energy range within which photons are generated and survive the final masking cut, respectively
        self.Emin_gen = energy_range[0]*0.5
        self.Emax_gen = energy_range[1]*1.5
        self.Emin_mask = energy_range[0]
        self.Emax_mask = energy_range[1]
        
        #exposure of detector (converted from cm^2yr to kpc^2s)
        self.exposure = exposure*(units.cm.to('kpc')**2)*units.yr.to('s')
        
        #anglular radii out to which photons are generated and survive the final masking cut, respectively, from Earth's perspective
        self.angular_cut_gen = angular_cut*1.5
        self.angular_cut_mask = angular_cut
        
        #angular distances from galactic equator beyond which photons are generated and survive the final masking cut, respectively,, from Earth's perspective
        self.lat_cut_gen = lat_cut*0.5
        self.lat_cut_mask = lat_cut
            
        #flux above which point sources are not generated, from Earth's perspective
        self.flux_cut = flux_cut

        #Number of types of sources contributing photons
        self.N_source_classes = len(abundance_luminosity_and_spectrum_list)

        self.verbose = verbose
        if (self.verbose):
            print("Analysis Type: " + self.analysis_type)
            print("N_parameters = ", self.N_parameters)
            print("map = ", self.is_map_list)
            print("source_classes = ", self.source_class_list)
            print("parameter min = ", self.param_min)
            print("parameter max = ", self.param_max)
            print("Emin_gen = ", self.Emin_gen)
            print("Emax_gen = ", self.Emax_gen)
            print("Emin_mask = ", self.Emin_mask)
            print("Emax_mask = ", self.Emax_mask)
            print("Rmax = ", self.Rmax)
            print("Lmin = ", self.Lmin)
            print("Lmax = ", self.Lmax)
            print("exposure = ", self.exposure, " kpc^2 s")
            print("angular_cut_gen = ", self.angular_cut_gen)
            print("angular_cut_mask = ", self.angular_cut_mask)
            print("lat_cut_gen = ", self.lat_cut_gen)
            print("lat_cut_mask = ", self.lat_cut_mask)
            print("N_source_classes = ", self.N_source_classes)

    ##########################################################################
    '''
    Basic statistical functions

    '''
    ##########################################################################

    def searchsorted2d(self, a, b):
        # Courtesy of user Divakar on Stack Overflow
        # Inputs : a is (m,n) 2D array and b is (m,) 1D array.
        # Finds np.searchsorted(a[i,:], b[i])) in a vectorized way by
        # scaling/offsetting both inputs and then using searchsorted

        # Get scaling offset and then scale inputs
        s = np.r_[0,(np.maximum(a.max(1)-a.min(1)+1,b)+1).cumsum()[:-1]]
        a_scaled = (a+s[:,None]).ravel()
        b_scaled = b+s

        # Use searchsorted on scaled ones and then subtract offsets
        return np.searchsorted(a_scaled,b_scaled)-np.arange(len(s))*a.shape[1]

    def sample_from_uniform(self, N_samples):
        # Generate random parameter samples drawn from uniform distribution given
        # by parameter ranges
        output_samples = np.zeros((N_samples, self.N_parameters))
        for ii in range(0,self.N_parameters):
            output_samples[:,ii] = np.random.uniform(low = self.param_min[ii],\
                                                     high = self.param_max[ii],\
                                                     size = N_samples)
        return output_samples
    
    def draw_from_pdf(self, cc, Pc, Ndraws):
        # draw random counts from P(c)
        cdf = np.cumsum(Pc)
        rands = np.random.rand(Ndraws)
        # Draw Ndraws times from Pc
        d_vec = np.searchsorted(cdf, rands)
        return d_vec
    
    #Takes a 2D array. pdf[x,y] should equal pdf(x,y). Returns two 1D arrays of x and y indices.
    #pdf*dx*dy must be passed to this func as pdf for normalization. If x or y are not linspaced, the pdf dimensions should be (n-1,m-1),
    #and the last indices of x and y will not be drawn
    def draw_from_2D_pdf(self, pdf, Ndraws = 0):
        if Ndraws == 0:
            Ndraws = int(round(np.sum(pdf)))
        flipped = False
        if np.shape(pdf)[0] > np.shape(pdf)[1]:
            pdf = pdf.T
            flipped = True
        x_pdf = np.sum(pdf, axis = 1)/np.sum(pdf)
        x_cdf = np.cumsum(x_pdf)
        x_rands = np.random.rand(Ndraws)
        x_indices = np.searchsorted(x_cdf, x_rands)
        y_cdfs = np.cumsum(pdf, axis = 1)/np.tile(np.sum(pdf, axis = 1), (np.size(pdf[0,:]),1)).T
        y_rands = np.random.rand(Ndraws)
        y_indices = np.zeros(np.size(x_indices), dtype = 'int')
        for i in range(np.size(pdf[:,0])):
            source_positions = np.where(x_indices == i)
            y_indices[source_positions] = np.searchsorted(y_cdfs[i,:], y_rands[source_positions])
        if flipped:
            return y_indices, x_indices
        else:
            return x_indices, y_indices

    def draw_from_isotropic_EPDF(self, params, source_index, exposure, Sangle, npix):
        '''
        Given a binned spectrum, draw photon counts in energy pixels for npix pixels

        Output will have dimensions (N_pix, N_energy_bins)
        '''
        if (not self.is_istropic_list[source_index]):
            print("attempting to draw from isotropic source that is not isotropic!!!")
        
        output = np.zeros((npix, self.nEbins))

        #Compute total flux in each energy bin (THIS IS INTEGRATED OVER ENERGY)
        E_flux = np.zeros(self.nEbins)
        for ei in range(0,self.nEbins):
            e = np.geomspace(self.Ebins[ei], self.Ebins[ei+1], 200)
            de = e[1:] - e[:-1]
            dnde = self.PDF_spec[source_index][1](params, e, self.args)
            E_flux[ei] = np.sum(0.5*de*(dnde[1:] + dnde[:-1]))

        #In principle, this could handle non-istropic PDFs very easily
        if (self.is_poisson_list[source_index]):
            #if poisson, then compute total flux
            dE_bins = self.Ebins[1:] - self.Ebins[:-1]
            total_flux = np.sum(E_flux)
            # draw total photon count for each pixel
            mean_photon_count_per_pix = exposure*Sangle*total_flux
            photon_counts_per_pix = np.random.poisson(lam = mean_photon_count_per_pix, size = npix)
        else:
            Pc = self.PDF_spec[source_index][0](params, self.Cbins, self.args)
            photon_counts_per_pix = self.draw_from_pdf(self.Cbins, Pc, npix)

        if (self.nEbins == 1):
            output[:,0] = photon_counts_per_pix
        if (self.nEbins > 1):
            #Set up spectral weights - these are the fraction of photons that fall into each energy bin            
            spec_weights = E_flux/np.sum(E_flux)
            
            #draw np.sum(photon counts per pix) times from a PDF
            #this is energy bin of all photons
            ebin_arr = self.draw_from_pdf(np.arange(len(spec_weights)), spec_weights, np.sum(photon_counts_per_pix))
                                          
            pix_indices = np.repeat(np.arange(npix),photon_counts_per_pix)
            for ei in range(0,self.nEbins):
                in_bin = np.where(ebin_arr == ei)[0]
                output[:,ei] = np.bincount(pix_indices[in_bin], minlength = npix)
                
        return output.astype('int')
    
    ##########################################################################
    '''
    Simulation generating functions

    '''
    ##########################################################################

    def create_sources(self, input_params, grains = 1000, epsilon = 0):
        '''
        This function creates a list of sources, where each source has a radial distance, mass, and luminosity
        '''
        source_info = {'luminosities': np.array([]),
                       'distances': np.array([]),
                       'single_p_distances': np.array([]),
                       'angles': np.zeros((0, 2)),
                       'single_p_angles': np.zeros((0, 2)),
                       'types': np.array([]),
                       'single_p_types' : np.array([])}

        # Loop over all source types
        self.N_source_classes = len(self.abun_lum_spec)
        for si in range(self.N_source_classes):
            if self.source_class_list[si] == 'isotropic_diffuse' or self.source_class_list[si] == 'healpix_map':
                continue

            elif self.source_class_list[si] == 'isotropic_faint_multi_spectra' or self.source_class_list[si] == 'isotropic_faint_single_spectrum':

                # Draw radii and luminosities from RL abundance
                RL = self.abun_lum_spec[si][0]
                radii, luminosities, single_p_radii = self.draw_luminosities_and_radii(input_params, RL, grains=grains, epsilon=epsilon)
                num_sources = np.size(luminosities)
                num_single_p_sources = np.size(single_p_radii)

                # Draw angles for every source [theta, phi]
                angles = np.ones([num_sources, 2])
                angles[:,0] = np.arccos(1 - 2*np.random.rand(num_sources))
                angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num_sources)
                single_p_angles = np.ones([num_single_p_sources, 2])
                single_p_angles[:,0] = np.arccos(1 - 2*np.random.rand(num_single_p_sources))
                single_p_angles[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = num_single_p_sources)
            
            elif self.source_class_list[si] == 'independent_spherical_multi_spectra' or self.source_class_list[si] == 'independent_spherical_single_spectrum':
                
                # Draw positions for each source relative to galactic center
                R = self.abun_lum_spec[si][0][0]
                Theta = self.abun_lum_spec[si][0][1]
                Phi = self.abun_lum_spec[si][0][2]
                radii, theta, phi = self.draw_spherical_positions_independent(input_params, R, Theta, Phi, grains = grains)
                num_sources = np.size(radii)
                angles = np.ones([num_sources, 2])
                angles[:,0] = theta
                angles[:,1] = phi

                # Draw luminosities for each source
                lums = np.exp(np.linspace(np.log(self.Lmin), np.log(self.Lmax), grains))
                lum_pdf = self.abun_lum_spec[si][1](lums[:-1], input_params) * (lums[1:]-lums[:-1])
                luminosities = lums[self.draw_from_pdf(lums[:-1], lum_pdf/np.sum(lum_pdf), num_sources)]
                
                # Single photon sources are not supported by this source class
                num_single_p_sources = 0
                single_p_angles = np.zeros([num_single_p_sources, 2])
                single_p_radii = np.zeros(num_single_p_sources)
                
            elif self.source_class_list[si] == 'independent_cylindrical_multi_spectra' or self.source_class_list[si] == 'independent_cylindrical_single_spectrum':
                
                # Draw positions for each source relative to galactic center
                R = self.abun_lum_spec[si][0][0]
                Z = self.abun_lum_spec[si][0][1]
                Phi = self.abun_lum_spec[si][0][2]
                r, z, phi = self.draw_cylindrical_positions_independent(input_params, R, Z, Phi, grains = grains)
                radii = np.sqrt(r**2 + z**2)
                num_sources = np.size(radii)
                angles = np.ones([num_sources, 2])
                angles[:,0] = np.pi/2 - np.arctan(z/r)
                angles[:,1] = phi

                # Draw luminosities for each source
                lums = np.exp(np.linspace(np.log(self.Lmin), np.log(self.Lmax), grains))
                lum_pdf = self.abun_lum_spec[si][1](lums[:-1], input_params) * (lums[1:]-lums[:-1])
                luminosities = lums[self.draw_from_pdf(lums[:-1], lum_pdf/np.sum(lum_pdf), num_sources)]
                
                # Single photon sources are not supported by this source class
                num_single_p_sources = 0
                single_p_angles = np.zeros([num_single_p_sources, 2])
                single_p_radii = np.zeros(num_single_p_sources)
                
            elif self.source_class_list[si] == 'dependent_multi_spectra' or self.source_class_list[si] == 'dependent_single_spectrum':
                print('not implemented')
                
            else:
                continue
            
            # Get the distance from Earth to each source
            x = self.GC_to_earth + radii*np.sin(angles[:,0])*np.cos(angles[:,1])
            y = radii*np.sin(angles[:,0])*np.sin(angles[:,1])
            z = radii*np.cos(angles[:,0])
            distances = np.sqrt(x**2 + y**2 + z**2)
            
            single_p_x = self.GC_to_earth + single_p_radii*np.sin(single_p_angles[:,0])*np.cos(single_p_angles[:,1])
            single_p_y = single_p_radii*np.sin(single_p_angles[:,0])*np.sin(single_p_angles[:,1])
            single_p_z = single_p_radii*np.cos(single_p_angles[:,0])
            single_p_distances = np.sqrt(single_p_x**2 + single_p_y**2 + single_p_z**2)
                
            # Get the angular positions as seen from Earth
            earth_angles = np.ones([num_sources, 2])
            earth_angles[:,0] = np.arccos(z/distances)
            earth_angles[:,1] = np.arccos(x/distances/np.sin(earth_angles[:,0]))
            earth_angles[:,1] = np.where(y > 0, earth_angles[:,1], 2*np.pi - earth_angles[:,1])

            single_p_earth_angles = np.ones([num_single_p_sources, 2])
            single_p_earth_angles[:,0] = np.arccos(single_p_z/single_p_distances)
            single_p_earth_angles[:,1] = np.arccos(single_p_x/single_p_distances/np.sin(single_p_earth_angles[:,0]))
            single_p_earth_angles[:,1] = np.where(single_p_y > 0, single_p_earth_angles[:,1], 2*np.pi - single_p_earth_angles[:,1])

            # Account for overdrawing single-photon sources
            prob_factors = ((self.GC_to_earth - single_p_radii)/single_p_distances)**2
            bad_source_indices = np.where(np.random.rand(num_single_p_sources) > prob_factors)
            single_p_radii = np.delete(single_p_radii, bad_source_indices)
            single_p_earth_angles = np.delete(single_p_earth_angles, bad_source_indices, axis = 0)
            single_p_distances = np.delete(single_p_distances, bad_source_indices)
            num_single_p_sources = np.size(single_p_radii)
            
            # Remove sources outside of the angular cut, inside of the latitude cut, and above the flux cut
            keep_i = np.where(np.logical_and(hp.rotator.angdist(np.array([np.pi/2, 0]), earth_angles.T) <= self.angular_cut_gen, 
                                            np.abs(np.pi/2 - earth_angles[:,0]) >= self.lat_cut_gen))[0]
            single_p_keep_i = np.where(np.logical_and(hp.rotator.angdist(np.array([np.pi/2, 0]), single_p_earth_angles.T) <= self.angular_cut_gen, 
                                            np.abs(np.pi/2 - single_p_earth_angles[:,0]) >= self.lat_cut_gen))[0]
            keep_i = keep_i[np.where(luminosities[keep_i]/(4*np.pi*(distances[keep_i]*units.kpc.to('cm'))**2) <= self.flux_cut)]
            num_sources = np.size(keep_i)
            num_single_p_sources = np.size(single_p_keep_i)
            luminosities = luminosities[keep_i]
            distances = distances[keep_i]
            single_p_distances = single_p_distances[single_p_keep_i]
            earth_angles = earth_angles[keep_i,:]
            single_p_earth_angles = single_p_earth_angles[single_p_keep_i,:]
            
            # Catalog the type of source
            types = si*np.ones(num_sources)
            single_p_types = si*np.ones(num_single_p_sources)
            
            source_info = {'luminosities':np.concatenate((source_info['luminosities'], luminosities)),
                           'distances':np.concatenate((source_info['distances'], distances)),
                           'single_p_distances':np.concatenate((source_info['single_p_distances'], single_p_distances)),
                           'angles':np.concatenate((source_info['angles'], earth_angles)),
                           'single_p_angles':np.concatenate((source_info['single_p_angles'], single_p_earth_angles)),
                           'types': np.concatenate((source_info['types'], types)),
                           'single_p_types': np.concatenate((source_info['single_p_types'], single_p_types))}

        if (self.verbose):
            print('Sources generated')
            for si in range(self.N_source_classes):
                num_mps = np.size(np.where(source_info['types'] == si))
                num_sps = np.size(np.where(source_info['single_p_types'] == si))
                print(f'Multi-photon sources of type {si}: {sum_mps}')
                print(f'Single-photon sources of type {si}: {sum_sps}')
        
        return source_info

    def generate_photons_from_sources(self, input_params, source_info, grains = 1000):
        '''
        Function returns list of photon energies and sky positions
        '''
        # Calculate mean expected flux from each source
        mean_photon_counts = self.exposure*source_info['luminosities']/(4.*np.pi*source_info['distances']**2.)

        # Poisson draw from mean photon counts to get realization of photon counts
        photon_counts = np.random.poisson(mean_photon_counts).astype('int')

        # Add single photon sources to the photon counts
        photon_counts = np.concatenate((photon_counts, np.ones(source_info['single_p_distances'].size).astype('int')))

        # Array of angles of photons
        angles = np.ones([np.sum(photon_counts), 2])
        angles[:,0] = np.repeat(np.concatenate((source_info['angles'][:,0], source_info['single_p_angles'][:,0])), photon_counts)
        angles[:,1] = np.repeat(np.concatenate((source_info['angles'][:,1], source_info['single_p_angles'][:,1])), photon_counts)
        
        # Array of photon energies
        energies = np.zeros(angles[:,0].size)

        # Array for combined single and multi photon source types
        source_types = np.concatenate((source_info['types'], source_info['single_p_types']))
        
        # Array to keep track of which source each photon came from
        photon_types = np.repeat(source_types, photon_counts)
         
        # Loop over all point sources
        self.N_source_classes = len(self.abun_lum_spec)
        for si in range(self.N_source_classes):
            if self.source_class_list[si] == 'isotropic_faint_multi_spectra' or self.source_class_list[si] == 'independent_spherical_multi_spectra' or self.source_class_list[si] == 'independent_cylindrical_multi_spectra':
                if np.count_nonzero(photon_counts) == 0:
                    energies = np.array([])
                    continue
                source_photon_counts = photon_counts[np.where(source_types == si)]
                if np.count_nonzero(source_photon_counts) == 0:
                    continue
                # Assign energies to all of those photons
                #get spectra for each physical source
                energy_vals = np.geomspace(self.Emin_gen, self.Emax_gen, grains)
                if self.source_class_list[si] == 'isotropic_faint_multi_spectra':
                    spectra = self.abun_lum_spec[si][1](energy_vals, num_spectra = np.count_nonzero(source_photon_counts), params = input_params)
                else:
                    spectra = self.abun_lum_spec[si][2](energy_vals, num_spectra = np.count_nonzero(source_photon_counts), params = input_params)
                #convert to spectra for each photon
                spectra = np.repeat(spectra, source_photon_counts[np.nonzero(source_photon_counts)], axis = 0)
                #normalize the spectra
                energy_m = np.tile(energy_vals, (np.sum(source_photon_counts), 1))
                norms = np.sum(spectra[:,:-1]*(energy_m[:,1:]-energy_m[:,:-1]), axis = 1)
                spectra = spectra/(np.tile(norms, (np.size(energy_vals), 1)).T)
                #get the cumulative distribution functions for each spectra
                CDFs = np.cumsum(spectra[:,:-1]*(energy_m[:,1:]-energy_m[:,:-1]), axis = 1)
                #draw photon energies
                rands = np.random.rand(np.sum(source_photon_counts))
                energies[np.where(photon_types == si)] = energy_vals[self.searchsorted2d(CDFs, rands)]

            if self.source_class_list[si] == 'isotropic_faint_single_spectrum' or self.source_class_list[si] == 'independent_spherical_single_spectrum' or self.source_class_list[si] == 'independent_cylindrical_single_spectrum':
                if np.count_nonzero(photon_counts) == 0:
                    energies = np.array([])
                    continue
                source_photon_counts = photon_counts[np.where(source_types == si)]
                if np.count_nonzero(source_photon_counts) == 0:
                    continue
                # Assign energies to all of those photons
                energy_vals = np.geomspace(self.Emin_gen, self.Emax_gen, grains)
                if self.source_class_list[si] == 'isotropic_faint_single_spectrum':
                    spectrum = self.abun_lum_spec[si][1](energy_vals, params = input_params)
                else:
                    spectrum = self.abun_lum_spec[si][2](energy_vals, params = input_params)
                norm = np.sum(spectrum[:-1]*(energy_vals[1:] - energy_vals[:-1]))
                Ei = self.draw_from_pdf(energy_vals[:-1], spectrum[:-1]*(energy_vals[1:] - energy_vals[:-1])/norm, np.sum(source_photon_counts))
                Es = energy_vals[Ei]
                energies[np.where(photon_types == si)] = Es
                
        # Loop over all isotropic diffuse and healpix map sources, appending angles and energies to the existing lists
        for si in range(self.N_source_classes):
            if self.source_class_list[si] == 'isotropic_diffuse':
                energy_vals = np.geomspace(self.Emin_gen, self.Emax_gen, grains)
                spectrum = self.abun_lum_spec[si][0](energy_vals, input_params)
                solid_angle = 2*np.pi*(1-np.cos(self.angular_cut_gen))
                exposure_correction = units.kpc.to('cm')**2
                num_photons = np.rint(solid_angle*np.sum(spectrum[1:]*(energy_vals[1:] - energy_vals[:-1]))*self.exposure*exposure_correction).astype('int')
                Es = energy_vals[self.draw_from_pdf(energy_vals, spectrum/np.sum(spectrum), num_photons)]
                As = self.draw_random_angles(num_photons)
                keep_i = np.where(np.abs(np.pi/2 - As[:,0]) >= self.lat_cut_gen)[0]
                energies = np.concatenate((energies, Es[keep_i]))
                angles = np.concatenate((angles, As[keep_i]))
                
            if self.source_class_list[si] == 'healpix_map':
                map_vals, map_E, map_i, N_side = self.abun_lum_spec[si][0](input_params)
                As, Es = self.draw_angles_and_energies_from_partial_map(map_vals, map_E, map_i, N_side)
                angles = np.concatenate((angles, As))
                energies = np.concatenate((energies, Es))

        '''
        photon_angles = np.concatenate((angles, source_info['single_p_angles']))
        '''
        '''
        single_p_energies = np.array([])
        '''
        '''
        for si in range(self.N_source_classes):
            type_indices_bool = np.where(source_info['types'] == si, True, False)
            num_single_p_sources = np.size(np.where(source_info['single_p_types'] == si, True, False))
            energy_array = np.linspace(self.Emin_gen, self.Emax_gen, grains)
            dndE = self.abun_lum_spec[si][1](input_params, energy_array)
            if type(dndE) == type(torch.tensor([])):
                dndE = dndE.numpy()
            energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                dndE/np.sum(dndE),
                                                np.sum(photon_counts, where = type_indices_bool))
            single_p_energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                dndE/np.sum(dndE),
                                                num_single_p_sources)
            energies = np.concatenate((energies, energy_array[energy_indices]))
            single_p_energies = np.concatenate((single_p_energies, energy_array[single_p_energy_indices]))
        energies = np.concatenate((energies, single_p_energies))
        for si in range(self.N_source_classes):
            if not self.is_background_list[si]:
                continue
            if self.is_isotropic_list[si]:
                num_E = int(round(self.Emax_gen - self.Emin_gen))
                energy_array = np.linspace(self.Emin_gen, self.Emax_gen, num_E + 1)
                dndE = self.abun_lum_spec[si][0](input_params, energy_array)
                if type(dndE) == type(torch.tensor([])):
                    dndE = dndE.numpy()
                solid_angle = 2*np.pi*2*(1-np.cos(np.radians(90. - self.lat_cut_gen)))
                N_iso_photons = np.random.poisson(np.round(np.sum(dndE*self.exposure*(units.kpc.to('cm')**2)*solid_angle)).astype('int'))
                As = np.ones([N_iso_photons, 2])
                As[:,0] = np.arccos(1 - np.random.rand(N_iso_photons))
                As[:,0] += (np.pi - 2*As[:,0])*np.random.randint(2, size = N_iso_photons)
                As[:,1] = np.random.uniform(low = 0., high = 2*np.pi, size = N_iso_photons)
                if type(dndE) == type(torch.tensor([])):
                    dndE = dndE.numpy()
                energy_indices = self.draw_from_pdf(np.arange(0,len(energy_array)),
                                                    dndE/np.sum(dndE), N_iso_photons)
                photon_angles = np.concatenate((photon_angles, As))
                energies = np.concatenate((energies, energy_array[energy_indices]))
        '''
        
        photon_info = {'angles':angles, 'energies':energies}

        if (self.verbose):
            print(photon_info)
        
        return photon_info

    #for isotropic adundances where luminosity may depend on radius
    #epsilon is the propability of recieveing a single photon below which single-photon sources are generated
    def draw_luminosities_and_radii(self, input_params, RL, grains = 1000, epsilon = 0):
        r = np.exp(np.linspace(np.log(0.001), np.log(self.Rmax), grains))
        lums = np.exp(np.linspace(np.log(self.Lmin), np.log(self.Lmax), grains))
        RL_PDF = RL(np.tile(r[:-1],(grains-1,1)).T, np.tile(lums[:-1],(grains-1,1)), input_params)
        dRdL = np.tile(4*np.pi * r[:-1]**2 * (r[1:]-r[:-1]), (grains-1,1)).T * np.tile((lums[1:]-lums[:-1]), (grains-1,1))
        RL_integral = RL_PDF * dRdL
        
        # binomially draw low luminosity sources to save computation time
        Dconserv = np.abs(self.GC_to_earth - np.tile(r[:-1],(grains-1,1)).T) #the closest possible distance from earth to a source generated at raduis r
        Dconserv = np.where(Dconserv == 0, 0.00000000001, Dconserv)
        C = (np.tile(lums[:-1],(grains-1,1)))*self.exposure/(4*np.pi*Dconserv**2) #upper bound on expected number of photons from such a source
        Ci = np.where(C < epsilon)
        C = np.where(C < epsilon, C, 0)
        p = C*np.exp(-C) #probability that exactly 1 photon is recieved from such a source
        num_single_p_sources_at_radii = np.random.poisson(np.sum(RL_integral*p, axis = 1))
        single_p_radii = np.repeat(r[:-1], num_single_p_sources_at_radii)
        
        # draw remaining sources from distribution
        RL_integral[Ci] = 0
        N_draws = np.round(np.sum(RL_integral)).astype(int)
        r_indices, lum_indices = self.draw_from_2D_pdf(RL_integral, N_draws)
        
        return r[r_indices], lums[lum_indices], single_p_radii
    
    #for independently distributed R*Theta*Phi abundances
    def draw_spherical_positions_independent(self, input_params, R, Theta, Phi, grains = 1000):
        r = np.exp(np.linspace(np.log(0.001), np.log(self.Rmax), grains))
        theta = np.linspace(0, np.pi, grains)
        phi = np.linspace(0, 2*np.pi, grains)
        r_integral = R(r[:-1], input_params) * r[:-1]**2 * (r[1:]-r[:-1])
        theta_integral = Theta(theta[:-1], input_params) * np.sin((theta[:-1])) * (theta[1:]-theta[:-1])
        phi_integral = Phi(phi[:-1], input_params) * (phi[1:]-phi[:-1])
        N_draws = np.round(np.sum(r_integral)*np.sum(theta_integral)*np.sum(phi_integral)).astype('int')
        r_i = self.draw_from_pdf(r, r_integral/np.sum(r_integral), N_draws)
        theta_i = self.draw_from_pdf(theta, theta_integral/np.sum(theta_integral), N_draws)
        phi_i = self.draw_from_pdf(phi, phi_integral/np.sum(phi_integral), N_draws)
        
        return r[r_i], theta[theta_i], phi[phi_i]
    
    #for independently distributed R*Z*Phi abundances
    def draw_cylindrical_positions_independent(self, input_params, R, Z, Phi, grains = 1000):
        r = np.exp(np.linspace(np.log(0.001), np.log(self.Rmax), grains))
        z_max = self.Rmax
        z = np.linspace(-z_max, z_max, grains)
        phi = np.linspace(0, 2*np.pi, grains)
        r_integral = R(r[:-1], input_params) * r[:-1] * (r[1:]-r[:-1])
        z_integral = Z(z[:-1], input_params) * (z[1:]-z[:-1])
        phi_integral = Phi(phi[:-1], input_params) * (phi[1:]-phi[:-1])
        N_draws = np.round(np.sum(r_integral)*np.sum(z_integral)*np.sum(phi_integral)).astype('int')
        r_i = self.draw_from_pdf(r, r_integral/np.sum(r_integral), N_draws)
        z_i = self.draw_from_pdf(z, z_integral/np.sum(z_integral), N_draws)
        phi_i = self.draw_from_pdf(phi, phi_integral/np.sum(phi_integral), N_draws)
        
        return r[r_i], z[z_i], phi[phi_i]
    
    #for full non-isotropic healpix maps
    def draw_angles_and_energies_from_full_map(self, map_vals, map_E, N_draws = 0):
        N_pix = map_all.shape[1]
        N_side = hp.npix2nside(N_pix)
        
        #For angular cut
        keep_i = np.where(hp.rotator.angdist(hp.pix2ang(N_side, np.linspace(0, N_pix-1, N_pix).astype('int')), [np.pi/2, 0]) < self.angular_cut_gen)[0]
        new_map_all = map_all[:, keep_i]

        #For lat cut
        masked_i = np.where(np.abs(hp.pix2ang(N_side, np.linspace(0, N_pix-1, N_pix).astype('int'))[0] - np.pi/2) < np.radians(self.lat_cut_gen))[0]
        new_map_all[:, masked_i] *= 0

        dE = map_E[1:] - map_E[:-1]
        integrand = new_map_all[:-1,:]*self.exposure*(units.kpc.to('cm')**2)*(4*np.pi/N_pix)*(np.tile(dE, (keep_i.size,1)).T)
        if N_draws == 0:
            N_draws = int(round(np.random.poisson(np.sum(integrand))))
        energy_i, pixel_i = self.draw_from_2D_pdf(integrand, N_draws)
        angles = hp.pix2ang(N_side, keep_i[pixel_i])
        return np.array(angles).T, map_E[energy_i]
    
    #for partial non-isotropic healpix maps
    def draw_angles_and_energies_from_partial_map(self, map_vals, map_E, map_i, N_side, N_draws = 0):
        full_map_N_pix = hp.nside2npix(N_side)
        N_pix = map_i.size
        
        #For lat cut
        masked_i = hp.query_strip(nside = N_side, theta1 = np.pi/2-self.lat_cut_gen, theta2 = np.pi/2+self.lat_cut_gen)
        map_vals[:, np.where(np.isin(map_i, masked_i))[0]] *= 0
        
        dE = map_E[1:] - map_E[:-1]
        integrand = map_vals[:-1,:]*self.exposure*(units.kpc.to('cm')**2)*(4*np.pi/full_map_N_pix)*(np.tile(dE, (N_pix,1)).T)
        if N_draws == 0:
            N_draws = int(round(np.random.poisson(np.sum(integrand))))
        energy_i, pixel_i = self.draw_from_2D_pdf(integrand, N_draws)
        angles = hp.pix2ang(N_side, map_i[pixel_i])
        return np.array(angles).T, map_E[energy_i]

    def draw_random_angles(self, num_angles):
        #Randomly draws angles within self.angular_cut_gen region. Note: angles inside self.lat_cut_gen are still returned
        angles = np.zeros((2, num_angles))
        angles[0,:] = np.arccos(1 - (1-np.cos(self.angular_cut_gen))*np.random.rand(num_angles))
        angles[1,:] = 2*np.pi*np.random.rand(num_angles)
        rotmat = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        return (hp.rotator.rotateDirection(rotmat, angles)).T
        
    def draw_from_isotropic_background_unbinned(self, Ebins, exposure, Sangle):
        e, dnde = self.e_isotropic, self.dnde_isotropic
        f = scipy.interpolate.interp1d(e, dnde, kind='linear', fill_value=0.)
        #lowE = np.exp(np.log(e[0]) - (np.log(e[1]) - np.log(e[0]))/2)
        #highE = e[-1] + (e[-1] - e[-2])/2
        lowE = Ebins[0:-1]
        highE = Ebins[1:]
        e = np.geomspace(lowE, highE, 1000)
        dnde = f(e)
        int_terms = dnde[1:]*(e[1:]-e[:-1])
        num_photons = int(round(exposure*Sangle*np.sum(int_terms)))
        e_indices = self.draw_from_pdf(np.arange(0,len(e)-1), int_terms/np.sum(int_terms), num_photons)
        energies = e[e_indices]
        return energies

    '''
    #for general non-isotropic abundances (requires large RAM)
    def draw_masses_radii_angles(self, input_params, abundance, luminosity, N_draws = 0, grains = 100):
        #intensityLim = 0.1#expected photons/bin <--not sure if this is justified, so turned off
        block = int(grains/10)
        massVals = np.geomspace(self.Mmin, self.Mmax, grains+1)
        radiusVals = np.geomspace(self.Rmin, self.Rmax, grains+1)
        thetaVals = np.linspace(0,np.pi,grains+1)
        phiVals = np.linspace(0,2*np.pi,grains+1)
        massPDF, radiusPDF, thetaPDF, phiPDF = np.zeros(grains), np.zeros(grains), np.zeros(grains), np.zeros(grains)
        for i in np.linspace(0, grains-block, int(grains/block)).astype(int):
            for j in np.linspace(0, grains-block, int(grains/block)).astype(int):
                for k in np.linspace(0, grains-block, int(grains/block)).astype(int):
                    massArray = np.tile(massVals[1:],(block,block,block,1)).T
                    radiusArray = np.tile(np.tile(radiusVals[1+i:1+i+block],(grains,1)).T,(block,block,1,1)).T
                    thetaArray = np.tile(np.tile(thetaVals[1+j:1+j+block],(block,1)).T,(grains,block,1,1))
                    phiArray = np.tile(phiVals[1+k:1+k+block],(grains,block,block,1))
                    PDF = abundance(input_params, massArray, radiusArray, thetaArray, phiArray)
                    del phiArray
                    luminosityArray, sigArray = luminosity(input_params, massArray, radiusArray)
                    del massArray, sigArray
                    dM = np.tile(massVals[1:]-massVals[:-1],(block,block,block,1)).T
                    dR = np.tile(np.tile(radiusVals[1+i:1+i+block]-radiusVals[i:i+block],(grains,1)).T,(block,block,1,1)).T
                    dT = thetaVals[1]-thetaVals[0]
                    dP = phiVals[1]-phiVals[0]
                    integrand = PDF*radiusArray**2*np.sin(thetaArray)*dM*dR*dT*dP
                    del PDF, dM, dR
                    distanceArray = np.sqrt((self.GC_to_earth + radiusArray*np.cos(thetaArray))**2 + (radiusArray*np.sin(thetaArray))**2)
                    del radiusArray, thetaArray
                    #intensityArray = integrand*luminosityArray*self.exposure/(4*np.pi*distanceArray**2) <--not sure if this is justified, so turned off
                    integrand = np.where(intensityArray < intensityLim, 0, integrand)
                    massPDF += np.sum(integrand, axis = (1,2,3))
                    radiusPDF[i:i+block] += np.sum(integrand, axis = (0,2,3))
                    thetaPDF[j:j+block] += np.sum(integrand, axis = (0,1,3))
                    phiPDF[k:k+block] += np.sum(integrand, axis = (0,1,2))
        print('pdfs')
        if N_draws == 0:
            N_draws = int(round(np.sum(massPDF)))
        print(N_draws)
        masses = massVals[self.draw_from_pdf(massVals[1:], massPDF/np.sum(massPDF), N_draws)]
        radii = massVals[self.draw_from_pdf(massVals[1:], massPDF/np.sum(massPDF), N_draws)]
        angles = np.ones((N_draws, 2))
        angles[:,0] = thetaVals[self.draw_from_pdf(thetaVals[1:], thetaPDF/np.sum(thetaPDF), N_draws)]
        angles[:,1] = phiVals[self.draw_from_pdf(phiVals[1:], phiPDF/np.sum(phiPDF), N_draws)]
        return masses, radii, angles
    
    #returns N_draws angles with granularity set by grains. Takes angular function density(theta, phi)
    def draw_angles_from_density(self, density, N_draws, grains = 10000):
        thetaVals = np.linspace(0,np.pi,grains)
        phiVals = np.linspace(0,2*np.pi,grains)
        PDF = density(np.tile(thetaVals[1:],(grains-1,1)).T, np.tile(phiVals[1:],(grains-1,1)))
        integrand = PDF*(np.tile(np.sin(thetaVals[1:]),(grains-1,1)).T)
        thetaPDF = np.sum(integrand, axis = 1)
        phiPDF = np.sum(integrand, axis = 0)
        angles = np.ones((N_draws, 2))
        angles[:,0] = thetaVals[self.draw_from_pdf(thetaVals[1:], thetaPDF/np.sum(thetaPDF), N_draws)]
        angles[:,1] = phiVals[self.draw_from_pdf(phiVals[1:], phiPDF/np.sum(phiPDF), N_draws)]

        return angles
    '''

    ##########################################################################
    '''
    Summary statistic functions

    '''
    ##########################################################################

    def get_summary(self, photon_info, summary_properties = {'summary_type':'energy_dependent_histogram',
                                                             'map_type':'healpix',
                                                             'mask_galactic_plane': None,
                                                             'N_pix':12*64**2,
                                                             'mask_galactic_center_latitude':None, #in radians
                                                             'N_energy_bins':10,
                                                              'histogram_properties':{'Nbins':10, 'Cmax_hist': 10, 'Cmin_hist': 0, 'energy_bins_to_use':'all'}
                                                            }):

        if (summary_properties['summary_type'] == 'energy_dependent_histogram'):
            summary = self.get_energy_dependent_histogram(photon_info, summary_properties)
        if (summary_properties['summary_type'] == 'energy_dependent_map'):
            summary = self.get_energy_dependent_map(photon_info, summary_properties)
        
        return summary
    
    def get_energy_dependent_histogram(self, photon_info, summary_properties):
        # Calculate the energy-dependent histogram given
        
        if 'valid' in photon_info:
            if not photon_info['valid']:
                return np.zeros((1, summary_properties['N_energy_bins'] * summary_properties['histogram_properties']['Nbins'])) * np.nan

        emap = self.get_energy_dependent_map(photon_info, summary_properties)
        energy_dependent_histogram = self.get_energy_dependent_histogram_from_map(emap, summary_properties)
        
        return energy_dependent_histogram
    
    def get_energy_dependent_map(self, photon_info, summary_properties):
        '''
        Given unbinned photon data, return maps with dimension npix x N_energy

        map_type can be healpix or internal
        '''
        
        #The output map
        N_pix = summary_properties['N_pix']
        N_energy_bins = summary_properties['N_energy_bins']
        # TO DO: implement batch generation of photon info
        N_batch = 1 
        
        #For each batch, construct an object which is a map for each energy bin
        energy_dependent_map = np.zeros((N_batch, N_pix, N_energy_bins))

        map_type = summary_properties['map_type']
        if (map_type == 'healpix'):
            NSIDE = np.sqrt(N_pix/12).astype('int')
            pixels = hp.ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])
        elif (map_type == 'internal'):
            NSIDE = np.sqrt(N_pix/12).astype('int')            
            pixels = self.internal_ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])

        #bin data by pixel
        Emin, Emax = summary_properties['Emin'], summary_properties['Emax']
        N_energy_bins = summary_properties['N_energy_bins']
        if summary_properties['log_energy_bins'] is True:
            E_bins = np.logspace(np.log10(Emin), np.log10(Emax), num=N_energy_bins+1)
        else:
            E_bins = N_energy_bins
        
        #All photon energies
        photon_energies = photon_info['energies']

        # To do: implement batched photon info
        #print('constructing map')
        for batchi in range(N_batch):
            if len(photon_energies) == 0:
                continue
            #Histogram works inclusively on the lower edge, so this should work
            hist, pix_edges, E_edges = np.histogram2d(pixels, photon_energies, range = ((0,N_pix),(Emin, Emax)), bins = [N_pix,np.logspace(np.log10(Emin), np.log10(Emax),num=N_energy_bins+1)])

            if summary_properties['galactic_plane_latitude_cut'] is not None:
                gal_lat = summary_properties['galactic_plane_latitude_cut']
                colat, _ = hp.pix2ang(NSIDE, np.arange(0, N_pix))
                pixels_in_plane = (colat > np.pi / 2 - gal_lat) & (colat < np.pi / 2 + gal_lat)
                hist[pixels_in_plane] = hp.UNSEEN

            energy_dependent_map[batchi,:,:] = hist
        
        return energy_dependent_map 

    def get_energy_dependent_histogram_from_map(self, input_map, summary_properties):
        '''
        Takes in binned data (i.e a map with dimension N_pix x N_energy) and return a summary statistic
        '''
        #Properties of map
        #N_E should match what's in summary_properties
        N_batch, N_pix, N_E = input_map.shape

        #properties of histogram
        N_bins = summary_properties['histogram_properties']['Nbins']
        Cmin_hist, Cmax_hist = summary_properties['histogram_properties']['Cmin_hist'], summary_properties['histogram_properties']['Cmax_hist']
        #Cmax_hist can be array or scalar

        output_summary = np.zeros((N_batch, N_bins*N_E))
        for bi in range(0,N_batch):
            summary_bi = np.zeros((N_bins, N_E))
            for ei in range(0,N_E):

                if (np.isscalar(Cmax_hist)):
                    max_counts_value = Cmax_hist
                else:
                    max_counts_value = Cmax_hist[ei]

                #Compute histogram of batch bi and energy bin ei
                hist, bin_edges = np.histogram(input_map[bi,:,ei], bins = N_bins, range = (Cmin_hist, max_counts_value))

                #store the histogram
                summary_bi[:,ei] = hist

            #Output summary is flattened version of energy-dependent histogram
            output_summary[bi,:] = summary_bi.transpose().flatten()

        return output_summary
    
    def get_partial_map_summary(self, photon_info, N_side, N_Ebins, Ebinspace = 'linear'):
        #returns a 2d array of (pix X energy) for a limited region of sky within an self.angular_cut_mask
        N_pix = 12*N_side**2
        pix_i = np.linspace(0, N_pix-1, N_pix, dtype = 'int')
        center = hp.ang2vec(np.pi/2, 0)
        close_pix_i = hp.query_disc(N_side, center, self.angular_cut_mask)
        if Ebinspace == 'linear':
            Ebins = np.linspace(self.Emin_mask, self.Emax_mask, N_Ebins + 1)
        elif Ebinspace == 'log':
            Ebins = np.geomspace(self.Emin_mask + 0.1, self.Emax_mask + 0.1, N_Ebins + 1) - 0.1
        partial_map = np.histogram2d(hp.ang2pix(N_side, photon_info['angles'][:,0], photon_info['angles'][:,1]), photon_info['energies'], bins = [np.size(close_pix_i), Ebins])
        
        return partial_map
    
    def get_roi_map_summary(self, photon_info, N_side, N_Ebins, Ebinspace = 'linear'):
        #returns a 2d array of (pix X energy) for a limited region of sky within an angular cut
        N_pix = 12*N_side**2
        pix_bins = np.linspace(0, N_pix, N_pix + 1, dtype = 'int')
        roi_pix_i = self.get_roi_pix_indices(N_side)
        if Ebinspace == 'linear':
            Ebins = np.linspace(self.Emin_mask, self.Emax_mask, N_Ebins + 1)
        elif Ebinspace == 'log':
            Ebins = np.geomspace(self.Emin_mask + 0.1, self.Emax_mask + 0.1, N_Ebins + 1) - 0.1
        partial_map = np.histogram2d(hp.ang2pix(N_side, photon_info['angles'][:,0], photon_info['angles'][:,1]), photon_info['energies'], bins = [pix_bins, Ebins])
        
        return partial_map[0][roi_pix_i, :]
    
    def get_counts_histogram_from_roi_map(self, roi_map, mincount, maxcount, N_countbins, countbinspace = 'linear'):
        if countbinspace == 'linear':
            countbins = np.linspace(mincount, maxcount, N_countbins + 1)
        elif countbinspace == 'log':
            countbins = np.geomspace(mincount + 0.1, maxcount + 0.1, N_countbins + 1) - 0.1
        hist = np.zeros((N_countbins, roi_map.shape[1]))
        for Ei in np.arange(roi_map.shape[1]):
            hist[:,Ei] = np.histogram(roi_map[:,Ei], bins = countbins)[0]
            
        return hist
    
    def get_roi_pix_indices(self, N_side):
        N_pix = 12*N_side**2
        pix_i = np.linspace(0, N_pix-1, N_pix, dtype = 'int')
        roi_pix_i = np.where(np.logical_and(hp.rotator.angdist(np.array([np.pi/2, 0]), hp.pix2ang(N_side, pix_i)) <= self.angular_cut_mask, np.abs(np.pi/2 - hp.pix2ang(N_side, pix_i)[0]) >= self.lat_cut_mask))[0]
        
        return roi_pix_i
    
    def get_map_from_unbinned(self, photon_info, N_pix, N_energy, map_type = 'healpix'):
        '''
        Given unbinned photon data, return maps with dimension npix x N_energy

        map_type can be healpix or internal
        '''
        
        #The output map
        output_summary = np.zeros((N_pix, N_energy))

        if (map_type == 'healpix'):
            NSIDE = np.sqrt(N_pix/12).astype('int')
            pixels = hp.ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])
        elif (map_type == 'internal'):
            NSIDE = np.sqrt(N_pix/12).astype('int')            
            pixels = self.internal_ang2pix(NSIDE, photon_info['angles'][:,0], photon_info['angles'][:,1])

        #bin data by pixel
        for pix_index in range(N_pix):
            pix_energies = photon_info['energies'][np.where(pixels == pix_index)]
            hist, hist_edges = np.histogram(pix_energies, bins = N_energy, range = (self.Emin_mask, self.Emax_mask))
            output_summary[pix_index,:] = hist
        
        return  output_summary

    '''
    #used in following get_map_from_unbinned function, works identically to healpix version but with different
    #spherical partition
    def internal_ang2pix(self, NSIDE, data_thetas, data_phis):
        thetas = np.arccos(1 - np.linspace(0, 2*NSIDE, 2*NSIDE + 1)/NSIDE)
        phis = np.linspace(0, 2*np.pi, NSIDE + 1)
        data_theta_index = np.searchsorted(thetas, data_thetas)-1
        data_phi_index = np.searchsorted(phis, data_phis)-1
        data_index = NSIDE*data_theta_index + data_phi_index

        return data_index
    '''
    ##########################################################################
    '''
    Observational functions
    '''
    ##########################################################################
    

    def apply_PSF(self, photon_info, obs_info, single_energy_psf = False, single_energy_value = None):
        '''
        Applies energy dependent Fermi PSF assuming normal incidence
        If input energy is outside (10^0.75, 10^6.5) MeV, the PSF of the nearest energy bin is applied
        Only valid for Fermi pass 8
        '''
        psf_fits_path, event_type = obs_info['psf_fits_path'], obs_info['event_type']
        if not psf_fits_path.endswith(event_type[:-1] + '.fits'):
            print('!!!!WARNING!!!!\n event_type not found in given psf_fits file\n PSF not applied\n!!!!WARNING!!!!')
            return photon_info
        
        num_photons = np.size(photon_info['energies'])
        mean_energy = np.mean(photon_info['energies'])
        photon_energies = np.copy(photon_info['energies'])
        if single_energy_psf:
            if single_energy_value != None:
                photon_energies[:] = single_energy_value
            else:
                photon_energies[:] = mean_energy
        
        hdul = fits.open(psf_fits_path)
        scale_hdu = 'PSF_SCALING_PARAMS_' + event_type
        fit_hdu = 'RPSF_' + event_type
        C = hdul[scale_hdu].data[0][0][:-1]
        beta = -hdul[scale_hdu].data[0][0][2]
        fit_ebins = np.linspace(0.75, 6.5, 24)
        distances = np.zeros(num_photons)
        
        #loop over energy bins in which params are defined
        for index in range(23):
            if index == 0:
                ebin_i = np.where(np.log10(photon_energies)<fit_ebins[index+1])
            elif index == 22:
                ebin_i = np.where(np.log10(photon_energies)>=fit_ebins[index])
            else:
                ebin_i = np.where(np.logical_and(np.log10(photon_energies)>=fit_ebins[index], np.log10(photon_energies)<fit_ebins[index+1]))
            NTAIL = hdul[fit_hdu].data[0][5][7][index]
            SCORE = hdul[fit_hdu].data[0][6][7][index]
            STAIL = hdul[fit_hdu].data[0][7][7][index]
            GCORE = hdul[fit_hdu].data[0][8][7][index]
            GTAIL = hdul[fit_hdu].data[0][9][7][index]
            FCORE = 1/(1 + NTAIL*STAIL**2/SCORE**2)
            x_vals = 10**np.linspace(-1, 1.5, 1000) #!!!Change lower bound to -3 before software release!!!
            kingCORE = (1/(2*np.pi*SCORE**2))*(1-(1/GCORE))*(1+(1/(2*GCORE))*(x_vals**2/SCORE**2))**(-GCORE)
            kingTAIL = (1/(2*np.pi*STAIL**2))*(1-(1/GTAIL))*(1+(1/(2*GTAIL))*(x_vals**2/STAIL**2))**(-GTAIL)
            PSF = FCORE*kingCORE + (1-FCORE)*kingTAIL
            PDFx = 2*np.pi*x_vals[:-1]*PSF[:-1]*(x_vals[1:]-x_vals[:-1])
            x = x_vals[self.draw_from_pdf(x_vals[:-1], PDFx/np.sum(PDFx), np.size(ebin_i))]
            S_P = np.sqrt((C[0]*(photon_energies[ebin_i]/100)**(-beta))**2 + C[1]**2)
            distances[ebin_i] = 2*np.sin(x*S_P/2)
        hdul.close()
        rotations = 2*np.pi*np.random.random(num_photons)
        #create orthonormal basis for each photon direction
        parallel = hp.ang2vec(photon_info['angles'][:,0], photon_info['angles'][:,1])
        perp1angles = photon_info['angles']
        perp1angles[:,0] += np.pi/2 
        over_indices = np.where(perp1angles[:,0]>np.pi)
        perp1angles[over_indices,0] = np.pi - perp1angles[over_indices,0]%np.pi
        perp1angles[over_indices,1] += np.pi
        perp1angles[over_indices,1] %= 2*np.pi
        perp1 = hp.ang2vec(perp1angles[:,0], perp1angles[:,1])
        perp2angles = photon_info['angles']
        perp2angles[:,0] = np.pi/2*np.ones(np.size(perp2angles[:,0]))
        perp2angles[:,1] += np.pi/2
        perp2angles[:,1] %= 2*np.pi
        perp2 = hp.ang2vec(perp2angles[:,0], perp2angles[:,1])
        #construct new direction from orthonormal basis
        new_parallel = np.tile(np.cos(distances), (3,1)).T*parallel
        new_perp = np.tile(np.sin(distances), (3,1)).T*(np.tile(np.cos(rotations), (3,1)).T*perp1 + np.tile(np.sin(rotations), (3,1)).T*perp2)
        obs_photon_info = copy.deepcopy(photon_info)
        obs_photon_info['angles'] = np.array(hp.vec2ang(new_parallel+new_perp)).T
        '''
        delta_thetas = distances*np.cos(rotations)
        delta_phis = distances*np.sin(rotations)
        photon_info['angles'][:,0] += delta_thetas
        over_indices = np.where(np.logical_or(photon_info['angles'][:,0] > np.pi, photon_info['angles'][:,0] < 0))
        photon_info['angles'][over_indices,0] = np.pi - photon_info['angles'][over_indices,0]%np.pi
        photon_info['angles'][over_indices,1] += np.pi
        photon_info['angles'][:,1] += delta_phis
        photon_info['angles'][:,1] %= 2*np.pi
        '''
         
        return obs_photon_info
    
    def apply_energy_dispersion(self, photon_info, obs_info, single_energy_ed = False, single_energy_value = None):
        '''
        Applies Fermi energy dispersion assuming normal incidence
        If input energy is outside (10^0.75, 10^6.5) MeV, the energy dispersion of the nearest energy bin is applied
        Only valid for Fermi pass 8
        '''
        edisp_fits_path, event_type = obs_info['edisp_fits_path'], obs_info['event_type']
        if not edisp_fits_path.endswith(event_type[:-1] + '.fits'):
            print('!!!!WARNING!!!!\n event_type not found in given edisp_fits file\n Energy Dispersion not applied\n!!!!WARNING!!!!')
            return photon_info
        
        num_photons = np.size(photon_info['energies'])
        mean_energy = np.mean(photon_info['energies'])
        photon_energies = np.copy(photon_info['energies'])
        if single_energy_ed:
            if single_energy_value != None:
                photon_energies[:] = single_energy_value
            else:
                photon_energies[:] = mean_energy
        
        hdul = fits.open(edisp_fits_path)
        scale_hdu = 'EDISP_SCALING_PARAMS_' + event_type
        fit_hdu = 'ENERGY DISPERSION_' + event_type
        C = hdul[scale_hdu].data[0][0]
        fit_ebins = np.linspace(0.75, 6.5, 24)
        differences = np.zeros(num_photons)
        #loop over energy bins in which params are defined
        for index in range(23):
            if index == 0:
                ebin_i = np.where(np.log10(photon_energies)<fit_ebins[index+1])
            if index == 22:
                ebin_i = np.where(np.log10(photon_energies)>=fit_ebins[index])
            else:
                ebin_i = np.where(np.logical_and(np.log10(photon_energies)>=fit_ebins[index], np.log10(photon_energies)<fit_ebins[index+1]))
            F = hdul[fit_hdu].data[0][4][7][index]
            S1 = hdul[fit_hdu].data[0][5][7][index]
            K1 = hdul[fit_hdu].data[0][6][7][index]
            BIAS1 = hdul[fit_hdu].data[0][7][7][index]
            BIAS2 = hdul[fit_hdu].data[0][8][7][index]
            S2 = hdul[fit_hdu].data[0][9][7][index]
            K2 = hdul[fit_hdu].data[0][10][7][index]
            PINDEX1 = hdul[fit_hdu].data[0][11][7][index]
            PINDEX2 = hdul[fit_hdu].data[0][12][7][index]
            x_vals = np.linspace(-15, 15, 1000)
            x_low1, x_high1 = np.where(x_vals < BIAS1), np.where(x_vals >= BIAS1)
            x_low2, x_high2 = np.where(x_vals < BIAS2), np.where(x_vals >= BIAS2)
            g1, g2 = np.ones(1000), np.ones(1000)
            prefac1 = PINDEX1/(S1*sp.special.gamma(1/PINDEX1))*K1/(1+K1**2)
            prefac2 = PINDEX2/(S2*sp.special.gamma(1/PINDEX2))*K2/(1+K2**2)
            g1[x_low1] = prefac1*np.exp(-(1/(K1*S1)*np.abs(x_vals[x_low1]-BIAS1))**PINDEX1)
            g2[x_low2] = prefac2*np.exp(-(1/(K2*S2)*np.abs(x_vals[x_low2]-BIAS2))**PINDEX2)
            g1[x_high1] = prefac1*np.exp(-(K1/S1*np.abs(x_vals[x_high1]-BIAS1))**PINDEX1)
            g2[x_high2] = prefac2*np.exp(-(K2/S2*np.abs(x_vals[x_high2]-BIAS2))**PINDEX2)
            D = F*g1 + (1-F)*g2
            x = x_vals[self.draw_from_pdf(x_vals[:-1], D/np.sum(D), np.size(ebin_i))]
            E = photon_energies[ebin_i]
            theta = 0
            S_D = C[0]*np.log10(E)**2 + C[1]*np.cos(theta)**2 + C[2]*np.log10(E) + C[3]*np.cos(theta) + C[4]*np.log10(E)*np.cos(theta) + C[5]
            differences[ebin_i] = x*E*S_D
        hdul.close()
        
        obs_photon_info = copy.deepcopy(photon_info)
        obs_photon_info['energies'] += differences
         
        return obs_photon_info
    
    def apply_exposure(self, photon_info, obs_info):
        """Modify the generate photons to simulate a direction-dependent exposure.

        Photons are removed with probability 1 - exposure_map(theta, phi) / self.exposure. This assumes that photons have been generated with a max exposure of self.exposure and then are removed to simulate a directional dependence in the exposure map.

        The attribute self.exposure is a constant value. Photons are generated using this constant value.
        Optionally, an exposure_map can be provided that describes the dependence of exposure on the direction on the sky.
        
        'exposure_map' is a function that of the angles on the sky f(theta, phi) [in the same coordinate system as photon_info['angles'].
        If 'exposure_map' is a string, the function will try and load a healpix map and use it accordingly.

        :param photon_info: dictionary of photon_angles
        :param exposure_map: function of sky angles (theta, phi) describing the direction-dependent exposure. Or a file path name to a healpix map of the expsoure
        :returns: modified photon_dict
        """
        exposure_map = obs_info['exposure_map']
        
        # If there is no exposure map, we assume the exposure is constant and do not modify the photon list
        if exposure_map is None:
            return photon_info
        return photon_info

        # If exposure_map is a string, try to load a healpy map
        if isinstance(exposure_map, str): 
            try:
                hpx_map = np.load(exposure_map)
            except FileNotFoundError:
                raise FileNotFoundError('Exposure map must be either the relative file path name of a healpy exposure map or a function of sky angles f(theta,phi)')
            # Create the exposure_map funciton from the healpy exposure map
            nside = hp.npix2nside(len(hpx_map))
            def exposure_map(theta, phi): return hpx_map[hp.ang2pix(nside, theta, phi)]

        # Get the exposure for each photon
        exposures = exposure_map(*photon_info['angles'].T)

        # Double check that the exposure used to generate photons is consistent with this exposure map
        # If they are not consistent, the exposure_map is rescaled
        if not np.isclose(self.exposure, exposures.max()):
            exposures *= self.exposure / exposures.max()
            if self.verbose:
                print(f'\t-->Provided exposure and exposure_map are inconsistent.')
                print(f'\t-->Exposure map will be rescaled by a factor of {self.exposure / exposures.max()} to rectify')

        # Calculate the probability of rejecting the photon
        probabilities = 1 - exposures / self.exposure

        # Determine indices of removed photons randomly
        remove_photon_indices = (probabilities > np.random.random_sample(len(probabilities)))

        # take these photons out of the photon dict
        for key, values in photon_info.items():
            photon_info[key] = values[remove_photon_indices]

        if self.debug:
            print(f'\t-->Using direction-dependent exposure, {remove_photon_indices.sum()} photons removed')
            print(f'\t-->There are {~remove_photon_indices.sum()} photons remaining')

        return photon_info
    
    def apply_mask(self, photon_info, obs_info):
        '''
        Removes photons outside of self.angular_cut_mask, inside self.lat_cut_mask, and outside (self.Emin_mask, self.Emax_mask)
        Arbitrary mask from obs_info NOT YET IMPLEMENTED
        '''
        keep_i = np.where(np.logical_and(hp.rotator.angdist(np.array([np.pi/2, 0]), photon_info['angles'].T) <= self.angular_cut_mask, np.abs(np.pi/2 - photon_info['angles'][:,0]) >= self.lat_cut_mask))[0]
        keep_i = keep_i[np.where(np.logical_and(photon_info['energies'][keep_i] >= self.Emin_mask, photon_info['energies'][keep_i] <= self.Emax_mask))]
        
        obs_photon_info = copy.deepcopy(photon_info)
        obs_photon_info['angles'] = photon_info['angles'][keep_i,:]
        obs_photon_info['energies'] = photon_info['energies'][keep_i]
        
        return obs_photon_info
    
    def mock_observe(self, photon_info, obs_info):
        #photon_info contains all information about individual photons
        #obs_info is a dictionary containing info about the observation process
        
        if np.any(np.isnan(photon_info['energies'])):
            print('!!!!WARNING!!!!\n photon energies contain NaNs\n exposure map, psf, energy dispersion, and mask not applied\n!!!!WARNING!!!!')
            return photon_info

        obs_photon_info = copy.deepcopy(photon_info)
        obs_photon_info = self.apply_exposure(obs_photon_info, obs_info)
        obs_photon_info = self.apply_PSF(obs_photon_info, obs_info)
        obs_photon_info = self.apply_energy_dispersion(obs_photon_info, obs_info)
        obs_photon_info = self.apply_mask(obs_photon_info, obs_info)
        
        return obs_photon_info

    ##########################################################################
    '''
    New code for Fermi analysis
    '''
    ##########################################################################
    
    #Moffat function used in PSF
    def King(x, sigma, gamma):
        return(1/(2*np.pi*sigma**2))*(1-(1/gamma))*(1+(1/(2*gamma))*(x**2/sigma**2))**(-gamma)
    
    def PSF_energy_dispersion(self, photon_info, angle_res, energy_res):
        num_photons = np.size(photon_info['energies'])
        '''
        #old method of approximating surface of sphere as flat and dropping a 2d-gaussian on it, then smear angles
        distances = np.sqrt(-2*angle_res**2*np.log(1-np.random.random(num_photons)))
        '''
        C0 = 6.38e-2
        C1 = 1.26e-3
        beta = 0.8
        #parameters below are not correct, need PSF FITS file
        Ntail = 1
        Score = 1
        Gcore = 1
        Stail = 1
        Gtail = 1
        Fcore = 1/(1 + Ntail*Stail**2/Score**2)
        x_vals = np.linspace(0, 10, 1000)
        PSF = Fcore*self.King(x_vals, Score, Gcore) + (1-Fcore)*self.King(x_vals, Stail, Gtail)
        x = self.draw_from_pdf(x_vals, PSF*2*np.pi*x_vals/np.sum(PSF*2*np.pi*x_vals), num_photons)
        S_p = np.sqrt((C0*(photon_info['energies']/100)**(-beta))**2 + C1**2)
        distances = 2*np.sin(x*S_p/2)
        rotations = 2*np.pi*np.random.random(num_photons)
        delta_thetas = distances*np.cos(rotations)
        delta_phis = distances*np.sin(rotations)
        photon_info['angles'][:,0] += delta_thetas
        over_indices = np.where(np.logical_or(photon_info['angles'][:,0] > np.pi, photon_info['angles'][:,0] < 0))
        photon_info['angles'][over_indices,0] = np.pi - photon_info['angles'][over_indices,0]%np.pi
        photon_info['angles'][over_indices,1] += np.pi
        photon_info['angles'][:,1] += delta_phis
        photon_info['angles'][:,1] %= 2*np.pi
        #smear energies
        sig = energy_res/(2*np.sqrt(2*np.log(2))) #assumes energy_res is FWHM of gaussian. sig*energy is then the standard deviation
        photon_info['energies'] = np.random.normal(photon_info['energies'], sig*photon_info['energies'])
        return photon_info
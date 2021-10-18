 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:27:18 2021

@author: alcheffot
"""
import numpy as np
import matplotlib.pyplot as plt
import poppy
import ceo
from astropy.io import fits
from astropy import units as u

class GMTpoppy(poppy.AnalyticOpticalElement):
    '''This is a wrapper class to obtain GMT transmission and opd information without 
    using the poppy FITSoptical element class.
    input : number of pixels'''
    def __init__(self,npix ,pupil_diam = 25.5*u.m, *args, **kwargs):
        """ If your optic has adjustible parameters, then save them as attributes here """
        self.pupil_diam =pupil_diam
        self.npix = npix
        poppy.AnalyticOpticalElement.__init__(self,name = 'gmt',
                                              planetype=poppy.poppy_core.PlaneType.pupil,**kwargs)
        self.pupil_diam =pupil_diam
        self.gmtopd = np.zeros((npix,npix))
        self.gmttrans = np.ones((npix,npix))

    def get_opd(self,wave):
        return self.gmtopd

    def get_transmission(self, wave):
        return self.gmttrans

    # behind the scenes poppy  will calculate:
    #    phasor = transmission = np.exp(1.j * 2 * np.pi / wave.wavelength * opd)


class phaseContrastSensor(object):
    """object implementing all the necessary functions for the phase contrast sensor
    to produce a measurement.
    
    Parameters
    ----------
    phaseMaskDiameter: float [lambda/D]
        Diameter of the central depression of the phase mask in unit of lambda/D
    phaseMaskDelay: float [wavelength]
        delay created by the central depression in unit of fraction of the central wavelength
    wavelength: float [meters]
        central wavelength of the light used in the phase contrast sensor. in meters
    telescopeDiameter: float (optionnal, default = 25.695 m) [meters]
        diameter of the telescope pupil in meter. needed for poppy to give 
        proper units and sizing to pictures.
    poppyOverSampling : int (optionnal, default = 6) [zero padding]
        Oversampling of the focal plan, or zeros padding of the pupil plan.6 
        means the array is 6 times bigger than the size of the pupil in pixels or
        that the diffraction limited PSF has 6 pixels accross the FWHM
    
    """
    
    def __init__(self,phaseMaskDiameter,phaseMaskDelay, wavelength, 
                 nPix, telescopeDiameter = 25.695, poppyOverSampling = 6 , bandwidth = 0):
        self.maskDelay = phaseMaskDelay
        self.maskRadius = phaseMaskDiameter/2
        self.maskDiameter = phaseMaskDiameter
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.telescopeDiameter = telescopeDiameter
        self.poppyOverSampling = poppyOverSampling
        self.nPix = nPix
        # self.fitsPupilFile = None
        # self.fitsPhaseFile = None
        # self.fitsPupilFileName = None
        # self.fitsPhaseFileName = None
        self.frame = np.zeros((nPix,nPix))
        self.nBPixelInPupil = None
        self.longExposureframe = None
        self.normalisEDFrame = None
        self.referenceFrame = None
        self.interactionMatrix = None
        self.reconstructionMatrix = None
        self.reconstructionVector = None
        self.commandVector = None
        self.curWavelength = None
        self.noMask = False
        self.fluxPers = None
        
    
    def reset(self):
        """reset to 0 the following valiables
                longExposureFrame
                frame
                fitsPhaseFile"""
        self.frame = np.zeros((self.nPix,self.nPix))
    
    def poppyFits(self,array2conv,fileName = 'gmtPhase.fits',typeOfArray = 'phase'):
        """This function takes an array and turns it into a fits that can be read by poppy.FITSopticsElement
        
        input: the array to convert,
        optionnal input: the name of the file, The type of array being converted
        output: the name of the fits file"""
        
        hdu  = fits.PrimaryHDU(array2conv)
        hdu.header['WAVELEN'] = self.wavelength
        hdu.header['DIAM'] = self.telescopeDiameter
        hdu.header['DIFFLMT'] = self.wavelength/self.telescopeDiameter
        hdu.header['OVERSAMP'] = 1
        hdu.header['DET_SAMP'] = 1
        hdu.header['PIXELSCL'] = self.telescopeDiameter/self.nPix
        if typeOfArray == 'phase':
            hdu.header['PIXUNIT'] = 'arcsec'
            hdu.header['BUNIT'] = 'meters'
            self.fitsPhaseFile = fileName
        elif typeOfArray == 'transmision':
            hdu.header['PIXUNIT'] = 'meters'
            self.fitsPupilFile = fileName
        
        hdul = fits.HDUList(hdu)
        hdul.writeto(fileName,overwrite= True)
        return fileName
        
    
    
    def propagate(self,wavefrontObject):
        """does the propagation into the phase contrast sensor and produces an image of the phase contrast 
        
        input: telescope wavefront, flux per unit of ms for the whole collecting area, telescope pupil
        optionnal input: 
        ouput: a frame
        
        """
        #turn the latest wavefront into a poppy compatible fits
        # if self.fitsPhaseFileName is not None:
        #     self.poppyFits(wavefrontObject.phase.host(),fileName = self.fitsPhaseFileName)
        # else:
        #     self.poppyFits(wavefrontObject.phase.host())
        
        if self.fluxPers == None:#I need a flux! If you did not set it up I put magic numbers in it
            #fluxPers = Number of Photons * telescope collecting area * 1 ms * telescope throuput * detector quantum efficiency
            self.fluxPers = wavefrontObject.nPhoton[0] * 357.0 *  0.9**4 * 0.5
        
        if self.curWavelength == None:
            self.curWavelength = self.wavelength
            
        # gmtpupil = poppy.FITSOpticalElement(transmission = self.fitsPupilFile,pixelscale = "PIXELSCL")
        
        # if self.fitsPhaseFile != None:
        #     telPhase = poppy.FITSOpticalElement(opd = self.fitsPhaseFile, pixelscale = "PIXELSCL")
        gmtInput = GMTpoppy(self.nPix)
        gmtInput.gmttrans=wavefrontObject.amplitude.host()
        gmtInput.gmtopd = wavefrontObject.phase.host()
        
        det = poppy.Detector(name='detector', pixelscale=self.telescopeDiameter/self.nPix, fov_pixels=self.nPix)
        
        if self.noMask == False:
            pm = poppy.CircularPhaseMask(name = 'zeus',radius = self.maskRadius* \
                                          (self.wavelength/self.telescopeDiameter)*180*3600/np.pi, \
                                              retardance = self.maskDelay*self.wavelength/self.curWavelength)
        
        osys = poppy.OpticalSystem(oversample = self.poppyOverSampling, npix = self.nPix)
        osys.add_pupil(gmtInput)
        # osys.add_pupil(gmtpupil)
        # if self.fitsPhaseFile != None:
        #     osys.add_pupil(telPhase)
        if self.noMask == False:
            osys.add_image(pm)
        else:
            osys.add_image()
        
        osys.add_pupil(det)

        poppyFrame = osys.calc_psf(wavelength = self.curWavelength)
        # print(np.sum(poppyFrame[0].data))
        self.frame += poppyFrame[0].data * self.fluxPers*10**-3

    
    
    def cameraNoise(self, RON=0,bias=0,excessNoise = 1):
        """This is meant to simulate photon noise and various camera noises
        (and introduces photon noise by default)
        input: frame, 
        optional input : RON, background noise, EMCCD Excessnoise
        output: noisy image
        """
        rng = np.random.default_rng()
        
        self.frame = rng.poisson(self.frame).astype(float)
        if excessNoise>1:
            nonnullPix = self.frame>0
            self.frame[nonnullPix] = rng.gamma(self.frame[nonnullPix]/(excessNoise-1))
            self.frame *= (excessNoise-1)
        self.frame += rng.normal(loc = bias, scale = RON, size = self.frame.shape)
        
        
    
    def set_reference_image(self,wavefrontObject):
        """ calculate the reference image of the perfect telescope without noise
        
        input: wavefront of the perfect telescope
        output: image of the perfect telescope through the phase contrast
        """
        # tmpFlux = self.fluxPers.copy()
        # self.fluxPers = 1
        self.propagate(wavefrontObject)
        self.referenceFrame = self.frame.copy()#/(self.fluxPers*10**-3)*self.nBPixelInPupil
        # self.fluxPers = tmpFlux.copy()
        return 1
        
    
    def transformationMatrix(self,):
        """Return the tranformation matrix for rotation and translation
        
        input: The matrix to transform, wanted rotation and x and y translations
        output: the transformed matrix"""
    
    def pupilRegistration(self):
        """The latest frame and finds the transfert matrix of the pupil
        
        input: the latest frame found in self
        optionnal input: 
        output: the parameters of the pupil (center and orientation)
        """
        
        
        
    def calibrate(self, ):
        """Unsure what this should do
        """
        
        
        
    def analyse(self, frame, exposureTime):
        """Unsure what this should do for now """


        
    def process(self,exposureTime):
        """Compute the normalised image from which one can retrieve the measurement
        
        input: the exposure time (to scale the reference image)
        optional input:
        output: the normalised image
        """
        exposureTime *= 10**3
        if not isinstance(self.referenceFrame , np.ndarray):
            print("you first need to create the reference frame, see method set_reference_image")
            return 0
        self.rawSignal = self.frame.copy()
        self.frame = (self.frame/(exposureTime))-self.referenceFrame
        
        # self.frame = self.frame - (self.referenceFrame*exposureTime)
        # self.frame = self.frame - (self.referenceFrame*exposureTime*self.fluxPers)
        # self.frame = self.frame - (self.referenceFrame*np.sum(self.frame))
        return 1
    
    

    
        
        
        
        
        
        
        





if __name__=='__main__':
    
    M2_n_modes = 675 # Karhunen-Loeve per M2 segment
    #M2_modes_set = u"ASM_DDKLs_S7OC04184_675kls"
    M2_modes_set = u"ASM_fittedKLs_doubleDiag"
    expTimeMs = 0.001
    
    nPx = 128
    
    #---- Telescope parameters
    D = 25.5                # [m] Diameter of simulated square (slightly larger than GMT diameter) 
    PupilArea = 357.0       # [m^2] Takes into account baffled segment borders
    M2_baffle = 3.5    # Circular diameter pupil mask obscuration
    project_truss_onaxis = False
    tel_throughput = 0.9**4 # M1 + M2 + M3 + GMTIFS dichroic = 0.9^4 
    nseg = 7
    detectorQE = 0.5
    
    simul_truss_mask=False  # If True, it applies to closed loop simulation, and slope-null calibration only (NOT IM calibration)
    gmtStandardIM = False
    
    #base configuration of the wavefront
    wl2nd = 1250e-9
    delta_wl2nd = 150e-9
    e0_Iband = 2.61e12 # zeropoint in photons/s in I band over the GMT pupil (check official photometry table!)
    
    #---- Scale zeropoint for the desired bandwidth:
    delta_wl_Iband = 150e-9  # check official photometry table!
    e0 = e0_Iband * (delta_wl2nd/delta_wl_Iband) / PupilArea    #in ph/m^2/s in the desired bandwidth
    
    mag = 9.3
    gs = ceo.Source([wl2nd,delta_wl2nd,e0], magnitude=mag, zenith=0.,azimuth=0., rays_box_size=D, 
            rays_box_sampling=nPx, rays_origin=[0.0,0.0,25])
    gs.rays.rot_angle = 15*np.pi/180
    nPhotPers = gs.nPhoton[0] * PupilArea * tel_throughput * detectorQE *0.1
    
    gmt = ceo.GMT_MX(M2_mirror_modes=M2_modes_set, M2_N_MODE=M2_n_modes)
    
    gmt.M2_baffle = M2_baffle
    gmt.project_truss_onaxis = project_truss_onaxis
    
    #create the object for the phase contrast
    zelda = phaseContrastSensor(1.5,0.0625,wl2nd,nPx)
    strokes = 25*10**-9 #later for interaction matrix
    zelda.fluxPers = nPhotPers
    # zelda.fluxPers = 50
    #first create the pupil mask (needed for poppy)
    
    #This is a prerequisite for poppy to work. Maybe that should be part of a method?
    gmt.reset()
    gs.reset()
    zelda.reset()
    gmt.propagate(gs)
    gmtMask = gs.amplitude.host()
    zelda.nBPixelInPupil = np.sum(gmtMask)
    plt.figure(1)
    plt.clf()
    plt.imshow(gmtMask)
    plt.colorbar()
    #turn the pupil maks into a format poppy can read
    fileName = zelda.poppyFits(gmtMask,'GMTpupil.fits', 'transmision')
    
    #does a frame without piston with mask
    toto = zelda.propagate(gs)
    plt.figure(2)
    plt.clf()
    plt.imshow(zelda.frame)
    plt.colorbar()
    
    # #Now apply a piston to one segment and do the phase contrast image of this:
    gmt.reset()
    gs.reset()
    zelda.reset()
    gmt.M2.modes.a[0,0] = 50*10**-9
    gmt.M2.modes.update()
    gmt.propagate(gs)
    tmp = gs.phase.host().copy()
    # #calculate a frame with the current phase contained in gs
    zelda.propagate(gs)
    test = np.sum(zelda.frame)
    # zelda.cameraNoise(1,0,2)
    plt.figure(3)
    plt.clf()
    plt.imshow(zelda.frame)
    plt.colorbar()
    #############test area for the wrapper poppy object#######################
    # set up the same propagation as before
    # gmt.reset()
    # gs.reset()
    
    # gmt.M2.modes.a[0,0] = 50*10**-9
    # gmt.M2.modes.update()
    # gmt.propagate(gs)
    
    # #now I need to take this telescope output and feed it to my wrapper object.
    # gmtp = GMTpoppy(nPx,pupil_diam = D)
    # # gmtp.gmttrans = gs.amplitude.host()
    # gmtp.gmttrans = gmtMask
    # gmto = GMTpoppy(nPx,pupil_diam = D)
    # gmto.gmtopd = gs.phase.host()
    # plt.figure(5)
    # plt.clf()
    # plt.imshow(gs.phase.host()-tmp)
    # #Then set up the optical system for the propagation
    
    # det = poppy.Detector(name='detector', pixelscale=zelda.telescopeDiameter/zelda.nPix, fov_pixels=zelda.nPix)

    # pm = poppy.CircularPhaseMask(name = 'zeus',radius = zelda.maskRadius* \
    #                               (zelda.wavelength/zelda.telescopeDiameter)*180*3600/np.pi, \
    #                                   retardance = zelda.maskDelay*zelda.wavelength/zelda.curWavelength)
        
    # osys = poppy.OpticalSystem(oversample = zelda.poppyOverSampling, npix = zelda.nPix)
    # osys.add_pupil(gmtp)
    # osys.add_pupil(gmto)
    # if zelda.noMask == False:
    #     osys.add_image(pm)
    # else:
    #     osys.add_image()
    
    # osys.add_pupil(det)

    # poppyFrame = osys.calc_psf(wavelength = zelda.curWavelength)
    
    # plt.figure(4)
    # plt.clf()
    # plt.imshow((poppyFrame[0].data*zelda.fluxPers*10**-3)-zelda.frame)
    # plt.colorbar()
    # print(test-np.sum(poppyFrame[0].data)*zelda.fluxPers*10**-3)
    #with the following we create the reference frame for later normalisation of picture.
    #first we need a flat wavefront
    gmt.reset()
    gs.reset()
    zelda.reset()
    gmt.propagate(gs)
    
    #now we feed the wavefront to the object
    zelda.set_reference_image(gs)
    plt.figure(4)
    plt.clf()
    plt.imshow(zelda.referenceFrame*gmtMask)
    plt.colorbar()
    plt.title("reference image")
    

    
    #next is the normalisation function, at this point you should already have created 
    #at least the reference frame
    #first we want to test with the flat wavefront
    
    gmt.reset()
    gs.reset()
    zelda.reset()
    
    for i in range(int(expTimeMs*10**3)):
        gs.reset()
        gmt.propagate(gs)
        zelda.propagate(gs)
    
    zelda.process(expTimeMs)
    plt.figure(5)
    plt.clf()
    plt.imshow(zelda.frame)
    plt.colorbar()
    plt.title("this image should be full of zeros")
    
    #now repeat this with a piston on one segment
    # gmt.reset()
    # gs.reset()
    # zelda.reset()
    # gmt.M2.modes.a[0,0] = 30*10**-9
    # gmt.M2.modes.update()
    
    # for i in range(int(expTimeMs*10**3)):
    #     gs.reset()
    #     gmt.propagate(gs)
    #     #calculate a frame with the latest frame and add it to itself
    #     zelda.propagate(gs)
    
    # zelda.process(expTimeMs)
    
    # plt.figure(6)
    # plt.clf()
    # plt.imshow(zelda.frame*gmtMask)
    # plt.colorbar()
    
    # # # next we create the reconstruction matrix accoring to GMT standards
    
    if gmtStandardIM:
        tmpfitsPupFile = zelda.fitsPupilFile
        gmt.project_truss_onaxis = False
        gs.reset()
        gmt.reset()
        gmt.propagate(gs)
        
        zelda.poppyFits(gs.amplitude.host() \
                                    ,fileName='noTrussGMTMask.fits', typeOfArray= 'amplitude')
    
    interactionMatrix  = np.zeros((nseg-1,nPx**2))
    for s in range(nseg-1):
        #set up the wavefront to feed to the propagation
        gs.reset()
        gmt.reset()
        zelda.reset()
        gmt.M2.modes.a[s,0] = strokes
        gmt.M2.modes.update()
        gmt.propagate(gs)
        
        # feed the wavefront to the propagate method
        zelda.propagate(gs)
        

        zelda.process(expTimeMs)
        print(np.mean(zelda.frame[gmtMask.astype(bool)]))
        interactionMatrix[s] = zelda.frame.reshape(-1)/(strokes)
        if gmtStandardIM:#if we removed it replace the truss on the images
            zelda.frame *= gmtMask
    
    # plt.figure(7)
    # plt.clf()
    # fig,axs = plt.subplots(num=7, ncols=3,nrows=3)
    # for i in range(interactionMatrix.shape[0]):
    #     axs[i//3,i%3].imshow(interactionMatrix[i].reshape(gmtMask.shape))
    reconstructionMatrix = np.zeros((nPx**2,nseg))
    reconstructionMatrix[:,:6] = np.linalg.pinv(interactionMatrix)
    
    # if gmtStandardIM:
    #     zelda.fitsPupilFile = tmpfitsPupFile
    #     gmt.project_truss_onaxis = True
    
    
    # #next we generate a random piston on a random segment and calculate the reconstructed piston
    # randSeg = np.random.randint(0,7)
    randSeg = 1
    randpist = 150*10**-9
    # randpist = (np.random.rand()-0.25)*750*10**-9/2
    print("segment {} has {} nm piston".format(randSeg,randpist*10**9))
    
    gmt.reset()
    gs.reset()
    zelda.reset()
    gmt.M2.modes.a[randSeg,0] = randpist
    gmt.M2.modes.update()
    gmt.propagate(gs)
    zelda.propagate(gs)
    
    zelda.process(expTimeMs)
    reconstructionVector =  zelda.frame.reshape(-1) @ reconstructionMatrix 
    reconstructionVector -= reconstructionVector[6]
    print(reconstructionVector)
    
    # lets do a sweep for the fun of it
    pist2apply = np.arange(-60,61,1)*10**-9
    randSeg = 5
    res = []
    for p in pist2apply:
        gmt.reset()
        gs.reset()
        zelda.reset()
        gmt.M2.modes.a[randSeg,0] = p
        gmt.M2.modes.update()
        gmt.propagate(gs)
        
        zelda.propagate(gs)
        zelda.process(expTimeMs)
        reconstructionVector = zelda.frame.reshape(-1) @ reconstructionMatrix
        reconstructionVector -= reconstructionVector[6]
        res.append(reconstructionVector)
        
    # res = np.array(res)
    # and display how the result went
    plt.figure(8)
    plt.clf()
    plt.plot(pist2apply,res,'+-')
    
    # test the camera noise
    gmt.reset()
    zelda.reset()
    for t in range(int(expTimeMs*10**3)):
        gs.reset()
        gmt.propagate(gs)
        zelda.propagate(gs)
    
    noiselessFrame = zelda.frame.copy()
    zelda.frame*=150
    plt.figure(9,figsize=(12,8))
    plt.clf()
    fig,axs = plt.subplots(num = 9, nrows=1,ncols=3)
    axs[0].imshow(zelda.frame)
    
    zelda.cameraNoise(10,0)
    axs[1].imshow(zelda.frame)
    
    #try to normalise the now noisy image
    zelda.process(expTimeMs)
    axs[2].imshow(zelda.frame)
    
    #Now a noisy sweep to see if it works 
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #This may take a very long while if you execut it!
    
    pist2apply = np.arange(-3*750,3*751,750/8)*10**-9
    randSeg = 4
    res = []
    
    expTimeMs = 0.150
    for p in pist2apply:
        zelda.reset()
        gmt.reset()
        gs.reset()
        
        gmt.M2.modes.a[randSeg,0] = p
        gmt.M2.modes.update()
        gmt.propagate(gs)
        
        # for i in range(int(expTimeMs*10**3)):
        gs.reset()
        gmt.propagate(gs)
        zelda.propagate(gs)
        #this is cheating do not do that at home!
        zelda.frame *= expTimeMs*10**3
        zelda.cameraNoise(1,0,2)
        zelda.process(expTimeMs)
        reconstructionVector = zelda.frame.reshape(-1) @ (reconstructionMatrix)
        reconstructionVector -= reconstructionVector[6]
        res.append(reconstructionVector)
        
    # res = np.array(res)
    # and display how the result went
    plt.figure(10)
    plt.clf()
    plt.plot(pist2apply,res,'+-')
    
    
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

class phaseContrastSensor(object):
    """object implementing all the necessary functions for the phase contrast sensor
    to produce a measurement.
    Documentation under developpement."""
    
    def __init__(self,phaseMaskDiameter,phaseMaskDelay, wavelength, 
                 nPix, bandwidth = 0, telescopeDiameter = 25.695, poppyOverSampling = 6):
        self.maskDelay = phaseMaskDelay
        self.maskRadius = phaseMaskDiameter/2
        self.maskDiameter = phaseMaskDiameter
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.telescopeDiameter = telescopeDiameter
        self.poppyOverSampling = poppyOverSampling
        self.nPix = nPix
        self.fitsPupilFile = None
        self.fitsPhaseFile = None
        self._p = np.zeros((nPix,nPix))
        self.normalisEDFrame = None
        self.referenceFrame = None
        self.interactionMatrix = None
        self.reconstructionMatrix = None
        self.reconstructionVector = None
        self.commandVector = None
        
    
    def reset(self):
        """reset to 0 the following valiables
                longExposureFrame
                frame
                fitsPhaseFile"""
        self._p = np.zeros((self.nPix,self.nPix))
    
    def poppyFits(self,array2conv,fileName = 'gmtPhase.fits',typeOfArray = 'phase'):
        """This function takes an array and turns it into a fits that can be read by poppy.FITSopticsElement
        
        input: the array 2 convert,
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
        
    
    def normaliseFrame(self,frame,expTimeMs):
        """Normalise the frame given in input
        
        input: a frame
        optional input: exposure time in milliseconds
        output: normalised frame"""
        if not isinstance(self.referenceFrame , np.ndarray):
            print("you first need to create the reference frame, see method setReference")
            return 0
        self.normalisEDFrame = frame - (self.referenceFrame*expTimeMs)
        return 1
    
    def propagate(self,wavefrontObject, fluxPerMs,curWavelength = None,noMask = False):
        """does the propagation into the phase contrast sensor and produces an image of the phase contrast 
        
        input: telescope wavefront, flux per unit of ms for the whole collecting area, telescope pupil
        optionnal input: 
        ouput: a frame
        
        """
        self.poppyFits(wavefrontObject.phase.host())
        
        if curWavelength == None:
            curWavelength = self.wavelength
            
        gmtpupil = poppy.FITSOpticalElement(transmission = self.fitsPupilFile,pixelscale = "PIXELSCL")
        
        if self.fitsPhaseFile != None:
            telPhase = poppy.FITSOpticalElement(opd = self.fitsPhaseFile, pixelscale = "PIXELSCL")
        
        det = poppy.Detector(name='detector', pixelscale=self.telescopeDiameter/self.nPix, fov_pixels=self.nPix)
        
        if noMask == False:
            pm = poppy.CircularPhaseMask(name = 'zeus',radius = self.maskRadius* \
                                          (self.wavelength/self.telescopeDiameter)*180*3600/np.pi, \
                                              retardance = self.maskDelay*self.wavelength/curWavelength)
        
        osys = poppy.OpticalSystem(oversample = self.poppyOverSampling, npix = self.nPix)
        osys.add_pupil(gmtpupil)
        if self.fitsPhaseFile != None:
            osys.add_pupil(telPhase)
        if noMask == False:
            osys.add_image(pm)
        else:
            osys.add_image()
        
        osys.add_pupil(det)

        poppyFrame = osys.calc_psf(wavelength = curWavelength)
        self._p += poppyFrame[0].data *fluxPerMs
        return 1
    
        
        
    
    def camera(self):
        """This is meant to simulate photon noise and various camera noises"""
    
    
    def setReference(self,wavefrontObject,flux):
        """ calculate the reference image of the perfect telescope without noise
        
        input: wavefront of the perfect telescope
        output: image of the perfect telescope through the phase contrast
        """
        self.propagate(wavefrontObject,flux)
        self.referenceFrame = self.frame.copy()
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
        
        
        
    def calibrate(self, strokes, flux, nseg, wavefrontObject,telescopeObject,gmtStandardIM = True, gmtWithTrussMask = []):
        """creates the interaction matrix of the phase contrast sensor
        and use it to do the reconstruction matrix
        
        input: piston to be applied to build the interaction matrix, the flux of the star, the number of segments, the wavefront object, the telescope object
        optionnal input: a flag describing whether to use the gmt reconstruction matrix protocole or not
        output: reconstruction matrix
        """
        if gmtStandardIM:
            if gmtWithTrussMask is []:
                print("Warning: gmtMaskWithTruss not provided \n Assuming the last amplitude in wavefront object contains a uniformly lit and contain the truss " )
                gmtWithTrussMask = wavefrontObject.amplitude.host()
            tmpfitsPupFile = self.fitsPupilFile
            telescopeObject.project_truss_onaxis = False
            wavefrontObject.reset()
            telescopeObject.reset()
            telescopeObject.propagate(gs)
            
            self.poppyFits(wavefrontObject.amplitude.host() \
                                        ,fileName='noTrussGMTMask.fits', typeOfArray= 'amplitude')
        
        self.interactionMatrix  = np.zeros((nseg,self.nPix**2))
        for s in range(nseg):
            #set up the wavefront to feed to the propagation
            wavefrontObject.reset()
            telescopeObject.reset()
            telescopeObject.M2.modes.a[s,0] = strokes*10**-9
            telescopeObject.M2.modes.update()
            telescopeObject.propagate(wavefrontObject)
            
            #turn the wavefront into a poppy compatible format
            self.poppyFits(wavefrontObject.phase.host())
            
            # feed the wavefront to the propagate method
            self.propagate(flux)
            
            if gmtStandardIM:#if we removed it replace the truss on the images
                self.frame *= gmtWithTrussMask
            errorReturn = self.normaliseFrame(self.frame,1)
            self.interactionMatrix[s] = self.normalisEDFrame.reshape(-1)/strokes
        
        self.reconstructionMatrix = np.linalg.pinv(self.interactionMatrix)
        
        if gmtStandardIM:
            self.fitsPupilFile = tmpfitsPupFile
            telescopeObject.project_truss_onaxis = True
        
        return 1
        
        
    def analyse(self, frame, expTimeMs):
        """analyse the image and return the reconstruction vector
        
        input: an image array, the reconstruction matrix
        optionnal input:
        output: a vector of piston per segments"""
        if not isinstance(self.reconstructionMatrix, np.ndarray):
            print("error: requires the reconstruction matrix, please execute method calibrate")
            return 0
        errorReturn = self.normaliseFrame(frame, expTimeMs)
        self.reconstructionVector = self.normalisEDFrame.reshape(-1) @ self.reconstructionMatrix
        self.reconstructionVector -= self.reconstructionVector[0]
        return 1
        
    def commandVector(self):
        """take the reconstruction vector and turn it into a CEO compatible command vector
        
        input: the reconstruction vector
        optionnal input:
        output: the command vector for the NGAO"""
    
        
        
        
        
        
        
        





if __name__=='__main__':
    print('toto')
    
    M2_n_modes = 675 # Karhunen-Loeve per M2 segment
    #M2_modes_set = u"ASM_DDKLs_S7OC04184_675kls"
    M2_modes_set = u"ASM_fittedKLs_doubleDiag"
    
    
    nPx = 1024
    
    #---- Telescope parameters
    D = 25.5                # [m] Diameter of simulated square (slightly larger than GMT diameter) 
    PupilArea = 357.0       # [m^2] Takes into account baffled segment borders
    M2_baffle = 3.5    # Circular diameter pupil mask obscuration
    project_truss_onaxis = True
    tel_throughput = 0.9**4 # M1 + M2 + M3 + GMTIFS dichroic = 0.9^4 
    nseg = 7

    simul_truss_mask=True  # If True, it applies to closed loop simulation, and slope-null calibration only (NOT IM calibration)
    
    #base configuration of the wavefront
    wl2nd = 850e-9
    delta_wl2nd = 20e-9
    e0_Iband = 2.61e12 # zeropoint in photons/s in I band over the GMT pupil (check official photometry table!)
    
    #---- Scale zeropoint for the desired bandwidth:
    delta_wl_Iband = 150e-9  # check official photometry table!
    e0 = e0_Iband * (delta_wl2nd/delta_wl_Iband) / PupilArea    #in ph/m^2/s in the desired bandwidth
    
    mag = 10
    gs = ceo.Source([wl2nd,delta_wl2nd,e0], magnitude=mag, zenith=0.,azimuth=0., rays_box_size=D, 
            rays_box_sampling=nPx, rays_origin=[0.0,0.0,25])
    gs.rays.rot_angle = 15*np.pi/180
    nPhotPerMs = gs.nPhoton[0] * PupilArea * 10**-3
    
    gmt = ceo.GMT_MX(M2_mirror_modes=M2_modes_set, M2_N_MODE=M2_n_modes)
    
    gmt.M2_baffle = M2_baffle
    gmt.project_truss_onaxis = project_truss_onaxis
    
    #create the object for the phase contrast
    zelda = phaseContrastSensor(2,0.225,wl2nd,nPx)
    
    #first create the pupil mask (needed for poppy)
    
    gmt.reset()
    gs.reset()
    gmt.propagate(gs)
    gmtMask = gs.amplitude.host()
    plt.figure(1)
    plt.clf()
    plt.imshow(gmtMask)
    #turn the pupil maks into a format poppy can read
    fileName = zelda.poppyFits(gmtMask,'GMTpupil.fits', 'transmision')
    
    #does a frame without piston with mask
    toto = zelda.propagate(1,noMask=False)
    plt.figure(2)
    plt.clf()
    plt.imshow(zelda.frame*np.sum(gmtMask))
    
    #Now apply a piston to one segment and do the phase contrast image of this:
    gmt.reset()
    gs.reset()
    gmt.M2.modes.a[0,0] = 200*10**-9
    gmt.M2.modes.update()
    gmt.propagate(gs)
    
    #extract the phase out of this
    curPhase = gs.phase.host()
    
    #turn the phase into a poppy compatible format
    fileName = zelda.poppyFits(curPhase)
    #calculate a frame with this phase 
    toto = zelda.propagate(nPhotPerMs)
    
    plt.figure(3)
    plt.clf()
    plt.imshow(zelda.frame)
    
    #with the following we create the reference frame for later normalisation of picture.
    #first we need a flat wavefront
    gmt.reset()
    gs.reset()
    gmt.propagate(gs)
    
    #now we feed the wavefront to the object
    zelda.setReference(gs.phase.host(), 1)
    plt.figure(4)
    plt.clf()
    plt.imshow(zelda.referenceFrame)
    
    #next is the normalisation function, at this point you should already have created 
    #at least the reference frame
    #first we want to test with the flat wavefront
    zelda.normaliseFrame(zelda.frame, 1)
    plt.figure(5)
    plt.clf()
    plt.imshow(zelda.normalisEDFrame)
    plt.title("this image should be empty")
    
    #now repeat this with a piston on one segment
    gmt.reset()
    gs.reset()
    gmt.M2.modes.a[0,0] = 200*10**-9
    gmt.M2.modes.update()
    gmt.propagate(gs)
    fileName = zelda.poppyFits(gs.phase.host())
    #calculate a frame with this phase 
    toto = zelda.propagate(1)
    
    zelda.normaliseFrame(zelda.frame,1)
    
    plt.figure(6)
    plt.clf()
    plt.imshow(zelda.normalisEDFrame)
    
    # next we create the reconstruction matrix accoring to GMT standards
    zelda.calibrate(25,1,nseg,gs,gmt,gmtWithTrussMask=gmtMask)
    
    #next we generate a random piston on a random segment and calculate the reconstructed piston
    randSeg = np.random.randint(0,7)
    randpist = (np.random.rand()-0.25)*750*10**-9
    print("segment {} has {} nm piston".format(randSeg,randpist*10**9))
    
    gmt.reset()
    gs.reset()
    gmt.M2.modes.a[randSeg,0] = randpist
    gmt.M2.modes.update()
    gmt.propagate(gs)
    fileName = zelda.poppyFits(gs.phase.host())
    
    zelda.propagate(1)
    zelda.analyse(zelda.frame,1)
    print(zelda.reconstructionVector)
    
    #lets do a sweep for the fun of it
    pist2apply = np.arange(-3*750,3*751,750/8)*10**-9
    randSeg = 4
    res = []
    for p in pist2apply:
        gmt.reset()
        gs.reset()
        gmt.M2.modes.a[randSeg,0] = p
        gmt.M2.modes.update()
        gmt.propagate(gs)
        fileName = zelda.poppyFits(gs.phase.host())
        
        zelda.propagate(1)
        zelda.analyse(zelda.frame,1)
        res.append(zelda.reconstructionVector)
        
    res = np.ndarray(res)
    #and display how the result went
    plt.figure(7)
    plt.clf()
    plt.plot(pist2apply,res,'+-')
    
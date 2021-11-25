#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# #############################################################
#                                                             #
#  Original work: Cedric Plantet, cedric.plantet@inaf.it      #
#                                                             #
#  Python implementation: Fabio Rossi, fabio.rossi@inaf.it    #
#                                                             #
###############################################################

Class implementing the LIFT method, bsed on the paper:

@inproceedings{ao4elt3_13355,
        author = {{Cedric} {Plantet} and {Serge} {Meimon} and {Jean-Marc} {Conan} and {Benoit} {Neichel} and {Thierry} {Fusco}}, 
          title = {On-sky validation of LIFT on GEMS},
      booktitle = {Proceedings of the Third AO4ELT Conference},
          year = {2013},
        editor = {Simone Esposito and Luca Fini},
      publisher = {INAF - Osservatorio Astrofisico di Arcetri},
        address = {Firenze},
          isbn = {978-88-908876-0-4},
            doi = {10.12839/AO4ELT3.13355},
        bibcode = {2013aoel.confE..76P},
}
"""

import numpy as np
import cupy as cp
import scipy.ndimage as ndimage
from configparser import ConfigParser
from sympy.parsing.sympy_parser import parse_expr
from astropy.io import fits
import matplotlib.pyplot as plt
import ceo
from lift import *
import time


class LiftCEO(LIFT):
    
    
    
    def __init__(self,path , parametersFile,sectionName='lift'):
        """
        initialise the basic parameters and array needed
        """
        super().__init__( path = path, parametersFile = parametersFile, sectionName = sectionName)
        self.nPix = self.gridSize 
        # self.mask = self.CEOcompa(pupilMask)
        # self.mask = pupilMask.copy()
        self.fluxPers = 0
        self.frame = np.zeros((self.gridSize,self.gridSize))
        self.currentPhaseEstimate = []
        self.A_ML = []
        
    def CEOcompa(self,array2format):
        array2format = zeroPad(array2format,int(array2format.shape[0]/2*(self.orig_sampling_factor*2-1)))
        array2format = rebin(array2format,(self.gridSize\
                                           ,self.gridSize))
        return array2format

    def complex_amplitude(self,wavefrontObject, FT=False):
        A0 = cp.zeros((self.nPx*self.npad-1,self.nPx*self.npad-1))
        A0[0:self.nPx,0:self.nPx] = ceo.ascupy(wavefrontObject.amplitude)
        F0 = cp.zeros((self.nPx*self.npad-1,self.nPx*self.npad-1))
        F0[0:self.nPx,0:self.nPx] = ceo.ascupy(wavefrontObject.phase)
        # F0 -= A0*cp.array(self.gs.piston())   # Remove global piston (unsure we want this for lift)
        if FT==False:
            return A0*cp.exp(1j*self.knumber*F0)
        else:
            return cp.fft.fft2(A0*cp.exp(1j*self.knumber*F0))
        
    def propagate(self, wavefrontObject):
        """ function that produce an image"""
        # phase = wavefrontObject
        phase = self.CEOcompa(wavefrontObject.phase.host().copy())
        phase *= 2*np.pi/wavefrontObject.wavelength
        #this is a fix for some reusing of variable in lift that should not happen
        self.roi_tip, self.roi_tilt = self.radians_per_pixel/2, self.radians_per_pixel/2

        tmpFrame = self.focalPlaneImageLIFTAberration(phase)
        # tmpFrame *= nPhotPerMs
        tmpFrame *= self.fluxPers*10**-3/np.sum(tmpFrame)*2
        self.frame += tmpFrame

        return phase
    
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
    
    
    def calibrate(self):
        """PSF centroid? pupil registration?"""
        
    
    def reset(self):
        """reset the last frame and the camera image (at least)"""
        self.frame = np.zeros((self.gridSize,self.gridSize))
        
        
    def process(self):
        """turn the raw image out of the detector into a 'what ever' that allows to recover the piston per segment
        for lift it """
        
        
    def analyse(self, wavefrontObject):
        """propagate a noiseless wavefront to the detector and do the analysis of the image
        """
        self.propagate(wavefrontObject)
        self.process()
        


if __name__=='__main__':
    M2_n_modes = 675 # Karhunen-Loeve per M2 segment
    #M2_modes_set = u"ASM_DDKLs_S7OC04184_675kls"
    M2_modes_set = u"ASM_fittedKLs_doubleDiag"
    expTimeMs = 150
    
    nPx = 256
    
    #---- Telescope parameters
    D = 25.5                # [m] Diameter of simulated square (slightly larger than GMT diameter) 
    PupilArea = 357.0       # [m^2] Takes into account baffled segment borders
    M2_baffle = 3.5    # Circular diameter pupil mask obscuration
    project_truss_onaxis = True
    tel_throughput = 0.9**4 # M1 + M2 + M3 + GMTIFS dichroic = 0.9^4 
    nseg = 7
    detectorQE = 0.5
    
    simul_truss_mask=True  # If True, it applies to closed loop simulation, and slope-null calibration only (NOT IM calibration)
    gmtStandardIM = True
    
    #base configuration of the wavefront
    wl2nd = 805*10**-9
    delta_wl2nd = 150e-9
    e0_Iband = 2.61e12 # zeropoint in photons/s in I band over the GMT pupil (check official photometry table!)
    
    #---- Scale zeropoint for the desired bandwidth:
    delta_wl_Iband = 150e-9  # check official photometry table!
    e0 = e0_Iband * (delta_wl2nd/delta_wl_Iband) / PupilArea    #in ph/m^2/s in the desired bandwidth
    
    mag = 10
    gs = ceo.Source([wl2nd,delta_wl2nd,e0], magnitude=mag, zenith=0.,azimuth=0., rays_box_size=D, 
            rays_box_sampling=nPx, rays_origin=[0.0,0.0,25])
    gs.rays.rot_angle = 15*np.pi/180
    nPhotPerMs = gs.nPhoton[0] * PupilArea * 10**-3 * tel_throughput * detectorQE
    
    gmt = ceo.GMT_MX(M2_mirror_modes=M2_modes_set, M2_N_MODE=M2_n_modes)
    
    gmt.M2_baffle = M2_baffle
    gmt.project_truss_onaxis = project_truss_onaxis
    
    #lift is initialised from it's own parameter file currently stored here:
    path2liftParam = "/home/alcheffot/CEO/"
    liftParamFileName = "paramsLIFT"
    
    lift = LiftCEO(path2liftParam,liftParamFileName)
    lift.fluxPerMs = nPhotPerMs
    
    #This is a prerequisite for lift to work.
    gmt.reset()
    gs.reset()
    gmt.propagate(gs)
    gmtMask = gs.amplitude.host()
    lift.mask = lift.CEOcompa(gmtMask)
    
    
    
    plt.figure(1)
    plt.clf()
    plt.imshow(lift.mask)
    
    #creating a PSF using the lift propagate method
    lift.propagate(gs)
    plt.figure(2)
    plt.clf()
    plt.imshow(lift.frame)
    
    # trying a reconstruction with lift
    currentPhaseEstimate, A_ML = lift.phaseEstimation(lift.frame, False, 1e-6, 1e-5)
    
    #the result is given in micrometers
    print(A_ML[:6]*wl2nd*10**9/(2*np.pi))
    
    plt.figure(3)
    plt.clf()
    lift.cameraNoise(1,0)
    plt.imshow(lift.frame)
    
    # trying a reconstruction with lift
    currentPhaseEstimate, A_ML = lift.phaseEstimation(lift.frame, False, 1e-6, 1e-5)
    
    #the result is given in micrometers
    print(A_ML[:6]*wl2nd*10**9/(2*np.pi))
    
    #lets try a single segment at a time
    
    stroke = 150*10**-9
    strokes = np.linspace(-1500,1500,61)*10**-9
    # strokes = [150*10**-9]
    res = []
    resall = []
    inputPhase = []
    
    R2m = np.zeros((6,6))
    R2m[0,1] = 1
    R2m[1,0] = 1
    R2m[2,5] = 1
    R2m[3,4] = 1
    R2m[4,3] = 1
    R2m[5,2] = 1
    
    for s in range(7):
        res = []
        for p in strokes:
            gmt.reset()
            gs.reset()
            lift.reset()
            gmt.M2.modes.a[s,0] = p
            gmt.M2.modes.update()
            gmt.propagate(gs)
            # inputPhase.append(lift.propagate(gs))
            lift.propagate(gs)
            
            currentPhaseEstimate, A_ML = lift.phaseEstimation(lift.frame, False, 1e-6, 1e-5)
            A_ML = A_ML[:6]@R2m
            # print(A_ML[:6]*wl2nd*10**9/(2*np.pi))
            res.append(A_ML*lift.lambdaValue*10**-3/(2*np.pi))
        resall.append(res)

    plt.figure(5)
    plt.clf()
    plt.plot(strokes*10**6,res,'+-' )
    plt.ylabel("retrieved piston in $\mu$m")
    plt.xlabel("true piston applied in $\mu$m")
    plt.tight_layout()
    
    resall = np.array(resall)
    resall = np.moveaxis(resall,1,0)
    plt.figure(4)
    plt.clf()
    fig,axs = plt.subplots(num = 4, nrows = 3, ncols = 3, sharex=True, sharey=True)
    for s in range(7):
        axs[s//3,s%3].plot(strokes*10**6,resall[:,s,:])
        axs[s//3,s%3].set_title('segment {}'.format(s))
    fig.legend(['S0', 'S1', 'S2', 'S3', 'S4', 'S5' ],loc="lower right" , ncol = 3)
    
    fig.text(0.5, 0.04, 'true Piston in $\mu$m', ha='center')
    fig.text(0.04, 0.5, 'measured Piston in $\mu$m', va='center', rotation='vertical')
    fig.tight_layout()
    
    rng = np.random.default_rng(12345)
    truspos = []
    res = []
    for i in range(100):
        p = (rng.random(6)-0.5)*3.5*10**-6
        truspos.append(p)
        gmt.reset()
        gs.reset()
        lift.reset()
        gmt.M2.modes.a[:6,0] = p
        gmt.M2.modes.update()
        gmt.propagate(gs)
        # inputPhase.append(lift.propagate(gs))
        lift.propagate(gs)
        
        currentPhaseEstimate, A_ML = lift.phaseEstimation(lift.frame, False, 1e-6, 1e-5)
        A_ML = A_ML[:6]@R2m
        # print(A_ML[:6]*wl2nd*10**9/(2*np.pi))
        res.append(A_ML*lift.lambdaValue*10**-3/(2*np.pi))
        # resall.append(res)
    
    plt.figure(6)
    plt.clf()
    s = 0
    truspos = np.array(truspos)
    res = np.array(res)
    plt.plot(truspos[:,s]*10**6,res[:,s],'+')
    plt.plot(strokes*10**6, resall[:,s,s])
    plt.xlabel('true piston in $\mu$m ')
    plt.ylabel('reconstructed piston in $\mu$m')
    plt.grid()
    
    # inputPhase = np.array(inputPhase)
    # hdu = fits.PrimaryHDU(inputPhase)
    # hdul = fits.HDUList(hdu)
    # hdul.writeto('gmtInputPhase.fits',overwrite= True)
    # res = []
    # #Now I would like to make sure CEO and lift speak the same language regarding the segments
    # for s in range(6):
    #     gmt.reset()
    #     gs.reset()
    #     lift.reset()
    #     gmt.M2.modes.a[s,0] = 150*10**-9
    #     gmt.M2.modes.update()
    #     gmt.propagate(gs)
    #     # inputPhase.append(lift.propagate(gs))
    #     lift.propagate(gs)
        
    #     currentPhaseEstimate, A_ML = lift.phaseEstimation(lift.frame, False, 1e-6, 1e-5)
    #     # print(A_ML[:6]*wl2nd*10**9/(2*np.pi))
    #     res.append(A_ML[:6]*lift.lambdaValue*10**-3/(2*np.pi))
        
        
    # plt.figure(6)
    # plt.clf()
    # plt.plot(res,'+-' )
    # plt.ylabel("retrieved piston in $\mu$m")
    # plt.xlabel("true piston applied in $\mu$m")
    # plt.legend(['S0','S1','S2','S3','S4','S5'])
    # plt.tight_layout()
    
    # #lift does not number the segments the same way hene I use the reconstruction Matrix
    # #to make the correspondance between CEO and lift
    # R2m = np.zeros((6,6))
    # R2m[0,1] = 1
    # R2m[1,0] = 1
    # R2m[2,5] = 1
    # R2m[3,4] = 1
    # R2m[4,3] = 1
    # R2m[5,2] = 1
    
    # resCEO = [a@R2m for a in res]
    # plt.figure(7)
    # plt.clf()
    # plt.plot(resCEO,'+-' )
    # plt.ylabel("retrieved piston in $\mu$m")
    # plt.xlabel("true piston applied in $\mu$m")
    # plt.legend(['S0','S1','S2','S3','S4','S5'])
    # plt.tight_layout()
    



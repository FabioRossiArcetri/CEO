#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:50:05 2021

@author: alcheffot
A wrapper meant to caontain everything the second channel need for working with any of the wavefront sensors
"""
#----- CEO: GMT ray-tracing and wavefront sensing simulator
import ceo

#----- System related packages
import sys
import datetime as dt
import os.path
import pickle

#----- Math and Scientific Computing
import math
import numpy as np
import cupy as cp
from scipy import ndimage
from scipy import signal 
from scipy.interpolate import CubicSpline
import poppy
from astropy.io import fits

#----- Visualization
import matplotlib.pyplot as plt

# .ini file parsing
from configparser import ConfigParser,NoOptionError
from datetime import datetime

PYRAMID_SENSOR = 'pyramid'
PHASE_CONTRAST_SENSOR = 'phasecontrast' 
LIFT = 'lift'

def rebin(arr, new_shape):
    shape = (new_shape[0], int(arr.shape[0] // new_shape[0]),
             new_shape[1], int(arr.shape[1] // new_shape[1]))
    return arr.reshape(shape).mean(-1).mean(1)

class Chan2(object):
    """object meant to wrap everything the second channel may do"""
    
    def __init__(self,path,parametersFile,sectionName = '',sensorId = 0):
        parser = ConfigParser()
        parser.read(path + parametersFile + '.ini')
        
        self.chan1wl = []
        self.pistEstTime = []
        self.pistEstList = []
        self.correctionList = []
        self.debugframe = []
        self.ogc = 1
        self.ogc_iter = []
        self.fluxEst = []
        self.forCorrection = np.zeros(7)
        
        self.VISU = eval(parser.get('general', 'VISU'))
        self.simul_truss_mask = eval(parser.get('telescope', 'simul_truss_mask'))
        self.nseg = eval(parser.get('telescope','nseg'))
        self.active_corr = eval(parser.get('2ndChan', 'active_corr'))[sensorId]
        try:
            self.rescaleAtmResidual = eval(parser.get('2ndChan', 'rescale_atm_residual'))[sensorId]
            if self.rescaleAtmResidual:
                self.rescaleTo = eval(parser.get('2ndChan', 'rescale_to'))[sensorId]
                self.rescaleFactor = 1
                print('second channel atmospheric residuals will be rescaled to {} nm'
                      .format(self.rescaleTo*10**9))
        except NoOptionError:
            print('no atmospheric residuals rescaling' )
            self.rescaleAtmResidual = False
            self.rescaleFactor = 1
        
        #addition of parameters for the second channel: first the second channel source parameters
        self.cwl = eval(parser.get('2ndChan', 'wavelength2'))[sensorId]
        self.band = eval(parser.get('2ndChan', 'delta_wavelength2'))[sensorId]
        self.e0_inband = eval(parser.get('2ndChan', 'e0_inClosestBand'))[sensorId]
        self.band_inband = eval(parser.get('2ndChan', 'deltaWavelength_ClosestBand'))[sensorId]
        self.mag = eval(parser.get('2ndChan', 'mag'))[sensorId]
        self.e0 = self.e0_inband * (self.band/self.band_inband) / eval(parser.get('telescope', 'PupilArea'))    #in ph/m^2/s in the desired bandwidth
        self.throughput =  eval(parser.get('2ndChan','throughput'))[sensorId]  # NGWS board: 0.4
        self.detQE = eval(parser.get('2ndChan','detectorQE'))[sensorId]
        #setup of the parameters for the sensor itself.
        self.sensorType = eval(parser.get('2ndChan', 'sensorType'))
        self.RONval = eval(parser.get('2ndChan','RONval'))[sensorId]                # e- RMS
        try:
            self.background = eval(parser.get('2ndChan','background'))[sensorId]
        except NoOptionError:
            self.background = 0
        self.pyr_angle = eval(parser.get('2ndChan','pyr_angle'))[sensorId]         # angle between pyramid facets and GMT pupil (in deg)
        self.detthr = eval(parser.get('2ndChan','detthr'))[sensorId]
        
        # initialisation of the second channel wavefront sensor
        if self.sensorType.lower() == 'idealpistonsensor':
            self.pyr_binning = eval(parser.get('2ndChan', 'pyr_binning'))[sensorId]
            self.nLenslet = eval(parser.get('2ndChan','nLenslet'))[sensorId]//self.pyr_binning            # sub-apertures across the pupil
            self.nPx = self.nLenslet*eval(parser.get('2ndChan','nPx'))[sensorId]
            print('initialised elsewhere in the code')
            
        if self.sensorType.lower() == PYRAMID_SENSOR:
            self.pyr_binning = eval(parser.get('2ndChan', 'pyr_binning'))[sensorId]
            self.nLenslet = eval(parser.get('2ndChan','nLenslet'))[sensorId]//self.pyr_binning            # sub-apertures across the pupil
            self.nPx = self.nLenslet*eval(parser.get('2ndChan','nPx'))[sensorId]
            self.applyOgc = eval(parser.get('2ndChan', 'applyOgc'))[sensorId]
            self.pyr_separation = eval(parser.get('2ndChan','pyr_separation'))[sensorId]//self.pyr_binning     # separation between centers of adjacent sub-pupil images on the detector [pix]
            self.pyr_modulation = eval(parser.get('2ndChan','pyr_modulation'))[sensorId]     # modulation radius in lambda/D units
            self.pyr_fov = eval(parser.get('2ndChan','pyr_fov'))[sensorId]               # arcsec in diameter
            
            self.excess_noise = np.sqrt(eval(parser.get('2ndChan','excess_noise')))[sensorId] # EMCCD excess noise factor
            
            #--- 2nd channel specific parameters:  NOTE: Leave the blocking mask initialized to zero in order to do initial calibrations!!!
            self.blocking_mask_diam_forcalib = 0.
            self.blocking_mask_diam = eval(parser.get('2ndChan','blocking_mask_diam'))[sensorId] # in lambda/D units where lambda=gs.wavelength (see gs definition below)
            
            #Parameters for later calibration
            self.pyr_thr = eval(parser.get('2ndChan','pyr_thr'))[sensorId]
            self.percent_extra_subaps = eval(parser.get('2ndChan','percent_extra_subaps'))[sensorId]
            
            self.wfs = ceo.Pyramid(self.nLenslet, self.nPx, 
                                         modulation=self.pyr_modulation, 
                                         throughput=self.throughput * self.detQE, 
                                         separation=self.pyr_separation/self.nLenslet,
                                         high_pass_diam=self.blocking_mask_diam_forcalib)
            if self.pyr_modulation == 1.0: self.wfs.modulation_sampling = 16
            
            self.applySignalMasking = eval(parser.get('2ndChan','applySignalMasking'))[sensorId]
            if self.applySignalMasking == 'custom':
                self.path2sigstd = eval(parser.get('2ndChan','path2stdSig'))[sensorId]
                self.maskthr = eval(parser.get('2ndChan','maskThr'))[sensorId]
         
            
        elif self.sensorType.lower()==PHASE_CONTRAST_SENSOR:
            
            try :
                self.nLenslet = eval(parser.get('2ndChan','nLenslet'))[sensorId]
            except NoOptionError:
                self.nLenslet = 1
            self.nPx = eval(parser.get('2ndChan','nPx'))[sensorId]*self.nLenslet
            self.phaseMaskDiam = eval(parser.get('2ndChan','phaseMaskDiameter'))[sensorId]
            self.phaseMaskDelay = eval(parser.get('2ndChan','phaseMaskDelay'))[sensorId]
            self.poppyOversampling = eval(parser.get('2ndChan','poppyOversampling'))[sensorId]
            self.gmtCalib = eval(parser.get('2ndChan','gmtStandardCalibration'))[sensorId]
            self.applyOgc = eval(parser.get('2ndChan', 'applyOgc'))[sensorId]
            self.wfs = ceo.sensors.phaseContrastSensor(self.phaseMaskDiam,
                                                     self.phaseMaskDelay,
                                                     self.cwl,
                                                     self.nPx,
                                                     self.nLenslet,
                                                     poppyOverSampling = self.poppyOversampling)
            self.wfs.fitsPhaseFileName = 'gmtPhase{}.fits'.format(eval(parser.get('general', 'GPUnum')))
            self.wfs.fitsPupilFileName = 'GMTPupil{}.fits'.format(eval(parser.get('general', 'GPUnum')))
            self.excess_noise = eval(parser.get('2ndChan','excess_noise'))[sensorId]
            self.PSstroke = eval(parser.get('2ndChan', 'PSstroke'))[sensorId]
            self.wvl_fraction = eval(parser.get('2ndChan', 'wvl_fraction'))[sensorId]
            
        elif self.sensorType.lower() == LIFT:
            self.excess_noise = eval(parser.get('2ndChan','excess_noise'))[sensorId]
            self.wfs = ceo.sensors.LiftCEO( path, parametersFile,sectionName)
            #dual wavelength setup
            if parser.has_option('2ndChan','dualWL2ndChan'):
                self.dualWL2ndChan = eval(parser.get('2ndChan', 'dualWL2ndChan'))
                self.starcolor = eval(parser.get('2ndChan', 'starcolor'))
                self.aprioriTolerence = eval(parser.get('2ndChan', 'aprioriTolerence'))
                self.amp_conf = eval(parser.get('2ndChan', 'amp_conf'))
            
            self.nPx = self.wfs.gridSize
            # self.cwl = self.wfs.lambdaValue*10**-9
            
        else:
            print('sensor not supported, assuming an ideal piston sensor' )
        
        self.pixelSize = eval(parser.get('telescope', 'D'))/self.nPx        # pixel size [m/pix]
        
        self.exposure_time = eval(parser.get('2ndChan','exposureTime'))
        
        if self.sensorType.lower() in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR,LIFT]:
            self.gs = ceo.Source([self.cwl,self.band,self.e0], magnitude=self.mag, 
                                   zenith=0.,azimuth=0., rays_box_size=eval(parser.get('telescope', 'D')), 
                                   rays_box_sampling=self.nPx, rays_origin=[0.0,0.0,25])
            self.gs.rays.rot_angle = self.pyr_angle*np.pi/180
            if self.sensorType.lower() in [PHASE_CONTRAST_SENSOR,LIFT]:
                self.wfs.fluxPers = self.gs.nPhoton[0] *eval(parser.get('telescope', 'PupilArea')) * \
                                            self.throughput * self.detQE
                self.fluxEst = self.wfs.fluxPers*self.exposure_time
        
            print('Number of simulated NGAO GS photons at the second channel[ph/s/m^2]: %.1f'%(self.gs.nPhoton))
            print('Number of  expected NGAO GS photons at the second channel[ph/s/m^2]: %.1f'%(self.e0*10**(-0.4*self.mag)))
            
        #For telemetry data saving
        self.gs2wfe = []
        self.gs2segPistErr = []
    
        
    def reset(self):
        self.gs.reset()
        self.wfs.reset()
    
    
    def pyr_display_signals_base(self,wavefrontSensorobject, sx=0, sy=0, title=None,fignum = 0, figsize = (15,5)):
        sx2d = wavefrontSensorobject.get_sx2d(this_sx=sx)
        sy2d = wavefrontSensorobject.get_sy2d(this_sy=sy)
        fig, (ax1,ax2) = plt.subplots(num = fignum,ncols=2)
        fig.set_size_inches(figsize)
        if title==None:
            title = ['Sx', 'Sy']
        ax1.set_title(title[0])
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=False, right=False, left=False, labelleft=False)
        imm = ax1.imshow(sx2d, interpolation='None',origin='lower')#,origin='lower', vmin=-1, vmax=1)
        clb = fig.colorbar(imm, ax=ax1, format="%.4f")
        clb.ax.tick_params(labelsize=12)    
        ax2.set_title(title[1])
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=False, right=False, left=False, labelleft=False)
        imm2 = ax2.imshow(sy2d, interpolation='None',origin='lower')#,origin='lower', vmin=-1, vmax=1)
        clb2 = fig.colorbar(imm2, ax=ax2, format="%.4f")  
        clb2.ax.tick_params(labelsize=12)    
        return (sx2d,sy2d)
    
    
    def CalibAndRM(self,telescopeObj, figsize = (15,5)):
        """Initialisation of any of the three sensors considered for the second channel"""
        if self.rescaleAtmResidual:
            
            #link the phase part to this
            self.gs2phase = ceo.tools.ascupy(self.gs.wavefront.phase)
            self.gs.reset()
            telescopeObj.reset()
            telescopeObj.propagate(self.gs)
            self.gs2inpup = ceo.tools.ascupy(self.gs.wavefront.amplitude)
            
        if self.sensorType.lower() == PYRAMID_SENSOR:
            print("calibrating the pyramid wavefrontsensor for the second channel" )
            #identify the flux level.
            telescopeObj.reset()
            self.gs.reset()
            self.wfs.reset()
            telescopeObj.propagate(self.gs)
            self.wfs.propagate(self.gs)
            self.wfs.camera.noiselessReadOut(self.exposure_time)
            # self.wfs.camera.readOut(self.exposure_time,self.background,self.RONval,self.excess_noise)
            fr = self.wfs.camera.frame.host()
            self.fluxEst = np.sum(fr)
            
            
            
            self.gs.reset()
            telescopeObj.reset()
            telescopeObj.propagate(self.gs)
            self.ph_fda_on = self.gs.phase.host(units='nm')
            self.wfs.calibrate(self.gs, percent_extra_subaps=self.percent_extra_subaps
                                     , thr = self.pyr_thr)
            extr = (self.wfs.ccd_frame.shape[0]-240//self.pyr_binning)//2
            
            if self.VISU:
                fig, (ax1,ax2) = plt.subplots(ncols=2)
                fig.set_size_inches(figsize)
                ax1.imshow( np.sum(self.wfs.indpup, axis=0, dtype='bool')[extr:-1-extr, extr:-1-extr] )
            
            if self.VISU:
                imm2 = ax2.imshow(self.wfs.ccd_frame[extr:-1-extr, extr:-1-extr])
                fig.colorbar(imm2, ax=ax2)
            
            self.wfs.high_pass_diam = self.blocking_mask_diam
            #-----> pyramid WFS - M2 segment piston IM calibration
            print("KL0 - NGWS 2nd PYR IM:")
            self.D2 = telescopeObj.calibrate(self.wfs, self.gs, mirror="M2", 
                                         mode="Karhunen-Loeve", stroke=self.PSstroke, 
                                         first_mode=0, last_mode=1)
            if self.VISU:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(figsize)
                imm = ax1.pcolor(self.D2)
                ax1.grid()
                fig.colorbar(imm, ax=ax1)#, fraction=0.012) 
            if self.VISU:
                this_mode = 1  #[from 0 to 6]
            
                sx = self.D2[0:self.wfs.n_sspp,this_mode] / 1e9
                sy = self.D2[self.wfs.n_sspp:,this_mode]  / 1e9
            
                sx2d, sy2d = self.pyr_display_signals_base(self.wfs,sx,sy,figsize = figsize)
            
            self.D2m = self.D2.copy()
            if self.applySignalMasking and not(self.applySignalMasking in ['custom','onlypupil']):
                segpist_signal_mask = telescopeObj.NGWS_segment_piston_mask(self.wfs, self.gs)
                
                self.D2m = telescopeObj.NGWS_apply_segment_piston_mask(self.D2m, segpist_signal_mask['mask'])
                self.wfs.segpist_signal_mask = segpist_signal_mask
                print("Segment piston signal masks applied to channel 2.")
            elif self.applySignalMasking =='custom':
                with fits.open(self.path2sigstd+'signalstd_cwl850nm_seeing0mas_modu{:.0f}_sp{:.0f}.fits'
                               .format(self.pyr_modulation,self.blocking_mask_diam)) \
                                as file:
                    signalStd = file[0].data
                    
                telescopeObj.reset()
                self.gs.reset()
                self.wfs.reset()
                telescopeObj.propagate(self.gs)
                self.wfs.propagate(self.gs)
                sxval = np.abs(self.wfs.get_sx2d())>0
                syval = np.abs(self.wfs.get_sy2d())>0
                validSubApertureMap = sxval +syval
                
                self.mask = []
                vectorMask = []
                for s in range(self.nseg-1):
                    sxthr = signalStd[0,s]>self.maskthr*np.max(signalStd[0,s])
                    sythr = signalStd[1,s]>self.maskthr*np.max(signalStd[1,s])
                    sthr = sxthr+sythr
                    self.mask.append(sthr)
                    vectorMask.append(self.mask[-1][validSubApertureMap])
                #introduce empty array for the central segment
                sthr = np.zeros(sthr.shape,dtype=bool)
                self.mask.append(sthr)
                vectorMask.append(self.mask[-1][validSubApertureMap])
                
                segpist_signal_mask = {'mask': vectorMask,'thr': self.maskthr, 
                                       'stroke': np.linspace(-3*self.chan1wl,3*self.chan1wl,101)}
                self.D2m = telescopeObj.NGWS_apply_segment_piston_mask(self.D2m,segpist_signal_mask['mask'])
                self.wfs.segpist_signal_mask = segpist_signal_mask
            elif self.applySignalMasking == 'onlypupil':
                telescopeObj.reset()
                self.gs.reset()
                self.wfs.reset()
                telescopeObj.propagate(self.gs)
                self.wfs.propagate(self.gs)
                sxval = np.abs(self.wfs.get_sx2d())>0
                syval = np.abs(self.wfs.get_sy2d())>0
                validSubApertureMap = sxval +syval
                
                telescopeObj.reset()
                self.mask = []
                self.gs.reset()
                telescopeObj.propagate(self.gs)
                onlypupil = []
                for s in range(self.nseg-1):
                    # telescopeObj.reset()
                    # self.gs.reset()
                    # telescopeObj.M2.modes.a[s,0] =100*10**-9
                    # telescopeObj.M2.modes.update()
                    # telescopeObj.propagate(self.gs)
                    mask = rebin(self.gs.amplitude.host(),(self.nLenslet,self.nLenslet))>=0.8
                    self.mask.append(mask)
                    # self.mask.append(np.round(self.gs.phase.host()
                    #                           /(100*10**-9)).astype(np.int))
                    onlypupil.append(self.mask[-1][validSubApertureMap])
                mask = np.zeros((mask.shape),dtype=bool)
                self.mask.append(mask)
                onlypupil.append(self.mask[-1][validSubApertureMap])
                self.maskthr = 1
                segpist_signal_mask = {'mask': onlypupil,'thr': self.maskthr, 
                                       'stroke': np.linspace(-3*self.chan1wl,3*self.chan1wl,101)}
                self.D2m = telescopeObj.NGWS_apply_segment_piston_mask(self.D2m,segpist_signal_mask['mask'])
                self.wfs.segpist_signal_mask = segpist_signal_mask
                
                
                
                
                
            
            if self.VISU:
                this_mode = 1  #[from 0 to 6]
            
                sx = self.D2m[0:self.wfs.n_sspp,this_mode] / 1e9
                sy = self.D2m[self.wfs.n_sspp:,this_mode]  / 1e9
            
                sx2d, sy2d = self.pyr_display_signals_base(self.wfs,sx,sy,figsize = figsize)
                
            self.D2m = np.delete(self.D2m,[6],axis = 1)
            self.R2m = np.linalg.pinv(self.D2m)
            self.R2m = np.insert(self.R2m,[6],0,axis=0)
            
            if self.simul_truss_mask:
                telescopeObj.project_truss_onaxis = True
            
            #-- Update slope null vector when using the blocking mask
            self.gs.reset()
            telescopeObj.reset()
            telescopeObj.propagate(self.gs)
            
            #Override slope null vector
            self.wfs.reset()
            self.wfs.set_reference_measurement(self.gs)
            
        if self.sensorType.lower()==PHASE_CONTRAST_SENSOR:
            print('calibrating the phase contrast sensor for the second channel' )
            #This is a prerequisite for poppy to work.
            # telescopeObj.project_truss_onaxis = True
            telescopeObj.reset()
            self.gs.reset()
            self.wfs.reset()
            telescopeObj.propagate(self.gs)
            self.gmtMask = self.wfs.rebin(self.gs.amplitude.host(),[self.nLenslet,self.nLenslet])
            self.wfs.n_sspp = np.sum(self.gmtMask)
            #turn the pupil maks into a format poppy can read
            # fileName = self.wfs.poppyFits(gmtMask,self.wfs.fitsPupilFileName, 'transmision')
            

            telescopeObj.reset()
            self.gs.reset()
            self.wfs.reset()
            telescopeObj.propagate(self.gs)
            #Setting up the reference image for later normalisation
            self.wfs.set_reference_image(self.gs)
            #The gmt calib mode is currently not functionnal.
            if self.gmtCalib:
                # tmpfitsPupFile = self.wfs.fitsPupilFile
                tmpproject_truss_onaxis = telescopeObj.project_truss_onaxis
                telescopeObj.project_truss_onaxis = False
                self.gs.reset()
                telescopeObj.reset()
                telescopeObj.propagate(self.gs)
                
                # self.wfs.poppyFits(self.gs.amplitude.host() \
                #                             ,fileName='noTrussGMTMask.fits'
                #                             , typeOfArray= 'amplitude')
            
            self.interactionMatrix  = np.zeros((self.nseg-1,self.nLenslet**2))
            for s in range(self.nseg-1):
                #set up the wavefront to feed to the propagation
                self.gs.reset()
                telescopeObj.reset()
                self.wfs.reset()
                telescopeObj.M2.modes.a[s,0] = self.PSstroke
                telescopeObj.M2.modes.update()
                telescopeObj.propagate(self.gs)
                
                # feed the wavefront to the propagate method
                self.wfs.propagate(self.gs)
                self.wfs.process(10**-3)
                # self.wfs.frame /= self.wfs.fluxPers *10**-3
                self.interactionMatrix[s] = self.wfs.frame.reshape(-1)/(self.PSstroke)
                # if self.gmtCalib:#if we removed it replace the truss on the images
                    #/!\ this also remove the signal outside the segments!
                self.wfs.frame *= self.gmtMask
            
            self.R2m = np.zeros((self.nLenslet**2,self.nseg))
            self.R2m[:,:6] = np.linalg.pinv(self.interactionMatrix)
            
            if self.gmtCalib:
                # self.wfs.fitsPupilFile = tmpfitsPupFile
                telescopeObj.project_truss_onaxis = tmpproject_truss_onaxis
            
        if self.sensorType.lower()==LIFT:
            telescopeObj.reset()
            self.gs.reset()
            telescopeObj.propagate(self.gs)
            gmtMask = self.gs.amplitude.host()
            self.wfs.mask = self.wfs.CEOcompa(gmtMask)
            
            self.R2m = np.zeros((6,7))
            self.R2m[0,1] = 1
            self.R2m[1,0] = 1
            self.R2m[2,5] = 1
            self.R2m[3,4] = 1
            self.R2m[4,3] = 1
            self.R2m[5,2] = 1
            
            #dual wavelength setup
            if self.dualWL2ndChan and self.starcolor == 'monochromatic':
                converge = np.arange(-5*self.chan1wl,5.1*self.chan1wl,self.chan1wl)
                self.aprio= np.array([[a-self.aprioriTolerence,a+self.aprioriTolerence] 
                                      for a in converge])
            
            
    def propagate(self):
        if self.rescaleAtmResidual:
            self.rescaleFactor = cp.divide(cp.array(self.rescaleTo),
                                           cp.std(self.gs2phase[self.gs2inpup.astype(cp.bool)]))
            self.gs2phase *= self.rescaleFactor
        self.gs2wfe.append(self.gs.wavefront.rms()[0])
        self.gs2segPistErr.append(self.gs.piston(where='segments')[0])
        self.wfs.propagate(self.gs)
    
    
    def OGTL(self,ogc_chan1,cwl1):
        """/!\ This is not an actual Optical Gain Tracking Loop at the moment!!!
        This is merely a lookup table for the retrieval of an Optical Gain for the second channel
        """
        self.ogc_iter.append(self.ogc)
        
        #build from a fit of the points produced by Cedric
        a = -0.593
        b = 1.75
        c = -6.91*10**-7
        see = (ogc_chan1[0]-b-(c/cwl1))/a
        
        self.ogc = (a*see)+b+(c/self.cwl)
        
        
    
    def piston_est4(lambda0,lambda1,lambda2,s1,s2,
                span = [-5000*10**-9,5000*10**-9],
                amp_conf = 0.1, # seuil de confiance
                apriori = None, # tableau 2*n (n le nombre d'intervalles a considere pour 0
                lift = True, #on utilise lift
                zm = False,#on utilise le zernike
                silent = True):
    
        """
        This function is aimed at unwrapping a dual wavelength measurement
        input:
            lambda0, lambda1, lambda2: floats
                The wavelengths of the first channel (lambda0) and the wavelengths at 
                which you are measuring (lambda1 and lambda2) in meters
            
            s1, s2: floats
            The signal produced by a measurement at lambda1 and lambda2. This is unit less.
            
            span: array of 2 floats, optionnal
            The search space in which the piston can be in meters
            
            amp_conf: float, optional
            the confidence threshold for accepting a piston as possible solution. This is unitless
            
            apriori: array of 2xN, optionnal
            a priori contain all the intervals in which, the solution is likely to be.
            This is an array in which apriori[0] contain all the span begining and 
            apriori[1] all the span end. the boundaries must be given in meters
            
            lift : bool, optionnal
            set this option to True is you are using the lift wavefront sensor
            
            zm : bool, optionnal
            set this option to True if you are using the zernike phase contrast sensor.
            
            silent : bool optionnal
            Set this to False if you want the function to tell you it's conclusions live
            
        output:
            the estimate of the piston difference in meters 
        
        Authors : Cedric Plantet, Simonet Esposito, Enrico Pinna
        Translation to python: pyIDL and Anne-Laure Cheffot
        """
    
        lambda0*= 10**9
        lambda1*=10**9
        lambda2*=10**9
        span = [a*10**9 for a in span]
        
        if np.abs(s1) < .05 and np.abs(s2) < .05 :
            if not(silent) : print,'Signals < 0.05, returning 0'
            return 0
            
        if apriori is None:
            apriori = np.array([span.copy()])
        else:
            apriori *= 10**9
        npts = 1e5
        x = np.arange(npts)*(span[1]-span[0])/(npts-1)+span[0]
    
        if not zm  :
            if abs(s1) > 1: s1 = np.sign(s1)
            if abs(s2) > 1: s2 = np.sign(s2)
        else :
            s1 = (s1 < 1) > (-.8)
            s2 = (s2 < 1) > (-.8)
        
    
        mask0 = np.zeros(int(npts))
        for i in range(0, len(apriori)):
            cur_idx = np.where(np.logical_and(x>= apriori[i,0],x<=apriori[i,1]))
            mask0[cur_idx] = 1
        
    
        if not lift :
            signal1 = np.sin(x*2*np.pi/lambda1)
            signal2 = np.sin(x*2*np.pi/lambda2)
            if zm :
                signal1[signal1 < 0] *= 0.8
                signal2[signal2 < 0] *= 0.8
        else :
            signal1 = np.arctan(np.tan(x*np.pi/lambda1))*2./np.pi
            signal2 = np.arctan(np.tan(x*np.pi/lambda2))*2./np.pi
    
        #pistons that give the expected signal within the threshold
        mask1 = np.logical_and(signal1 >= s1-amp_conf, signal1 < s1+amp_conf)
        #
        #Same for 2nd lambda
        mask2 = np.logical_and(signal2 >= s2-amp_conf, signal2 < s2+amp_conf)
            #
        mask = mask1*mask2*mask0
        # plt.plot(x,mask0)
        # plt.plot(x,mask1)
        # plt.plot(x,mask2)
        # plt.plot(x,mask)
        #stop
        #detect intervals & select potential solutions
        idx_est = (mask).nonzero()[0]
        # idx_est = idx_est[0]
        count_est = len(idx_est)
        if count_est != 0 :
            opd_est = np.mean(x[idx_est])
            opd_est_ptv = max(x[idx_est])-min(x[idx_est])
            if opd_est_ptv > lambda0/2. :
                if np.sum(np.sign(x[idx_est]) != np.sign(x[idx_est[0]])) == 0 :
                    if not silent : print('Ambiguous estimation with same sign')
                    return np.sign(x[idx_est[0]])*lambda0
                else :
                   if not silent : print('Ambiguous estimation')
                   return 0
    
                
            return opd_est
        else :
            if not silent : print('No solution detected')
            return 0
    
    def piston_est3(self,lambda0,lambda1,lambda2,s1,s2,
                span = [-5000*10**-9,5000*10**-9],# nm
                amp_conf = 0.1, # seuil de confiance
                apriori = None, # tableau 2*n (n le nombre d'intervalles a considere pour 0
                lift = True, #on utilise lift
                zm = False):#on utilise le zernike
        """
        This function is aimed at unwrapping a dual wavelength measurement
        input:
            lambda0, lambda1, lambda2: floats
                The wavelengths of the first channel (lambda0) and the wavelengths at 
                which you are measuring (lambda1 and lambda2) in meters
            
            s1, s2: floats
            The signal produced by a measurement at lambda1 and lambda2. This is unit less.
            
            span: array of 2 floats, optionnal
            The search space in which the piston can be in meters
            
            amp_conf: float, optional
            the confidence threshold for accepting a piston as possible solution. This is unitless
            
            apriori: array of 2xN, optionnal
            a priori contain all the intervals in which, the solution is likely to be.
            This is an array in which apriori[0] contain all the span begining and 
            apriori[1] all the span end. the boundaries must be given in meters
            
            lift : bool, optionnal
            set this option to True is you are using the lift wavefront sensor
            
            zm : bool, optionnal
            set this option to True if you are using the zernike phase contrast sensor.
            
        output:
            the estimate of the piston difference in meters 
        
        Authors : Cedric Plantet, Simonet Esposito, Enrico Pinna
        Translation to python: pyIDL and Anne-Laure Cheffot
        """
        #for compatibility with CEO
        #CEO uses only meters for wavelength this function uses nm
        #Hance converting the input from meters to nanometers
        lambda0*= 10**9
        lambda1*=10**9
        lambda2*=10**9
        span = [a*10**9 for a in span]
        if apriori is None:
            apriori = np.array([span.copy()])
        else:
            apriori *= 10**9
        
    
        if np.abs(s1) < .05 and np.abs(s2) < .05 :
            # print('Signals < 0.05, returning 0')
            return 0
        
        
        npts = 1e5
        x = np.arange(npts)*(span[1]-span[0])/(npts-1)+span[0]
        
        if not zm :
            if abs(s1) > 1: s1 = np.sign(s1)
            if abs(s2) > 1: s2 = np.sign(s2)
        else:#TODO this case does not work in python. maybe one day
            s1 = (s1 < 1) > (-.8)
            s2 = (s2 < 1) > (-.8)
        
        
        mask0 = np.zeros(int(npts))
        for i in range(0, len(apriori)):
            cur_idx = np.where(np.logical_and(x>= apriori[i,0],x<=apriori[i,1]))
            # cur_idx = (x >= apriori[0]).nonzero()
            # cur_idx = cur_idx[0]
            # i] & x <= apriori[1,i] = len(cur_idx)
            mask0[cur_idx] = 1
        
        
        if not lift :
            signal1 = np.sin(x*2*np.pi/lambda1)
            signal2 = np.sin(x*2*np.pi/lambda2)
            if zm :
                signal1[signal1 < 0] *= 0.8
                signal2[signal2 < 0] *= 0.8
        else:
            signal1 = np.arctan(np.tan(x*np.pi/lambda1))*2./np.pi
            signal2 = np.arctan(np.tan(x*np.pi/lambda2))*2./np.pi
        
        
        #pistons that give the expected signal within the threshold
        mask1 = np.logical_and(signal1 >= s1-amp_conf, signal1 < s1+amp_conf)
        #
        #Same for 2nd lambda
        mask2 = np.logical_and(signal2 >= s2-amp_conf, signal2 < s2+amp_conf)
        #
        
        mask = mask1*mask2*mask0
        #stop
        #detect intervals & select potential solutions
        idx_est = mask.nonzero()[0]#
        # idx_est = idx_est[0]
        count_est = len(idx_est)
        if count_est != 0 :
            opd_est = np.mean(x[idx_est])
            opd_est_ptv = np.max(x[idx_est])-np.min(x[idx_est])
            if opd_est_ptv > lambda0/2. :
                # print,'Ambiguous estimation'
                i_amb = 1
                while opd_est_ptv > lambda0/2. :
                    mask_shift = mask*np.roll(mask,i_amb)
                    idx_est_amb = (mask_shift).nonzero()	#
                    idx_est_amb = idx_est_amb[0]
                    count_shift = len(idx_est_amb)
                    if count_shift != 0 :
                        opd_est_amb = np.mean(x[idx_est_amb])
                        opd_est_ptv = np.max(x[idx_est_amb])-np.min(x[idx_est_amb])
                            
                    i_amb += 1
                
                opd_est = opd_est_amb
                
            return opd_est*10**-9
        else:
            # print,'No solution detected'
            mask01 = mask0*mask1
            mask02 = mask0*mask2
            idx_est01 = (mask01).nonzero()	#
            idx_est01 = idx_est01[0]
            count_est01 = len(idx_est01)
            idx_est02 = (mask02).nonzero()	#
            idx_est02 = idx_est02[0]
            count_est02 = len(idx_est02)
          
            if count_est01 > 0 :
                opd_est01 = np.mean(x[idx_est01])
                opd_est_ptv01 = np.max(x[idx_est01])-np.min(x[idx_est01])
            else :
                opd_est01 = 0
                opd_est_ptv01 = 2*lambda0
                
            if count_est02 > 0 :
                opd_est02 = np.mean(x[idx_est02])
                opd_est_ptv02 = np.max(x[idx_est02])-np.min(x[idx_est02])
            else :
                opd_est02 = 0
                opd_est_ptv02 = 2*lambda0
            
        
            if opd_est_ptv01 < lambda0/2.: 
                opd_est0102 = opd_est01 
            else :
                if opd_est_ptv02 < lambda0/2.: 
                    opd_est0102 = opd_est02 
                else :
                    opd_est0102 = np.mean([opd_est01,opd_est02])
            
            #return,0
            return opd_est0102*10**-9
    
    
    def pistonRetrival(self,sensor_measurement):
        """use the measurement verctor(s) out of any sensor to produce a command vector for the closed loop
        input:
            sensor_measurement: a list of single dimension array or a single dimention array
            any arrays should have self.nseg elements
            
        output:
            command_vector: a single dimension array of self.nseg elements"""
        
        command_vector = np.array([np.sign(a) if np.abs(a) > self.detthr else 0 
                                   for a in self.piston_estimate])*self.chan1wl*0.9
        
        return command_vector
            
    def process(self, dbg = False,figsize = (15,5)):
        if self.sensorType.lower() == PYRAMID_SENSOR:
            # self.wfs.camera.noiselessReadOut(self.exposure_time)
            self.wfs.camera.readOut(self.exposure_time, 
                                          self.RONval,
                                          self.background,
                                          self.excess_noise)
            self.wfs.process()
            meas = self.wfs.get_measurement()
            self.piston_estimate = self.R2m @ meas
            self.piston_estimate -= self.piston_estimate[6]
            
            if self.applyOgc:
                self.piston_estimate *= self.ogc
            
            self.forCorrection = self.pistonRetrival(self.piston_estimate)

            if dbg:
                plt.figure()
                plt.imshow(self.gs.phase.host(),origin = 'lower')

            self.wfs.reset()
        if self.sensorType==PHASE_CONTRAST_SENSOR:
            
            self.wfs.cameraNoise(self.RONval,0,self.excess_noise)
            # estflux = np.sum(self.wfs.frame*self.gmtMask)
            self.wfs.process(self.exposure_time)
            self.piston_estimate = self.wfs.frame.reshape(-1) @ self.R2m
            self.piston_estimate -= self.piston_estimate[6]
            if self.applyOgc:
                self.piston_estimate *= self.ogc
            self.forCorrection = self.pistonRetrival(self.piston_estimate)
            
            if dbg:
                plt.figure()
                plt.imshow(self.wfs.frame)
                
                self.debugframe.append(self.wfs.frame)
            self.wfs.reset()
        
        if self.sensorType==LIFT:
            
            self.wfs.cameraNoise(self.RONval,self.background,self.excess_noise)
            self.fluxEst = np.sum(self.wfs.frame)
            currentPhaseEstimate, A_ML = self.wfs.phaseEstimation(self.wfs.frame, False, 1e-6, 1e-5)
            
            self.piston_estimate = A_ML[:6] @ self.R2m
            # self.piston_estimate *= self.gs.wavelength/(2*np.pi)
            self.piston_estimate = np.arctan(np.tan(self.piston_estimate/2))*2/np.pi
            # self.forCorrection = self.pistonRetrival(self.piston_estimate)
            
        if not(self.active_corr):
            self.forCorrection *= 0

if __name__ == '__main__':
    inputFolder = '/home/alcheffot/CEO/'
    parametersFileName = 'paramsLIFT1250-1600'
    
    c2 = Chan2(inputFolder,parametersFileName)
    c2.chan1wl = 715*10**-9
    M2_n_modes = 675 # Karhunen-Loeve per M2 segment
    #M2_modes_set = u"ASM_DDKLs_S7OC04184_675kls"
    M2_modes_set = u"ASM_fittedKLs_doubleDiag"
    gmt = ceo.GMT_MX(M2_mirror_modes=M2_modes_set, M2_N_MODE=M2_n_modes)
    
    #---- Telescope parameters
    D = 25.5                # [m] Diameter of simulated square (slightly larger than GMT diameter) 
    PupilArea = 357.0       # [m^2] Takes into account baffled segment borders
    tel_throughput = 0.9**4 # M1 + M2 + M3 + GMTIFS dichroic = 0.9^4 
    gmt.M2_baffle = 3.5    # Circular diameter pupil mask obscuration
    gmt.project_truss_onaxis = False
    simul_truss_mask=False  # If True, it applies to closed loop simulation, and slope-null calibration only (NOT IM calibration)
    
    # c2.CalibAndRM(gmt)
    
    # # segpist_signal_mask = gmt.NGWS_segment_piston_mask(c2.wfs, c2.gs)
    # # c2.D2m = c2.D2.copy()
    # # c2.D2m = gmt.NGWS_apply_segment_piston_mask(c2.D2m, segpist_signal_mask['mask'])
    # # c2.wfs.segpist_signal_mask = segpist_signal_mask
    
    # # gmt.reset()
    # # c2.gs.reset()
    # # c2.wfs.reset()
    # # gmt.propagate(c2.gs)
    # # c2.wfs.propagate(c2.gs)
    
    # # validSubApertureMap = np.abs(c2.wfs.get_sx2d())>0
    # # vectorMask = c2.mask[0][validSubApertureMap]
    
    # #did the masking work
    # seg = 2
    # # toto,tata = c2.pyr_display_signals_base(c2.wfs,c2.D2m[:c2.wfs.n_sspp,seg],
    # #                                         c2.D2m[c2.wfs.n_sspp:,seg],
    # #                                         title = 'interaction matrix')
    # # c2.pyr_display_signals_base(c2.wfs,c2.R2m[seg,:c2.wfs.n_sspp],
    # #                             c2.R2m[seg,c2.wfs.n_sspp:],
    # #                             title = 'reconstruction matrix' )
    
    # gmt.reset()
    # c2.gs.reset()
    # c2.wfs.reset()
    # gmt.M2.modes.a[seg,0] =60*10**-9
    # gmt.M2.modes.update()
    # gmt.propagate(c2.gs)
    # c2.wfs.propagate(c2.gs)
    # # c2.wfs.frame*=150
    # # c2.wfs.cameraNoise(0,0,1)
    # c2.wfs.process(0.001)
    # # c2.process()
    # plt.figure(1)
    # plt.clf()
    # # plt.imshow(c2.wfs.frame-c2.wfs.referenceFrame)
    # plt.imshow(c2.wfs.frame)
    # # c2.pyr_display_signals_base(c2.wfs,*c2.wfs.get_measurement(out_format='list' ),
    # #                             title = 'signal')
    
    # #sweep test
    # # piston = np.linspace(-5*715,5*(715+1),201)*10**-9
    # # wlmult = np.linspace(-5*715,5*(715+1),11)*10**-9
    # piston = np.linspace(-715,715,21)*10**-9
    # seg = 1

    # recall = []
    # recmwl = []
    # for p in piston:
    #     c2.gs.reset()
    #     gmt.reset()
    #     gmt.M2.modes.a[seg,0] =p
    #     gmt.M2.modes.update()
    #     c2.wfs.reset()
    #     gmt.propagate(c2.gs)
    #     c2.wfs.propagate(c2.gs)
    #     c2.wfs.frame*=150
    #     c2.process()
    #     # measvec = c2.wfs.get_measurement()
    #     # rec = c2.R2m @ measvec
    #     rec = c2.piston_estimate
    #     recall.append(rec[seg])
    #     # if p in wlmult:
    #     #     recmwl.append(rec[seg])
    
    # plt.figure(2)
    # plt.clf()
    # plt.plot(piston,recall)
    # plt.plot(wlmult,recmwl,'o' )
    
    
    





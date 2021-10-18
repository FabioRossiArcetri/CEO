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
from configparser import ConfigParser
from datetime import datetime

PYRAMID_SENSOR = 'pyramid'
PHASE_CONTRAST_SENSOR = 'phasecontrast' 
LIFT = 'lift'


class Chan2(object):
    """object meant to wrap everything the second channel may do"""
    
    def __init__(self,path,parametersFile):
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
        
        self.VISU = eval(parser.get('general', 'VISU'))
        self.simul_truss_mask = eval(parser.get('telescope', 'simul_truss_mask'))
        self.nseg = eval(parser.get('telescope','nseg'))
        self.active_corr = eval(parser.get('2ndChan', 'active_corr'))
        
        
        #addition of parameters for the second channel: first the second channel source parameters
        self.cwl = eval(parser.get('2ndChan', 'wavelength2'))
        self.band = eval(parser.get('2ndChan', 'delta_wavelength2'))
        self.e0_inband = eval(parser.get('2ndChan', 'e0_inClosestBand'))
        self.band_inband = eval(parser.get('2ndChan', 'deltaWavelength_ClosestBand'))
        self.mag = eval(parser.get('2ndChan', 'mag'))
        self.e0 = self.e0_inband * (self.band/self.band_inband) / eval(parser.get('telescope', 'PupilArea'))    #in ph/m^2/s in the desired bandwidth
        self.throughput =  eval(parser.get('2ndChan','throughput'))  # NGWS board: 0.4
        self.detQE = eval(parser.get('2ndChan','detectorQE'))
        #setup of the parameters for the sensor itself.
        self.sensorType = eval(parser.get('2ndChan', 'sensorType'))
        self.RONval = eval(parser.get('2ndChan','RONval'))                # e- RMS
        self.pyr_angle = eval(parser.get('2ndChan','pyr_angle'))         # angle between pyramid facets and GMT pupil (in deg)
        self.detthr = eval(parser.get('2ndChan','detthr'))
        
        # initialisation of the second channel wavefront sensor
        if self.sensorType.lower() == 'idealpistonsensor':
            self.pyr_binning = eval(parser.get('2ndChan', 'pyr_binning'))
            self.nLenslet = eval(parser.get('2ndChan','nLenslet'))//self.pyr_binning            # sub-apertures across the pupil
            self.nPx = self.nLenslet*eval(parser.get('2ndChan','nPx'))
            print('initialised elsewhere in the code')
            
        if self.sensorType.lower() == PYRAMID_SENSOR:
            self.pyr_binning = eval(parser.get('2ndChan', 'pyr_binning'))
            self.nLenslet = eval(parser.get('2ndChan','nLenslet'))//self.pyr_binning            # sub-apertures across the pupil
            self.nPx = self.nLenslet*eval(parser.get('2ndChan','nPx'))
            self.applyOgc = eval(parser.get('2ndChan', 'applyOgc'))
            self.pyr_separation = eval(parser.get('2ndChan','pyr_separation'))//self.pyr_binning     # separation between centers of adjacent sub-pupil images on the detector [pix]
            self.pyr_modulation = eval(parser.get('2ndChan','pyr_modulation'))     # modulation radius in lambda/D units
            self.pyr_fov = eval(parser.get('2ndChan','pyr_fov'))               # arcsec in diameter
            
            self.excess_noise = np.sqrt(eval(parser.get('2ndChan','excess_noise'))) # EMCCD excess noise factor
            
            #--- 2nd channel specific parameters:  NOTE: Leave the blocking mask initialized to zero in order to do initial calibrations!!!
            self.blocking_mask_diam_forcalib = 0.
            self.blocking_mask_diam = eval(parser.get('2ndChan','blocking_mask_diam')) # in lambda/D units where lambda=gs.wavelength (see gs definition below)
            
            #Parameters for later calibration
            self.pyr_thr = eval(parser.get('2ndChan','pyr_thr'))
            self.percent_extra_subaps = eval(parser.get('2ndChan','percent_extra_subaps'))
            
            self.wfs = ceo.Pyramid(self.nLenslet, self.nPx, 
                                         modulation=self.pyr_modulation, 
                                         throughput=self.throughput * self.detQE, 
                                         separation=self.pyr_separation/self.nLenslet,
                                         high_pass_diam=self.blocking_mask_diam_forcalib)
            if self.pyr_modulation == 1.0: self.wfs.modulation_sampling = 16
            
            self.applySignalMasking = eval(parser.get('2ndChan','applySignalMasking'))
            if self.applySignalMasking == 'custom':
                self.path2sigstd = eval(parser.get('2ndChan','path2stdSig'))
                self.maskthr = eval(parser.get('2ndChan','maskThr'))
                
            
        elif self.sensorType.lower()==PHASE_CONTRAST_SENSOR:
            self.nPx = eval(parser.get('2ndChan','nPx'))
            self.phaseMaskDiam = eval(parser.get('2ndChan','phaseMaskDiameter'))
            self.phaseMaskDelay = eval(parser.get('2ndChan','phaseMaskDelay'))
            self.poppyOversampling = eval(parser.get('2ndChan','poppyOversampling'))
            self.gmtCalib = eval(parser.get('2ndChan','gmtStandardCalibration'))
            self.applyOgc = eval(parser.get('2ndChan', 'applyOgc'))
            self.wfs = ceo.sensors.phaseContrastSensor(self.phaseMaskDiam,
                                                     self.phaseMaskDelay,
                                                     self.cwl,
                                                     self.nPx,
                                                     poppyOverSampling = self.poppyOversampling)
            self.wfs.fitsPhaseFileName = 'gmtPhase{}.fits'.format(eval(parser.get('general', 'GPUnum')))
            self.wfs.fitsPupilFileName = 'GMTPupil{}.fits'.format(eval(parser.get('general', 'GPUnum')))
            self.excess_noise = eval(parser.get('2ndChan','excess_noise'))
            
        elif self.sensorType.lower() == LIFT:
            self.excess_noise = eval(parser.get('2ndChan','excess_noise'))
            self.wfs = ceo.sensors.LiftCEO( path, parametersFile)
            
            self.nPx = self.wfs.gridSize
            self.cwl = self.wfs.lambdaValue*10**-9
            
        else:
            print('sensor not supported, assuming an ideal piston sensor' )
        
        self.pixelSize = eval(parser.get('telescope', 'D'))/self.nPx        # pixel size [m/pix]
        self.PSstroke = eval(parser.get('2ndChan', 'PSstroke'))
        self.wvl_fraction = eval(parser.get('2ndChan', 'wvl_fraction'))
        self.exposure_time = eval(parser.get('2ndChan','exposureTime'))
        
        if self.sensorType.lower() in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR,LIFT]:
            self.gs = ceo.Source([self.cwl,self.band,self.e0], magnitude=self.mag, 
                                   zenith=0.,azimuth=0., rays_box_size=eval(parser.get('telescope', 'D')), 
                                   rays_box_sampling=self.nPx, rays_origin=[0.0,0.0,25])
            self.gs.rays.rot_angle = self.pyr_angle*np.pi/180
            if self.sensorType.lower() in [PHASE_CONTRAST_SENSOR,LIFT]:
                self.wfs.fluxPers = self.gs.nPhoton[0] *eval(parser.get('telescope', 'PupilArea')) * \
                                            self.throughput * self.detQE
        
            print('Number of simulated NGAO GS photons at the second channel[ph/s/m^2]: %.1f'%(self.gs.nPhoton))
            print('Number of  expected NGAO GS photons at the second channel[ph/s/m^2]: %.1f'%(self.e0*10**(-0.4*self.mag)))
            
    def pyr_display_signals_base(self,wavefrontSensorobject, sx=0, sy=0, title=None,figsize = (15,5)):
        sx2d = wavefrontSensorobject.get_sx2d(this_sx=sx)
        sy2d = wavefrontSensorobject.get_sy2d(this_sy=sy)
        fig, (ax1,ax2) = plt.subplots(ncols=2)
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
        if self.sensorType.lower() == PYRAMID_SENSOR:
            print("calibrating the pyramid wavefrontsensor for the second channel" )
            #identify the flux level.
            telescopeObj.reset()
            self.gs.reset()
            self.wfs.reset()
            telescopeObj.propagate(self.gs)
            self.wfs.propagate(self.gs)
            self.wfs.camera.readOut(0.001,0,0,1)
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
            if self.applySignalMasking and (self.applySignalMasking!='custom'):
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
            gmtMask = self.gs.amplitude.host()
            self.wfs.nBPixelInPupil = np.sum(gmtMask)
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
            
            self.interactionMatrix  = np.zeros((self.nseg-1,self.nPx**2))
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
                self.interactionMatrix[s] = self.wfs.frame.reshape(-1)/(self.PSstroke)
                if self.gmtCalib:#if we removed it replace the truss on the images
                    #/!\ this also remove the signal outside the segments!
                    self.wfs.frame *= gmtMask
            
            self.R2m = np.zeros((self.nPx**2,self.nseg))
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
            # self.wfs.R2m *= 10**-9
    
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
            
            self.wfs.camera.readOut(self.exposure_time, 
                                          self.RONval,
                                          0,
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
            
            self.wfs.cameraNoise(self.RONval,0,self.excess_noise)
            currentPhaseEstimate, A_ML = self.wfs.phaseEstimation(self.wfs.frame, False, 1e-6, 1e-5)
            
            self.piston_estimate = A_ML[:6] @ self.R2m
            self.piston_estimate *= self.gs.wavelength/(2*np.pi)
            self.forCorrection = self.pistonRetrival(self.piston_estimate)
        
        if not(self.active_corr):
            self.forCorrection *= 0

if __name__ == '__main__':
    inputFolder = '/home/alcheffot/CEO/'
    parametersFileName = 'paramsZPC1250'
    
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
    
    c2.CalibAndRM(gmt)
    
    # segpist_signal_mask = gmt.NGWS_segment_piston_mask(c2.wfs, c2.gs)
    # c2.D2m = c2.D2.copy()
    # c2.D2m = gmt.NGWS_apply_segment_piston_mask(c2.D2m, segpist_signal_mask['mask'])
    # c2.wfs.segpist_signal_mask = segpist_signal_mask
    
    # gmt.reset()
    # c2.gs.reset()
    # c2.wfs.reset()
    # gmt.propagate(c2.gs)
    # c2.wfs.propagate(c2.gs)
    
    # validSubApertureMap = np.abs(c2.wfs.get_sx2d())>0
    # vectorMask = c2.mask[0][validSubApertureMap]
    
    #did the masking work
    seg = 2
    # toto,tata = c2.pyr_display_signals_base(c2.wfs,c2.D2m[:c2.wfs.n_sspp,seg],
    #                                         c2.D2m[c2.wfs.n_sspp:,seg],
    #                                         title = 'interaction matrix')
    # c2.pyr_display_signals_base(c2.wfs,c2.R2m[seg,:c2.wfs.n_sspp],
    #                             c2.R2m[seg,c2.wfs.n_sspp:],
    #                             title = 'reconstruction matrix' )
    
    gmt.reset()
    c2.gs.reset()
    c2.wfs.reset()
    gmt.M2.modes.a[seg,0] =60*10**-9
    gmt.M2.modes.update()
    gmt.propagate(c2.gs)
    c2.wfs.propagate(c2.gs)
    c2.wfs.frame*=150
    c2.wfs.cameraNoise(1,0,1.3)
    c2.wfs.process(0.001)
    # c2.process()
    plt.figure(1)
    plt.clf()
    # plt.imshow(c2.wfs.frame-c2.wfs.referenceFrame)
    plt.imshow(c2.wfs.frame)
    # c2.pyr_display_signals_base(c2.wfs,*c2.wfs.get_measurement(out_format='list' ),
    #                             title = 'signal')
    
    #sweep test
    # piston = np.linspace(-5*715,5*(715+1),201)*10**-9
    # wlmult = np.linspace(-5*715,5*(715+1),11)*10**-9
    piston = np.linspace(-60,61,121)*10**-9
    seg = 1

    recall = []
    recmwl = []
    for p in piston:
        c2.gs.reset()
        gmt.reset()
        gmt.M2.modes.a[seg,0] =p
        gmt.M2.modes.update()
        c2.wfs.reset()
        gmt.propagate(c2.gs)
        c2.wfs.propagate(c2.gs)
        c2.wfs.frame*=150
        c2.process()
        # measvec = c2.wfs.get_measurement()
        # rec = c2.R2m @ measvec
        rec = c2.piston_estimate
        recall.append(rec[seg])
        # if p in wlmult:
        #     recmwl.append(rec[seg])
    
    plt.figure(2)
    plt.clf()
    plt.plot(piston,recall)
    # plt.plot(wlmult,recmwl,'o' )
    
    
    





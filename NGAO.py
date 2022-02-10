# 1 CLT
# 2 save simulations as TN in the raid1 folder for now subfolder perfectSensor
# 3 load back simu results and run analisys
# 4 comparison between simulation instances (parameter sets)
# 5 check if a parameter set was simulated already

#----- CEO: GMT ray-tracing and wavefront sensing simulator
import ceo

#----- System related packages
import sys
import datetime as dt
import os.path
import pickle
import copy
import shutil

#----- Math and Scientific Computing
import math
import numpy as np
import cupy as cp
from scipy import ndimage
from scipy import signal 
from scipy.interpolate import CubicSpline
import poppy

#----- Visualization
import matplotlib.pyplot as plt
from astropy.io import fits

# .ini file parsing
from configparser import ConfigParser,NoOptionError
from datetime import datetime

from chan2 import * 

#for the phase contrast sensor and lift
# tmppath = os.getcwd()
# os.chdir('/home/alcheffot/CEO/python/ceo/sensors/')
# import phaseContrastSensor as zpc
# import LiftCEO
# os.chdir(tmppath)


 
PYRAMID_SENSOR = 'pyramid'
PHASE_CONTRAST_SENSOR = 'phasecontrast' 
LIFT = 'lift'

def zern_num(_num_):
    n = np.floor(np.sqrt(8*_num_-7)-1) // 2
    m = np.where( (n%2)==1, 1+2*((_num_-1-(n*(n+1))//2)//2),
                              2*((_num_  -(n*(n+1))//2)//2))
    return n.astype('int'), m.astype('int')

def poly_nomial(x, coefs):
    deg = len(coefs)-1
    y = np.zeros(x.size)
    for k in range(deg+1):
        y += coefs[k] * x**(deg-k)
    return y

#------ Functions below required for OMGI
from scipy import optimize

def rejectTF(gi, nu, Te, tau, ffi):
    return 1 / ( 1 + olTF(gi,nu,Te,tau,ffi))

#-- Detector or Sample-and-Hold TF:
def Itf(nu, Te):
    red = 1j*nu*(2*np.pi*Te)
    return (1 - np.exp(-red)) / red

#-- Pure delay TF: 
def Dtf(nu, tau):
    return np.exp(-1j*nu*(2*np.pi*tau))


#-- Leaky integrator controller:
def Ctf(gi, nu, Te, ffi):
    return gi / (1 - ffi*np.exp(-1j*nu*(2*np.pi*Te)))

#-- ASM response:
def ASMtf(nu):
    #fz = 800.;
    #dz = 0.75;
    #s = complex(0, 2*pi*nu);
    #(2*pi*fz)^2 ./ ( s.^2 + (4*pi*dz*fz).*s + (2*pi*fz)^2 );
    return 1

#-- Open-loop TF:
def olTF(gi, nu, Te, tau, ffi):
    return Itf(nu, Te)**2 * Dtf(nu, tau) * Ctf(gi, nu, Te, ffi);
    #return ASMtf(nu) .* Itf(nu, Te).^2 .* Dtf(nu,tau) .* Ctf(gi, nu, Te);
    


class NGAO(object):
    def __init__(self, path, parametersFile,load = False,singleSim = False):
        self.tnString = datetime.today().strftime('%Y%m%d_%H%M%S')
        self.path = path
        parser = ConfigParser()
        parser.read(path +'/'+ parametersFile + '.ini')        
        print(path +'/'+ parametersFile + '.ini')
        self.GPUnum = eval(parser.get('general', 'GPUnum'))
        
        if singleSim:
            self.tempFolder = path+'/savedir{:.0f}/'.format(self.GPUnum)+self.tnString
            os.makedirs(self.tempFolder,exist_ok=True)
            shutil.copy(path+'/'+parametersFile+'.ini', self.tempFolder)
        elif not(load):
            self.tempFolder = path+'/savedir{:.0f}/'.format(self.GPUnum)+self.tnString
            os.makedirs(self.tempFolder,exist_ok=True)
            
            shutil.move(path+'/'+parametersFile+'.ini', self.tempFolder)
            

        # ini_temp_filename='./savedir'+str(self.GPUnum)+'/'+parametersFile+'.ini'
        # os.system('cp '+ path+parametersFile+'.ini ' + ini_temp_filename)
        
        cp.cuda.Device(self.GPUnum).use()
        self.dir_calib = eval(parser.get('general', 'dir_calib'))
        self.atm_dir = eval(parser.get('general', 'atm_dir'))
        self.TN_dir = eval(parser.get('general', 'TN_dir'))
        self.simul_onaxis_AO = eval(parser.get('general', 'simul_onaxis_AO'))
        self.analyse_island_effect = eval(parser.get('general','analyse_island_effect'))
        # Karhunen-Loeve per M2 segment
        self.M2_n_modes = eval(parser.get('general', 'M2_n_modes'))
        if self.simul_onaxis_AO:
            self.M2_modes_set = u"ASM_fittedKLs_doubleDiag"
        else:
            self.M2_modes_set = ""        
            self.M2_n_modes = 0            
        self.simul_turb = eval(parser.get('general', 'simul_turb'))
        self.simul_windload = eval(parser.get('general', 'simul_windload'))
        self.simul_M1polish = eval(parser.get('general', 'simul_M1polish'))
        self.simul_wfs_noise = eval(parser.get('general', 'simul_wfs_noise'))        
        self.totSimulTime = eval(parser.get('general', 'totSimulTime'))
        self.Tsim = eval(parser.get('general', 'Tsim'))
        #Select M1 map
        if self.simul_M1polish:
            # self.M1_map = u'M1_print_through'     # Print-through errors.
            self.M1_map = u'M1_meeting_proposed_spec'  # M1 polishing errors.
        else:
            self.M1_map = ''            
        self.do_psf_le = eval(parser.get('general', 'do_psf_le'))
        self.coro_psf = eval(parser.get('general', 'coro_psf'))
        self.do_wo_coro_too = eval(parser.get('general', 'do_wo_coro_too')) # If True, compute both w/coro and w/o coro in the same simulation
        self.ogtl_simul = eval(parser.get('general', 'ogtl_simul'))
        self.eval_perf_modal = eval(parser.get('general', 'eval_perf_modal'))
        self.eval_perf_modal_turb = eval(parser.get('general', 'eval_perf_modal')) and self.simul_turb
        self.VISU = eval(parser.get('general', 'VISU'))
        self.do_Phase_integration = eval(parser.get('general', 'do_Phase_integration'))
        self.lim = eval(parser.get('general', 'lim'))
        self.npad = eval(parser.get('general', 'npad'))
        self.sep_lD = eval(parser.get('general', 'sep_lD'))        
        self.seg_pist_scramble = eval(parser.get('general', 'seg_pist_scramble'))
        if self.seg_pist_scramble:
            self.pist_scramble_rms = eval(parser.get('general', 'pist_scramble_rms'))
            if parser.has_option('general','scrambleSeed'):
                self.scramble_seed = eval(parser.get('general', 'scrambleSeed'))
            else:
                self.scramble_seed = 654321
        self.M2_modes_scramble = eval(parser.get('general', 'M2_modes_scramble'))
        self.save_telemetry = eval(parser.get('general', 'save_telemetry'))        
        self.do_psf_le = eval(parser.get('general', 'do_psf_le'))        
        self.do_crazy_spp = eval(parser.get('general', 'do_crazy_spp'))        
        self.fake_fast_spp_convergence = eval(parser.get('general', 'fake_fast_spp_convergence'))
        self.save_phres_decimated = eval(parser.get('general', 'save_phres_decimated'))
        self.save_psfse_decimated = eval(parser.get('general', 'save_psfse_decimated'))        
        self.AOinitTime = eval(parser.get('general', 'AOinit'))
        self.SPPctrlInitTime = eval(parser.get('general', 'SPPctrlInit'))
        self.SPP2ndChInitTime = eval(parser.get('general', 'SPP2ndChInit'))
        
        self.forcePhased = eval(parser.get('general', 'forcePhase'))
        if self.forcePhased:
            self.forceseg = eval(parser.get('general', 'forceSeg'))
            self.forceto = eval(parser.get('general', 'forceto'))
        try:
            self.tot_delay = eval(parser.get('general', 'tot_delay'))
        except NoOptionError:
            self.tot_delay = 2

        self.sep_req = self.sep_lD * self.lim/ 24.5 * ceo.constants.RAD2MAS  # in mas
        self.knumber = 2.*cp.pi/self.lim

        self.tid = ceo.StopWatch()  # Keep the time
        
        if self.simul_M1polish and self.simul_onaxis_AO:
            self.gmt = ceo.GMT_MX(M1_mirror_modes=self.M1_map, M1_N_MODE=1,
                             M2_mirror_modes=self.M2_modes_set, M2_N_MODE=self.M2_n_modes)
        elif not self.simul_M1polish and self.simul_onaxis_AO:
            self.gmt = ceo.GMT_MX(M2_mirror_modes=self.M2_modes_set, M2_N_MODE=self.M2_n_modes)

        elif self.simul_M1polish and not self.simul_onaxis_AO:
            self.gmt = ceo.GMT_MX(M1_mirror_modes=self.M1_map, M1_N_MODE=1)
        else:
            self.gmt = ceo.GMT_MX()
                        
        if self.simul_onaxis_AO:
            cmdArg = "\{\" + '\"~\/CEO\/gmtMirrors\/\"" + self.M2_modes_set + ".ceo" + "\}"
            M2fn = os.system("!readlink -f " + cmdArg )
            print('M2 modes loaded from:' +  str(M2fn))

        self.D = eval(parser.get('telescope', 'D'))
        
        self.fp_pixscale = self.lim/(self.npad*self.D)*ceo.constants.RAD2MAS  #mas/pixel in focal plane

        self.segmentD = eval(parser.get('telescope', 'segmentD'))
        self.PupilArea = eval(parser.get('telescope', 'PupilArea'))
        self.tel_throughput = eval(parser.get('telescope', 'tel_throughput'))
        self.gmt.M2_baffle = eval(parser.get('telescope', 'M2_baffle'))
        self.gmt.project_truss_onaxis = eval(parser.get('telescope', 'project_truss_onaxis'))
        self.simul_truss_mask = eval(parser.get('telescope', 'simul_truss_mask'))
        self.nseg = eval(parser.get('telescope','nseg'))
        self.Roc = self.gmt.M2_baffle / self.segmentD #TODOFR
        if self.simul_onaxis_AO:

            self.band = eval(parser.get('pyramid1stChan', 'band'))
            self.mag = eval(parser.get('pyramid1stChan', 'mag'))        
            # ph/s in R+I band over the GMT pupil (correction factor applied)
            self.e0 = 9.00e12 /368. * self.PupilArea
            self.bkgd_mag = eval(parser.get('pyramid1stChan', 'bkgd_mag'))
            self.nLenslet = eval(parser.get('pyramid1stChan', 'nLenslet'))
            self.nPx = eval(parser.get('pyramid1stChan', 'nPx_fact')) * self.nLenslet
            self.pixelSize = self.D/self.nPx
            self.pyr_separation = eval(parser.get('pyramid1stChan', 'pyr_separation'))
            self.pyr_modulation = eval(parser.get('pyramid1stChan', 'pyr_modulation'))
            self.pyr_angle = eval(parser.get('pyramid1stChan', 'pyr_angle'))
            self.pyr_thr = eval(parser.get('pyramid1stChan', 'pyr_thr'))
            self.percent_extra_subaps = eval(parser.get('pyramid1stChan', 'percent_extra_subaps'))
            self.throughput = self.tel_throughput * eval(parser.get('pyramid1stChan', 'throughput_factor'))
            self.pyr_fov = eval(parser.get('pyramid1stChan', 'pyr_fov'))
            self.RONval = eval(parser.get('pyramid1stChan', 'RONval'))
            
            # TODO: here?
            self.Zstroke = eval(parser.get('pyramid1stChan', 'Zstroke'))
            self.z_first_mode = eval(parser.get('pyramid1stChan', 'z_first_mode'))
            self.last_mode = eval(parser.get('pyramid1stChan', 'last_mode'))
            
            self.seg_pist_sig_masked = eval(parser.get('pyramid1stChan', 'seg_pist_sig_masked'))
            self.seg_sig_masked = eval(parser.get('pyramid1stChan', 'seg_sig_masked'))
            self.remove_seg_piston = eval(parser.get('pyramid1stChan', 'remove_seg_piston'))

            self.rec_type = eval(parser.get('pyramid1stChan', 'rec_type'))
            self.ao_thr = eval(parser.get('pyramid1stChan', 'ao_thr'))
            
            self.spp_rec_type = eval(parser.get('pyramid1stChan', 'spp_rec_type'))

            self.excess_noise = eval(parser.get('pyramid1stChan', 'excess_noise'))
            self.ogtl_Ts = eval(parser.get('pyramid1stChan', 'ogtl_Ts'))
            self.ogtl_gain = eval(parser.get('pyramid1stChan', 'ogtl_gain'))
            self.ogtl_detrend_deg = eval(parser.get('pyramid1stChan', 'ogtl_detrend_deg'))
            if parser.has_option('pyramid1stChan','probeInjectionStarts'):
                self.ogtl_probeInjectionStarts = eval(parser.get('pyramid1stChan','probeInjectionStarts'))
            else:
                self.ogtl_probeInjectionStarts = 0.25
            
            self.use_presaved_ogtl = eval(parser.get('pyramid1stChan', 'use_presaved_ogtl'))
            self.omgi = eval(parser.get('pyramid1stChan', 'omgi'))
            self.use_presaved_omgi = eval(parser.get('pyramid1stChan', 'use_presaved_omgi'))

            #---- Pyramid initialization
            self.wfs = ceo.Pyramid(self.nLenslet, self.nPx, modulation=self.pyr_modulation, throughput=self.throughput, separation=self.pyr_separation/self.nLenslet)
            if self.pyr_modulation == 1.0: 
                self.wfs.modulation_sampling = 16
            #wfs.modulation_sampling = 64

            #---- NGS initialization
            self.gs = ceo.Source(self.band, magnitude=self.mag, zenith=0.,azimuth=0., rays_box_size=self.D, 
                    rays_box_sampling=self.nPx, rays_origin=[0.0,0.0,25])
            self.gs.rays.rot_angle = self.pyr_angle*np.pi/180

            print('Number of NGAO GS photons/s/m^2: %.1f'%(self.gs.nPhoton))
            print('Number of expected NGAO GS photons [ph/s/m^2]: %.1f'%(self.e0*10**(-0.4*self.mag)/self.PupilArea))
            print(u"Number of pixels across %1.1f-m array: %d"%(self.D,self.nPx))
            
            
            #Initialise the the second channel
            self.secondChannelType = eval(parser.get('2ndChan', 'sensorType'))
            
            if self.secondChannelType == 'lift':
                sectionName = eval(parser.get('2ndChan','liftSectionName' ))
                
                # self.doubleChan2 = [ Chan2(path, parametersFile,sectionName[0],0),
                #               Chan2(path, parametersFile,sectionName[1],1)]
                if not(load):
                    self.doubleChan2 = [ Chan2(self.tempFolder, parametersFile,sectionName[0],0),
                                  Chan2(self.tempFolder, parametersFile,sectionName[1],1)]
                else:
                    self.doubleChan2 = [ Chan2(path, parametersFile,sectionName[0],0),
                                  Chan2(path, parametersFile,sectionName[1],1)]
                self.chan2 = self.doubleChan2[0]
                self.doubleChan2[0].chan1wl = self.gs.wavelength
                self.doubleChan2[1].chan1wl = self.gs.wavelength
            elif self.secondChannelType == 'idealpistonsensor':
                self.chan2 =Chan2(path, parametersFile)
                self.onps = ceo.IdealSegmentPistonSensor(self.D, self.nPx, segment='full')
            else:
                self.chan2 = Chan2(path, parametersFile)
    
                # self.gs.reset()
                # self.gmt.reset()
                # self.gmt.propagate(self.gs)
                # self.onps.calibrate(self.gs)  # computes reference vector
            
                # #-----> Ideal SPS - M2 segment piston IM calibration
                # print("KL0 - SPS IM:")
                # self.PSstroke = 25e-9 #m
                # self.D_M2_PSideal = self.gmt.calibrate(self.onps, self.gs, mirror="M2", mode="Karhunen-Loeve", stroke=self.PSstroke, first_mode=0, last_mode=1)
            
                # # index to KL0 in the command vector (to update controller state with 2nd NGWS command)
                # self.KL0_idx = self.n_mode*np.array([0,1,2,3,4,5])
                # self.R_M2_PSideal = np.linalg.pinv(self.D_M2_PSideal)#ideal phasing sensor reconstructor
                # self.SPP2ndCh_thr = self.gs.wavelength * 0.95#detection threshold for the second channel
                
                
                        
        if self.simul_turb:
            self.atm_duration = eval(parser.get('turbulence', 'screenDuration'))
            self.atm_t0 = eval(parser.get('turbulence', 't0atm'))
            if (self.atm_t0+self.totSimulTime)>self.atm_duration:
                print('warning: not enough atmosphere, increasing the atm_duration')
                # self.atm_t0 = self.atm_duration-self.totSimulTime
                self.atm_duration = self.totSimulTime+self.atm_t0
            self.n_duration = np.ceil(int(self.atm_duration))
            self.atm_duration = 1.0
            
            # self.atm_t0 /= self.Tsim
            self.wind_scale = eval(parser.get('turbulence', 'wind_scale'))
            self.zen_angle = eval(parser.get('turbulence', 'zen_angle'))
            
            self.L0 = eval(parser.get('turbulence', 'L0'))
            self.altitude = np.array(eval(parser.get('turbulence', 'altitude')))
            self.xi0 = np.array(eval(parser.get('turbulence', 'xi0')))
            if parser.has_option('turbulence','seeing' ):
                self.seeing = eval(parser.get('turbulence', 'seeing'))
                self.r0a  = 0.9759 * 500e-9/(self.seeing * ceo.constants.ARCSEC2RAD)  # Fried parameter at zenith [m]
                self.r0   = self.r0a * np.cos( self.zen_angle*np.pi/180 )**(3./5.)
            else:
                self.r0 = eval(parser.get('turbulence', 'r0'))
                self.r0a = self.r0 / np.cos( self.zen_angle*np.pi/180 )**(3./5.)
                self.seeing = 0.9759*500*10**-9/self.r0a * ceo.constants.RAD2ARCSEC
            self.wind_speed = self.wind_scale[0] * np.array(eval(parser.get('turbulence', 'wind_speed')))
            self.wind_direction = np.array(eval(parser.get('turbulence', 'wind_direction')))
            self.meanV = np.sum(self.wind_speed**(5.0/3.0)*self.xi0)**(3./5.)
            self.tau0 = 0.314*self.r0/self.meanV
        else: 
            self.seeing = 0.0
        self.simul_variable_seeing = eval(parser.get('turbulence', 'simul_variable_seeing'))
        
        
        self.TN_dir += self.chan2.sensorType +'/cwl{0}/seeing{1}-{2}/mag{3}/'.format(int(self.chan2.cwl*1e9), 
                           int(np.floor(self.seeing)), int((self.seeing-np.floor(self.seeing))*1000),int(self.mag))
        

    def M2_KL_modes(self,figsize = (15,5)):
        
        # Retrieve M2 KL modes
        M2 = self.gmt.M2.modes.M.host()
        print(M2.shape)
        
        #Create circular mask
        rows = self.gmt.M2.modes.N_SAMPLE
        cols = self.gmt.M2.modes.N_SAMPLE
        nsets = self.gmt.M2.modes.N_SET
        nkls = self.gmt.M2.modes.N_MODE

        xVec = np.linspace(-1,1,cols)
        yVec = np.linspace(-1,1,rows)
        [x,y] = np.meshgrid(xVec,yVec) # rows x cols
        r = np.hypot(x,y)

        #Mask for outer segments
        M2masko = np.full((rows,cols),np.nan)
        M2masko[(r <= 1)]=1.0
        M2npo = np.sum(r <= 1)

        #Mask for central segment
        M2maskc = np.full((rows,cols),np.nan)
        #Roc = 0.344
        
        M2maskc[np.logical_and(r <= 1, r >= self.Roc)] = 1.0
        M2npc = np.sum(M2maskc == 1)

        # Choose KL to display
        this_set = 1    # 0: outer segments;    1: central segment
        this_kl = 100

        if this_set == 0:
            M2mask = M2masko
            M2np = M2npo
        else:
            M2mask = M2maskc
            M2np = M2npc

        KLmap = np.reshape(M2[:,this_set*nkls+this_kl], (rows,cols), order='F')*M2mask
        KLrms = np.sqrt( np.sum(KLmap[M2mask==1]**2)/M2np )
        print("RMS of KL mode %d is: %.2f"%(this_kl, KLrms))

        if self.VISU==True:
            fig, (ax1,ax2) = plt.subplots(ncols=2)
            fig.set_size_inches(figsize)
            imm = ax1.imshow(KLmap, cmap=plt.cm.winter)
            fig.colorbar(imm, ax=ax1)
            ax1.set_title('M2 KL %d'%(this_kl), fontsize=15)

            ax2.plot(xVec,KLmap[:,int(cols/2)])
            ax2.plot(yVec,KLmap[int(rows/2),:])
            ax2.grid()
            plt.show()


        # Compute RMS of general modes
        this_set = 0    # 0: outer segments;    1: central segment

        if this_set == 0:
            M2mask = M2masko
            M2np = M2npo
        else:
            M2mask = M2maskc
            M2np = M2npc

        KLrms = np.zeros(nkls)
        for this_kl in range(nkls):
            KLmap = np.reshape(M2[:,this_set*nkls+this_kl], (rows,cols), order='F')*M2mask
            KLrms[this_kl] = np.sqrt( np.sum(KLmap[M2mask==1]**2)/M2np )

        if self.VISU:
            plt.plot(KLrms,'+--')
            #plt.ylim((0.99,1.01))
            #plt.xlim((0,20))
            plt.grid()
            plt.xlabel('KL number')
            plt.ylabel('KL RMS')
            plt.show()
        
    def propagatePyramid(self,figsize = (15,5)):
        
        if self.simul_onaxis_AO:
            self.gs.reset()
            self.gmt.reset()
            self.gmt.propagate(self.gs)
            self.ph_fda_on = self.gs.phase.host(units='nm')

            #-- Calibrate PYR (Pupil registration, and slope null vector)
            self.wfs.calibrate(self.gs, percent_extra_subaps=self.percent_extra_subaps, thr = self.pyr_thr)
            extr = (self.wfs.ccd_frame.shape[0]-240)//2  # hard-coded..... cropping the frame to match OCAM2 actual number of pixels

            if self.VISU:
                fig, (ax1,ax2) = plt.subplots(ncols=2)
                fig.set_size_inches(figsize)
                imm1 = ax1.imshow( np.sum(self.wfs.indpup, axis=0, dtype='bool')[extr:-1-extr, extr:-1-extr] )
                imm2 = ax2.imshow(self.wfs.ccd_frame[extr:-1-extr, extr:-1-extr])
                fig.colorbar(imm2, ax=ax2)
                plt.show()
                

    def KL_modes_exit_pupil(self,this_kl,amp,figsize = (15,5)):
                
        #show segment KL modes in the exit pupil
        # this_kl = 100
        # amp = 10e-9
        self.gmt.reset()
        self.gs.reset()
        self.gmt.M2.modes.a[:,this_kl] = amp
        self.gmt.M2.modes.update()
        self.gmt.propagate(self.gs)

        if self.VISU:
            plt.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r)#, origin='lower')#, vmin=-25, vmax=25)
            plt.colorbar()
            plt.title('segment KL#%d, %0.0f nm RMS'%(this_kl,amp*1e9))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plt.show()


    def pyr_display_signals_base(self,wavefrontSensorobject, sx, sy, title=None,figsize = (15,5)):
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
            
    def pyr_display_signals(self, title=None,figsize = (15,5)):                
        ## Visualize reference slope vector in 2D (for a flat WF)
        if self.simul_onaxis_AO and self.VISU:
            sx,sy = self.wfs.get_ref_measurement(out_format='list')
            return self.pyr_display_signals_base(self.wfs,sx, sy,figsize = figsize)
    
    def getOrCalibrateIM(self,figsize = (15,5)):
        """IM calibration for the first channel"""
        if self.simul_onaxis_AO==True:
            # tmpgmtTruss = self.gmt.project_truss_onaxis
            # self.gmt.project_truss_onaxis = False
            ### Calibrate IM and save or Restore from file
            RECdir = self.dir_calib
            #----> fitted KLs (ASM_fittedKLs)
            # fname = 'IM'+'_fittedKLs500_S7OC03945'+'_PYR_thr%1.3f_mod%d_SAv%d_px%1.2fcm_sep%d.npz'%(pyr_thr,int(wfs.modulation),wfs.n_sspp,pixelSize*100,pyr_separation)
            #-----> Double diagonalization modes (ASM_fittedKLs_doubleDiag)
            fname = 'IM'+'_KLsDD675_S7OC0%d'%(self.Roc*1e4)+'_PYR_thr%1.3f_mod%d_SAv%d_px%1.2fcm_sep%d.npz'%(self.pyr_thr,int(self.wfs.modulation),self.wfs.n_sspp,self.pixelSize*100,self.pyr_separation)
            self.IMfnameFull = os.path.normpath(os.path.join(RECdir,fname))
            print(self.IMfnameFull)
            #TODOFR: Flag to generate or not the file
            if not os.path.isfile(self.IMfnameFull):  
                self.D_M2_MODES = self.gmt.NGWS_calibrate(self.wfs,self.gs, stroke=self.Zstroke)
                tosave = dict(D_M2=self.D_M2_MODES, first_mode=self.z_first_mode, Stroke=self.Zstroke)
                np.savez(self.IMfnameFull, **tosave)
            else: 
                print(u'Reading file: '+self.IMfnameFull)
                ftemp = np.load(self.IMfnameFull)
                self.D_M2_MODES = ftemp.f.D_M2
                ftemp.close()
            
            # self.gmt.project_truss_onaxis = tmpgmtTruss

            nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
            self.n_mode = nall//self.nseg
            print(u'AO WFS - M2 Segment Modal IM:')
            print(self.D_M2_MODES.shape)
            
    def extractIMRequestedModes(self,figsize = (15,5)):
        if self.simul_onaxis_AO==True:
            self.D_M2_split = []
            for kk in range(self.nseg):
                Dtemp = self.D_M2_MODES[:, kk*self.n_mode : self.n_mode*(kk+1)]
                self.D_M2_split.append(Dtemp[:, 0:self.last_mode-self.z_first_mode])
            self.D_M2_MODES = np.concatenate(self.D_M2_split[0:self.nseg], axis=1) 
            nall = (self.D_M2_MODES.shape)[1]  ## number of KL DoFs calibrated
            self.n_mode = nall // self.nseg
            print('AO WFS - M2 Segment Modal IM:')
            print(self.D_M2_MODES.shape)
            # free memory
            del self.D_M2_split
            del Dtemp
            del self.last_mode
                        
    def applySignalMasks(self,figsize = (15,5)):
        if self.simul_onaxis_AO==True:
            self.D_AO = self.D_M2_MODES.copy()
            if self.seg_sig_masked:
                print("\nCalibrating segment signal masks...")
                segment_signal_mask = self.gmt.NGWS_segment_mask(self.wfs, self.gs)
                self.D_AO = self.gmt.NGWS_apply_segment_mask(self.D_AO, segment_signal_mask['mask'])
                self.wfs.segment_signal_mask = segment_signal_mask
                print("Segment signal masks applied.")
            if self.seg_pist_sig_masked:
                print("\nCalibrating segment piston signal masks...")
                segpist_signal_mask = self.gmt.NGWS_segment_piston_mask(self.wfs, self.gs)
                self.D_AO = self.gmt.NGWS_apply_segment_piston_mask(self.D_AO, segpist_signal_mask['mask'])
                self.wfs.segpist_signal_mask = segpist_signal_mask
                print("Segment piston signal masks applied.")                
            if self.simul_onaxis_AO and self.seg_pist_sig_masked and self.seg_sig_masked and self.VISU:
                segmSigMaskAll = np.sum(self.wfs.segpist_signal_mask['mask'][0:6],axis=0)
                pistSigMaskAll = np.sum(self.wfs.segment_signal_mask['mask'], axis=0)
                (sx2dtemp, sy2dtemp) = self.pyr_display_signals_base(self.wfs,segmSigMaskAll, pistSigMaskAll, 
                                           title=['segment piston masks', 'segment masks'],figsize = figsize)

                
    def showIMSignalsPrep(self,figsize = (15,5)):        
        self.gmt.reset()
        segment = 6
        # TODO: this looks strange...
        self.gmt.M2.modes.a[segment,0] = 1e-9
        self.gmt.M2.modes.update()
        self.gs.reset()
        self.gmt.propagate(self.gs)
        self.wfs.reset()
        self.wfs.analyze(self.gs)
        ll = self.wfs.get_measurement(out_format='list')
        (sx_temp,sy_temp) = self.pyr_display_signals_base(self.wfs,*ll,figsize = figsize)
        
        
    def showIMSignals(self, segment, mode,figsize = (15,5)):
        if self.simul_onaxis_AO==True and self.VISU==True:
            this_segment = segment   # from 0 to 6. Central segment: 6
            this_kl = mode
            this_mode = self.n_mode*this_segment+this_kl
            sx = self.D_AO[0:self.wfs.n_sspp,this_mode] / 1e9
            sy = self.D_AO[self.wfs.n_sspp:,this_mode]  / 1e9
            sx2d, sy2d = self.pyr_display_signals_base(self.wfs,sx,sy,figsize = figsize)
            
    def removeCentralPiston(self,figsize = (15,5)):            
        if self.simul_onaxis_AO:
            # Remove segment piston:
            if self.remove_seg_piston:
                self.segpist_idx = self.n_mode*np.array([6])
                self.D_AO = np.delete(self.D_AO, self.segpist_idx, axis=1) 
                
                
    def doIMsvd(self,figsize = (15,5)):
        if self.simul_onaxis_AO and self.rec_type=='LS':
            print('Condition number: %f'%np.linalg.cond(self.D_AO))
            self.Uz, self.Sz, self.Vz =np.linalg.svd(self.D_AO)
            if self.VISU:
                fig, ax = plt.subplots()
                fig.set_size_inches(figsize)
                ax.semilogy(self.Sz/np.max(self.Sz), 'o-', )
                ax.grid()
                ax.tick_params(labelsize=14)
                ax.set_xlabel('eigenmode number', fontsize=14)
                ax.set_ylabel('normalized singular value', fontsize=14)
                
                
    def showEigenMode(self, this_eigen,figsize = (15,5)):
        
        if self.simul_onaxis_AO and self.rec_type=='LS' and self.VISU:
            eigenmodevec = np.copy(self.Vz[this_eigen,:])
            if self.remove_seg_piston == True:
                for idx in self.segpist_idx:
                    eigenmodevec = np.insert(eigenmodevec,idx,0)
            self.gmt.reset()
            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = np.ascontiguousarray(eigenmodevec.reshape((self.nseg,-1))) *1e-6
            self.gmt.M2.modes.update()
            self.gs.reset()
            self.gmt.propagate(self.gs)

            fig, (ax1,ax2) = plt.subplots(ncols=2)
            fig.set_size_inches(figsize)
            imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation='None',cmap='RdYlBu',origin='lower')
            clb = fig.colorbar(imm, ax=ax1)  #, fraction=0.012, pad=0.03,format="%.1f")
            clb.set_label('nm WF', fontsize=12)
            ax2.plot(eigenmodevec, '+--')
            tx=ax2.set_xlabel('KL mode number')
            tx=ax2.set_ylabel('coefficient [a.u.]')
            plt.show()


    def generalizedIMInverse(self,figsize = (15,5)):
        if self.simul_onaxis_AO and self.rec_type=='LS':
            # select threshold to filter last eigenmode.
            #ao_thr = (Sz/np.max(Sz))[-2:].sum()/2 
            #TODOFR: .ini
            self.R_AO = np.linalg.pinv(self.D_AO, rcond=self.ao_thr)
            print('AO SH WFS - M2 Segment Modal Rec:')
            print(self.R_AO.shape)
            if self.VISU:
                plt.imshow(np.log(np.abs(self.R_AO)))
                plt.colorbar()

    def addRowsOfRemovedModes(self,figsize = (15,5)):
        if self.simul_onaxis_AO:
            if self.remove_seg_piston:
                for idx in self.segpist_idx:
                    self.R_AO = np.insert(self.R_AO, idx, 0, axis=0)
   
                print('AO SH WFS - M2 Segment Modal Rec:')
                print(self.R_AO.shape)

                self.R_AO = cp.asarray(self.R_AO)
                self.ntotmodes = self.R_AO.shape[0]
    
    def regularizedSegmentPistonReconstructor(self, figsize=(15,5)):
        
        #--- Restore pre-saved segment KL modes covariance matrix
        segklCov_file = np.load(self.dir_calib+'ASM_DDKLs_S7OC%05d_675kls_CovMat.npz'%(self.Roc*1e4))
        segklCov = segklCov_file.f.CovMat
        segklCov_nmodes = segklCov_file.f.n_modes
        print('Covariance matrix compute for r0=%0.1f cm and L0=%0.0f m'%(segklCov_file.f.r0*1e2,segklCov_file.f.L0))
        
        #--- Extract segKL CovMat with only requested KL modes
        segklCov_rem_idx = []   # index to seg KLs to remove
        for kk in range(self.nseg):
            segklCov_rem_idx.append( np.arange(segklCov_nmodes*kk + self.n_mode, segklCov_nmodes*(kk+1)) )
        segklCov_rem_idx = np.concatenate(segklCov_rem_idx)

        segklCov = np.delete(segklCov, segklCov_rem_idx, axis=1)
        segklCov = np.delete(segklCov, segklCov_rem_idx, axis=0)

        #--- Show the covariance matrix
        if self.VISU:
            plt.imshow(np.log10( np.abs(segklCov)*(1e9)**2), vmin=0, vmax=6, interpolation='None')
            CovTicks = np.arange(self.nseg) * self.n_mode
            plt.xticks(CovTicks) # xTicks will be set at multiples of # segment KLs
            plt.yticks(CovTicks) # xTicks will be set at multiples of # segment KLs

            clb = plt.colorbar()
            clb.set_label('$\log(\mid C_{KL}\mid)$', fontsize=15)
            
        #--- Extract Chh
        allsp_idx = self.n_mode*np.array([0,1,2,3,4,5,6]) 
        hhCov = np.delete(segklCov, allsp_idx, axis=1)
        hhCov = np.delete(hhCov, allsp_idx, axis=0)

        #--- Extract Cph
        self.allhh_idx = np.setdiff1d(np.arange(self.n_mode*self.nseg), allsp_idx)
        phCov = np.delete(segklCov, allsp_idx, axis=1)
        phCov = np.delete(phCov, self.allhh_idx, axis=0)
        
        #--- Segment piston estimator based on HO turb statistics
        hhcov_thr = 1e-5
        self.R_p = phCov @ np.linalg.pinv(hhCov, rcond=hhcov_thr)

        print('Regularized segment piston reconstructor for bootstrapping:')
        print(self.R_p.shape)        
                
                
    def idealPistonSensor(self,figsize = (15,5)):
        if self.simul_onaxis_AO:    
            # idealized SPS initialization
            self.onps = ceo.IdealSegmentPistonSensor(self.D, self.nPx, segment='full')
            self.gs.reset()
            self.gmt.reset()
            self.gmt.propagate(self.gs)
            self.onps.calibrate(self.gs)  # computes reference vector
            #-----> Ideal SPS - M2 segment piston IM calibration
            print("KL0 - SPS IM:")
            self.D_M2_PSideal = self.gmt.calibrate(self.onps, self.gs, mirror="M2", mode="Karhunen-Loeve", 
                                                   stroke=self.chan2.PSstroke, first_mode=0, last_mode=1)
            # index to KL0 in the command vector (to update controller state with 2nd NGWS command)
            self.KL0_idx = self.n_mode*np.array([0,1,2,3,4,5])
            if self.VISU:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(figsize)
                imm = ax1.pcolor(self.D_M2_PSideal)
                ax1.grid()
                fig.colorbar(imm, ax=ax1)#, fraction=0.012) 
                
            self.R_M2_PSideal = np.linalg.pinv(self.D_M2_PSideal)
            self.SPP2ndCh_thr = self.gs.wavelength * self.chan2.wvl_fraction
            
    
            
            

            
    def probe_signal(self, fs, kk,figsize = (15,5)):
        for pp in self.ogtl_probe_params:
            self.ogtl_probe_vec[pp['seg'], pp['mode']] = pp['amp']*np.sin(2*np.pi*pp['freq']/fs*kk)


    def optical_gain_cl(self, probes_in,probes_out,figsize = (15,5)):    
        probe_in_fitcoef = [np.polyfit(self.ogtl_timevec,this_probe,self.ogtl_detrend_deg) 
                            for this_probe in probes_in]
        self.ogtl_fitcoef.append(probe_in_fitcoef)
        probe_in_fit = [poly_nomial(self.ogtl_timevec,this_coeff) 
                        for this_coeff in probe_in_fitcoef]
        probes_in = np.array([pr-prf for (pr,prf) in zip(probes_in,probe_in_fit)])
        A_in = 2*np.sqrt(np.mean(probes_in * self.cosFn, axis=1)**2 
                         + np.mean(probes_in * self.sinFn, axis=1)**2)

        probe_out_fitcoef = [np.polyfit(self.ogtl_timevec,this_probe,self.ogtl_detrend_deg) 
                             for this_probe in probes_out]
        probe_out_fit = [poly_nomial(self.ogtl_timevec,this_coeff) 
                         for this_coeff in probe_out_fitcoef]
        probes_out = np.array([pr-prf for (pr,prf) in zip(probes_out,probe_out_fit)])
        A_out = 2*np.sqrt(np.mean(probes_out * self.cosFn, axis=1)**2 
                          + np.mean(probes_out * self.sinFn, axis=1)**2)

        OG = A_out / A_in   
        print('')
        for kk in range(len(self.seg_list)):
            print('  Optical Gain of mode S%d KL%d: %0.3f'%(self.seg_list[kk],self.mode_list[kk],OG[kk]))

        return OG
        
    def doOTGLsimul(self,figsize = (15,5)):
        if self.ogtl_simul:
            # Warning! hard-code below. It assues the basis used is ASM_fittedKLs_S7OC04184_675kls.ceo
            radord_data = dict(np.load(self.dir_calib+'ASM_fittedKLs_S7OC04184_675kls_radord.npz'))
            self.radord_all_outer = radord_data['outer_radord'][0:self.n_mode]
            self.radord_all_centr = radord_data['centr_radord'][0:self.n_mode]
            outer_max_radord = np.max(self.radord_all_outer)
            self.probe_radord = np.round(np.arange(6)*outer_max_radord/6+3).astype('int')
            self.probe_outer = np.zeros(6, dtype='int')
            for jj in range(6):
                self.probe_outer[jj] = np.argwhere(radord_data['outer_radord'] == self.probe_radord[jj])[1]
            # Add segment piston to the probe set
            # self.probe_outer = np.insert(self.probe_outer, 0, 0)
            # self.probe_radord = np.insert(self.probe_radord, 0, 1)
            
            print(self.probe_outer)
            print(self.probe_radord)
            
            # Define probe signal parameters [segment#, mode#, amplitude, frequency]
            # self.ogtl_probe_params = []
            # self.ogtl_probe_params.append(dict(seg=0, mode=self.probe_outer[0], amp=2.5e-9, freq=310))
            # self.ogtl_probe_params.append(dict(seg=1, mode=self.probe_outer[1], amp=2.5e-9, freq=310))
            # self.ogtl_probe_params.append(dict(seg=2, mode=self.probe_outer[2], amp=2.5e-9, freq=310))
            # self.ogtl_probe_params.append(dict(seg=3, mode=self.probe_outer[3], amp=2.0e-9, freq=310))
            # self.ogtl_probe_params.append(dict(seg=4, mode=self.probe_outer[4], amp=2.0e-9, freq=310))
            # self.ogtl_probe_params.append(dict(seg=5, mode=self.probe_outer[5], amp=2.0e-9, freq=310))
            ##### self.ogtl_probe_params.append(dict(seg=6, mode=probe_centr, amp=2.5e-9, freq=310))
            self.ogtl_probe_params = []
            # self.ogtl_probe_params.append(dict(seg=4, mode=self.probe_outer[0], amp=20.0e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=0, mode=self.probe_outer[0], amp=2.5e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=1, mode=self.probe_outer[1], amp=2.5e-9, freq=210))
            self.ogtl_probe_params.append(dict(seg=2, mode=self.probe_outer[2], amp=2.5e-9, freq=210))
            self.ogtl_probe_params.append(dict(seg=3, mode=self.probe_outer[3], amp=2.0e-9, freq=210))
            self.ogtl_probe_params.append(dict(seg=4, mode=self.probe_outer[4], amp=2.0e-9, freq=210))
            self.ogtl_probe_params.append(dict(seg=5, mode=self.probe_outer[5], amp=2.0e-9, freq=210))
            

            self.ogtl_probe_vec = np.zeros((self.nseg,self.n_mode))

            self.seg_list  = [kk['seg']  for kk in self.ogtl_probe_params]
            self.mode_list = [kk['mode'] for kk in self.ogtl_probe_params]
            self.freq_list = [kk['freq'] for kk in self.ogtl_probe_params]
            
            self.gmt.reset()
            self.gs.reset()
            
            for pp in self.ogtl_probe_params:
                self.ogtl_probe_vec[pp['seg'], pp['mode']] = pp['amp']
            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = self.ogtl_probe_vec
            self.gmt.M2.modes.update()
            self.gmt.propagate(self.gs)

            if self.VISU:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(figsize)
                imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('probe modes')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f", fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)
                plt.show()


    def dessenne_gain(self, fvec, a_OLCL_psd, figsize = (15,5)):

        def modalrms(gi, Ts, Td, ffi):
            absRTF2 = np.abs(rejectTF(gi, freq_vec, Ts, Td, ffi))**2
            resvari = np.sum(absRTF2 * ol_psd) * delta_freq_Hz
            return np.sqrt(resvari)*1e9

        nseg = a_OLCL_psd.shape[0]
        nmodes = a_OLCL_psd.shape[1]
        delta_freq_Hz = fvec[2]-fvec[1]
        freq_vec = fvec[1:]   # remove zero-frequency

        optgain = np.zeros((nseg,nmodes))
        for this_seg in range(nseg):
            for this_mode in range(nmodes):
                ol_psd = a_OLCL_psd[this_seg,this_mode,1:]  # remove zero-frequency
                ffi = self.ffAO.get()[this_seg*self.n_mode+this_mode]
                optres = optimize.minimize_scalar(modalrms, bounds=(0,0.5), method='bounded', args=(self.Tsim,self.Tsim,ffi))
                if optres.success==True:
                    optgain[this_seg,this_mode]=optres.x
        return optgain
    

    def update_r0(self, t):
        self.varseeing_iter = (self.seeing_max-self.seeing_init)*np.sin(2*np.pi*t/self.vs_T)+ self.seeing_init
        return 0.9759 * 500e-9/(self.varseeing_iter*ceo.constants.ARCSEC2RAD) 
    
    def simulTurbulence(self,figsize = (15,5)):        
        if self.simul_turb:
            print('       Mean wind speed : %2.1f m/s' % self.meanV)
            print('                  tau0 : %2.2f ms'%(self.tau0*1e3))
            print('    r0 @ 500nm @ %d deg: %2.1f cm'%(self.zen_angle,self.r0*1e2))
            print('seeing @ 500nm @ %d deg: %2.2f arcsec'%(self.zen_angle,0.9759*500e-9/
                                                           self.r0*ceo.constants.RAD2ARCSEC))

            #atm_fname = 'gmtAtmosphere_median_1min.bin'
            #atm_fullname = os.path.normpath(os.path.join(atm_dir,atm_fname))
            self.atm_fullname=None

            self.atm = ceo.Atmosphere(self.r0,self.L0,len(self.altitude),self.altitude,
                                      self.xi0,self.wind_speed,self.wind_direction,
                             L=26,NXY_PUPIL=self.nPx,fov=0.0*ceo.constants.ARCMIN2RAD, 
                             filename=self.atm_fullname, 
                             # duration=self.atm_duration)
                              duration=self.atm_duration, N_DURATION=self.n_duration)
            
            if self.simul_variable_seeing:
                self.zen_angle = 0
                self.seeing_init = 0.5
                self.seeing_max = 1.0
                self.vs_T = 60 # period of sinusoidal (simulation time should be up to half this amount)
            
        if self.simul_onaxis_AO and self.simul_truss_mask:
            self.gmt.project_truss_onaxis = True
            self.gs.reset()
            self.gmt.reset()
            self.gmt.propagate(self.gs)
            self.ph_fda_on = self.gs.phase.host(units='nm')
            #Override slope null vector
            self.wfs.reset()
            self.wfs.set_reference_measurement(self.gs)
            

            if self.VISU:
                ## Visualize reference slope vector in 2D (for a flat WF)
                ll = self.wfs.get_ref_measurement(out_format='list')
                (sx_ref_2d,sy_ref_2d) = self.pyr_display_signals_base(self.wfs,*ll,figsize = figsize)
                
                
                
    def modalPerfEval(self, this_mode=401,figsize = (15,5)):
        # Segment masks
        self.gmt.reset()
        self.gs.reset()
        self.gmt.propagate(self.gs)
        self.P = np.squeeze(np.array(self.gs.rays.piston_mask))
        self.GMTmask = np.sum(self.P,axis=0).astype('bool')
        self.npseg = np.sum(self.P, axis=1)
        self.nmaskPup = np.sum(self.P)

        ##---- Index to valid points within GMT pupil
        xm, ym = np.where(np.reshape(self.GMTmask,(self.nPx,self.nPx)))

        print("Number of valid mask points over each segment:\n", self.npseg)
        
        if self.VISU:
            plt.imshow(np.reshape(self.GMTmask,(self.nPx,self.nPx)), interpolation=None)    

        # segment KL modes
        if self.eval_perf_modal or self.eval_perf_modal_turb:
            # segKLmat: List containing the seven KLmats (one for each segment)
            self.segKLmat = [np.zeros((self.npseg[segId], self.gmt.M2.modes.n_mode)) for segId in range(self.nseg) ]
            KLamp = 5e-9
            for jj in range(self.gmt.M2.modes.n_mode):
                self.gmt.reset()
                self.gmt.M2.modes.a[:, jj] = KLamp
                self.gmt.M2.modes.update()
                self.gs.reset()
                self.gmt.propagate(self.gs)
                for segId in range(self.nseg):
                    self.segKLmat[segId][:,jj] = np.ravel(self.gs.phase.host()-self.ph_fda_on*1e-9)[self.P[segId,:]] / (KLamp) #normalize to unitary RMS WF
            # Compute KL RMS (should be unitary, but ray tracing to exit pupil may change slightly that...)
            # We will use the RMS to identify "zero" modes (e.g. in central segment w.r.t. outer segments)
            klrms = [np.sqrt(np.sum(self.segKLmat[segId]**2, axis=0)/self.npseg[segId]) for segId in range(self.nseg)]
            self.n_valid_modes = [np.max(np.where(klrms[segId] > 0.5))+1 for segId in range(self.nseg)]

            for segId in range(self.nseg):
                if self.VISU:
                    plt.semilogx(np.arange(self.gmt.M2.modes.n_mode)+1,klrms[segId], '+--', label='seg%d'%(segId+1))
                    plt.legend()
                    plt.show()
                    
            self.inv_segKLmat = [np.linalg.pinv(self.segKLmat[segId][:,0:self.n_valid_modes[segId]]) for segId in range(self.nseg)]
            if self.VISU:
                # Show a particular mode on all segments, just for fun
                caca = np.full(self.nPx**2, np.nan)
                for segId in range(self.nseg):
                    caca[self.P[segId,:]] = self.segKLmat[segId][:,this_mode]
                plt.imshow(caca.reshape((self.nPx,self.nPx)), interpolation='None')#, origin='lower')
                plt.title('KL#%d'%this_mode)
                plt.colorbar()
            
            
    def showSignalsModeSegAmp(self, this_seg, this_kl, this_amp ,figsize = (15,5)):
        if self.VISU:
            self.gmt.reset()
            self.gmt.M2.modes.a[this_seg,this_kl]= this_amp
            self.gmt.M2.modes.update()
            self.gs.reset()
            self.gmt.propagate(self.gs)
            self.wfs.reset()
            self.wfs.analyze(self.gs)
            extr = (self.wfs.ccd_frame.shape[0]-240)//2  # hard-coded..... cropping the frame to match OCAM2 actual number of pixels
            fig, (ax1,ax2) = plt.subplots(ncols=2)
            fig.set_size_inches(figsize)
            phase = (self.gs.phase.host(units='nm')-self.ph_fda_on).ravel()
            phase1 = np.full(self.nPx**2,np.nan)
            phase1[self.GMTmask] = phase[self.GMTmask]
            imm = ax1.imshow(phase1.reshape((self.nPx,self.nPx)), interpolation='None')#,cmap=plt.cm.gist_earth_r)#, origin='lower' vmin=-25, vmax=25)
            #ax1.set_title('S%d KL#%d'%(this_seg+1,this_kl))
            ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
            clb.set_label('$nm$ WF', fontsize=12)
            clb.ax.tick_params(labelsize=12)
            ax2.set_title('PWFS CCD frame')
            imm2 = ax2.imshow(self.wfs.camera.frame.host()[extr:-1-extr, extr:-1-extr])#, origin='lower')
            ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plt.show()
            #plt.colorbar()
            sxtemp = self.wfs.get_sx()
            sytemp = self.wfs.get_sy()
            [sx2dtemp, sy2dtemp] = self.pyr_display_signals_base(self.wfs,sxtemp,sytemp,figsize=figsize)
            
            
    def islandMode(self, index,figsize = (15,5)):
        S7mask = np.zeros(self.nPx**2)
        S7mask[self.P[index,:]]=1
        segmask_label, nlabels = ndimage.label(S7mask.reshape((self.nPx,self.nPx)))
        if self.VISU:
            plt.imshow(segmask_label)
            plt.colorbar()
            plt.show()
        print(nlabels)            
        #--- Remove "islands" of 1 or 2 pixels wrongly identified above as separate ones
        islands = []
        for ll in range(nlabels):
            caca = segmask_label == ll+1
            if np.sum(caca) > 10:
                islands.append(caca)
        if self.VISU:
            plt.imshow(np.sum(islands,axis=0))
            plt.colorbar()
            plt.show()
            
        #---- Create island influence matrix and inverse
        self.nisl = len(islands)
        self.IslMat = np.zeros((self.npseg[index],self.nisl))
        for jj in range(self.nisl):
            self.IslMat[:,jj] = islands[jj].ravel()[self.P[6,:]]

        self.inv_IslMat = np.linalg.pinv(self.IslMat)
        
        
        #---- Island modes as linear combination of segment KL modes
        if self.eval_perf_modal == True:
            Isl2segKLmat = self.inv_segKLmat[index] @ self.IslMat

            inv_S7KLmat_pistfree = np.linalg.pinv(self.segKLmat[index][:,1:self.n_mode])
            Isl2segKLmat_pistfree = inv_S7KLmat_pistfree @ self.IslMat
            inv_Isl2segKLmat_pistfree = np.linalg.pinv(Isl2segKLmat_pistfree)

            # Visualization of island modes linear decomposition
            
            this_isl = 5
            this_amp = 100e-9
            first_kl = 0 #write 0 if you want to include S7 segmnent piston
            this_comm = Isl2segKLmat[first_kl:,this_isl] * this_amp

            self.gmt.reset()
            self.gs.reset()
            self.gmt.M2.modes.a[index,first_kl:self.n_valid_modes[index]] = this_comm
            self.gmt.M2.modes.update()
            self.gmt.propagate(self.gs)

            if self.VISU:
                fig, (ax1,ax2) = plt.subplots(ncols=2)
                fig.set_size_inches(figsize)

                imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('on-axis WF')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)
                ax1.set_xlim([self.nPx/2-300,self.nPx/2+300])
                ax1.set_ylim([self.nPx/2-300,self.nPx/2+300])

                ax2.semilogx(np.arange(first_kl,self.n_valid_modes[index])+1+first_kl,this_comm*1e9)
                ax2.grid()
                ax2.set_xlabel('segment KL number + 1')
                ax2.set_ylabel('coeff amp [nm RMS]')
                plt.show()

            this_isl = 5
            this_amp = 100e-9
            first_kl = 1 #MUST be 1
            this_comm = Isl2segKLmat_pistfree[:,this_isl] * this_amp

            self.gmt.reset()
            self.gs.reset()
            self.gmt.M2.modes.a[index,first_kl:self.n_mode] = this_comm
            self.gmt.M2.modes.update()
            self.gmt.propagate(self.gs)

            if self.VISU:
                fig, (ax1,ax2) = plt.subplots(ncols=2)
                fig.set_size_inches(figsize)

                imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('on-axis WF')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)
                ax1.set_xlim([self.nPx/2-300,self.nPx/2+300])
                ax1.set_ylim([self.nPx/2-300,self.nPx/2+300])

                ax2.semilogx(np.arange(first_kl,self.n_mode)+1,this_comm*1e9)
                ax2.grid()
                ax2.set_xlabel('segment KL number + 1')
                ax2.set_ylabel('coeff amp [nm RMS]')
                plt.show()
                
                
    def globalTTModes(self,figsize = (15,5)):
        vv = np.linspace(-1,1,self.nPx)*(self.D/2)
        [x_ep,y_ep] = np.meshgrid(vv,vv) # rows x cols

        # Rotate coordinate system
        rotmat = np.array([[np.cos(self.gs.rays.rot_angle), -np.sin(self.gs.rays.rot_angle)],
                           [np.sin(self.gs.rays.rot_angle),  np.cos(self.gs.rays.rot_angle)]])
        xytemp = rotmat @ np.array([x_ep.ravel(),y_ep.ravel()])
        x_epr = np.reshape(xytemp[0,:],(self.nPx,self.nPx))
        y_epr = np.reshape(xytemp[1,:],(self.nPx,self.nPx))
        
        PTTmat = np.zeros((self.nmaskPup,3))
        PTTmat[:,0] = 1
        PTTmat[:,1] = x_epr[self.GMTmask.reshape((self.nPx,self.nPx))]
        PTTmat[:,2] = y_epr[self.GMTmask.reshape((self.nPx,self.nPx))]

        PTT_Dmat = np.matmul(np.transpose(PTTmat), PTTmat)/self.nmaskPup;
        PTT_Lmat = np.linalg.cholesky(PTT_Dmat)
        PTT_inv_Lmat = np.linalg.pinv(PTT_Lmat)
        self.PTTmato = np.matmul(PTTmat, np.transpose(PTT_inv_Lmat))

        print("WF RMS of PTT modes:")
        print(np.array_str(np.sum(self.PTTmato**2,axis=0)/self.nmaskPup, precision=2))

        self.inv_PTTmato = np.linalg.pinv(self.PTTmato)        
        self.gmt.reset()
        self.gs.reset()
        self.gmt.propagate(self.gs)
        self.wfgrad_ref = self.gs.wavefront.gradientAverageFast(self.D)*ceo.constants.RAD2MAS
        self.seg_wfgrad_ref = self.gs.segmentsWavefrontGradient()*ceo.constants.RAD2MAS
        
        ttamp = 100e-9  # in m WF RMS
        self.gmt.reset()
        mode = np.zeros((self.nPx**2))
        wfgradmat = np.zeros((2,2))
        for jj in range(2):
            mode[self.GMTmask] = self.PTTmato[:,jj+1]  # avoid global piston
            self.gs.reset()
            self.gmt.propagate(self.gs)
            self.gs.wavefront.axpy(ttamp,ceo.cuFloatArray(host_data=mode))
            wfgradmat[:,jj] = (self.gs.wavefront.gradientAverageFast(self.D)*ceo.constants.RAD2MAS - self.wfgrad_ref) / ttamp

        self.inv_wfgradmat = np.linalg.inv(wfgradmat)


    def phaseAveraging(self,figsize = (15,5)):
        if self.do_Phase_integration:
            self.imgs = ceo.Source('H', magnitude=8, zenith=0.,azimuth=0., rays_box_size=self.D, 
                    rays_box_sampling=self.nPx, rays_origin=[0.0,0.0,25])
            self.imgs.rays.rot_angle = self.pyr_angle*np.pi/180
            
            
    #------------------- Complex Amplitude computation.
    #     Inputs: None. It uses current amplitude and phase of gs object
    #     Keywords: FT: if True, returns the FT of the complex amplitude
    def complex_amplitude(self, FT=False,figsize = (15,5)):
        A0 = cp.zeros((self.nPx*self.npad-1,self.nPx*self.npad-1))
        A0[0:self.nPx,0:self.nPx] = ceo.ascupy(self.gs.amplitude)
        F0 = cp.zeros((self.nPx*self.npad-1,self.nPx*self.npad-1))
        F0[0:self.nPx,0:self.nPx] = ceo.ascupy(self.gs.phase)
        F0 -= A0*cp.array(self.gs.piston())   # Remove global piston
        if FT==False:
            return A0*cp.exp(1j*self.knumber*F0)
        else:
            return cp.fft.fft2(A0*cp.exp(1j*self.knumber*F0))

        
    #-------------------- Short exposure PSF computation.
    #     Input: W_ft -> Fourier Transform of the complex amplitude
    def psf_se(self, W_ft, shifted=False, norm_factor=1,figsize = (15,5)):
        if shifted==True:
            return cp.fft.fftshift( cp.abs(W_ft)**2 ) / norm_factor
        else:
            return cp.abs(W_ft)**2 / norm_factor

    # ------------------ Perfect coronagraph PSF computation
    def perfect_coro_psf_se(self, shifted=False, norm_factor=1,figsize = (15,5)):
        W1_ft = self.complex_amplitude(FT=True)
        SR_se = cp.exp(cp.array(-(self.knumber*self.gs.phaseRms())**2))  # Marechal approximation
        Wc_ft = W1_ft - cp.sqrt(SR_se)*self.Rf
        if self.do_wo_coro_too:
            return self.psf_se(W1_ft, shifted=shifted, norm_factor=norm_factor), self.psf_se(Wc_ft, shifted=shifted, norm_factor=norm_factor)
        else:
            return self.psf_se(Wc_ft, shifted=shifted, norm_factor=norm_factor) 


    #----------------- Computes the image displacement w.r.t to the reference position [in mas]
    def psf_centroid(self, myPSF, centr_ref,figsize = (15,5)):
        return np.array(ndimage.center_of_mass(myPSF) - centr_ref)*self.fp_pixscale

    #------------------ Estimates the SR (intensity at center of image from a normalized PSF)
    def strehl_ratio(self, myPSF, centr_ref,figsize = (15,5)):
        return myPSF[tuple(np.round(centr_ref).astype('int'))]

    def sr_and_centroid(self, myPSF,figsize = (15,5)):
        return self.strehl_ratio(myPSF), self.psf_centroid(myPSF)

    # ----------------- Estimate intensity at given separation
    def intensity_query(self, PSFprof, Rfvec, sep_query,figsize = (15,5)):
        ss = np.where(Rfvec > sep_query)[0][0]  # index of first distance value larger than sep_query
        int_req = PSFprof[ss] + \
            (PSFprof[ss]-PSFprof[ss-1]) / (Rfvec[ss]-Rfvec[ss-1]) * (sep_query-Rfvec[ss])
        return int_req
    
    #------------------- Funcdtion to show PSFs side by side
    #   im_display_size: +/- mas from center
    def show_two_psfs(self, myPSF1, myPSF2, im_display_size=150, log=True, clim=None,figsize = (15,5)):
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(figsize)

        im_range_mas = np.array([-im_display_size, im_display_size])
        im_range_pix = np.rint(im_range_mas/self.fp_pixscale + self.nPx*self.npad/2).astype(int)

        if log==True:
            myPSF1 = np.log10(myPSF1[im_range_pix[0]:im_range_pix[1],im_range_pix[0]:im_range_pix[1]])
            myPSF2 = np.log10(myPSF2[im_range_pix[0]:im_range_pix[1],im_range_pix[0]:im_range_pix[1]])
        else:
            myPSF1 = myPSF1[im_range_pix[0]:im_range_pix[1],im_range_pix[0]:im_range_pix[1]]
            myPSF2 = myPSF2[im_range_pix[0]:im_range_pix[1],im_range_pix[0]:im_range_pix[1]]

        imm = ax1.imshow(myPSF1, origin='lower', interpolation=None,
                   extent=[-im_display_size,im_display_size,-im_display_size,im_display_size]) 

        clb = fig.colorbar(imm, ax=ax1)
        if log==True: clb.set_label('$log_{10}$(PSF)', fontsize=12)
        imm.set_clim(clim)
        ax1.set_xlabel('mas', fontsize=12)
        ax1.tick_params(labelsize=12)

        imm2 = ax2.imshow(myPSF2, origin='lower', interpolation=None,
                   extent=[-im_display_size,im_display_size,-im_display_size,im_display_size])

        clb2 = fig.colorbar(imm2, ax=ax2)
        if log==True: clb2.set_label('$log_{10}$(PSF)', fontsize=12)
        imm2.set_clim(clim)
        ax2.set_xlabel('mas', fontsize=12)
        ax2.tick_params(labelsize=12)
        plt.show()

    #------------------- Function to show PSF
    #   im_display_size: +/- mas from center
    def show_psf(self, myPSF, im_display_size=150, clim=[-8,0], fig=None,ax1=None,figsize = (15,5)):
        if ax1==None:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(figsize)

        im_range_mas = np.array([-im_display_size, im_display_size])
        im_range_pix = np.rint(im_range_mas/self.fp_pixscale + self.nPx*self.npad/2).astype(int)

        imm = ax1.imshow(np.log10(myPSF[im_range_pix[0]:im_range_pix[1],im_range_pix[0]:im_range_pix[1]]), 
                   extent=[-im_display_size,im_display_size,-im_display_size,im_display_size], 
                         origin='lower', interpolation=None)
        clb = fig.colorbar(imm, ax=ax1)
        imm.set_clim(clim)
        ax1.set_xlabel('mas')
        
    #------------------- Function to show PSF and radial profile
    #   im_display_size: +/- mas from center
    def show_psf_and_profile(self, myPSF, myProf, im_display_size=150, clim=[-8,0],figsize = (15,5)):
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(figsize)
        self.show_psf(myPSF, im_display_size=im_display_size, clim=clim, fig=fig,ax1=ax1)
        ax2.loglog(self.Rf.ravel(), myPSF.ravel(), '.')
        #ax2.hold('on')
        ax2.loglog(self.Rfvec,myProf, 'r', linewidth=3)
        #ax2.hold('off')
        ax2.set_xlim([1e1,1e4])
        ax2.set_ylim([1e-10,1])
        ax2.grid()
        ax2.set_xlabel('radial distance [mas]')
        ax2.set_ylabel('normalized intensity') 
        plt.show()
        
    def showDL_PSF(self,figsize = (15,5)):
        ## init psf visualization data
        self.gs.reset()
        self.gmt.reset()
        self.gmt.propagate(self.gs)
        self.norm_pup = np.sum(self.gs.amplitude.host())**2
        W0_ft = self.complex_amplitude(FT=True)
        self.PSF0 = self.psf_se(W0_ft, shifted=True, norm_factor=self.norm_pup)
        self.centr_ref = np.array(ndimage.center_of_mass(self.PSF0.get()))
        nfx, nfy = self.PSF0.shape
        Xf, Yf = np.ogrid[0:nfx, 0:nfy]
        self.Rf = np.hypot(Xf-self.centr_ref[0], Yf-self.centr_ref[1]) * self.fp_pixscale # distance from PSF center in mas
        binSize = self.fp_pixscale # bin size for radial average in mas
        nbins = np.round(self.Rf.max()/binSize)
        self.Rflabel = np.rint(nbins * self.Rf/self.Rf.max())
        self.Rfidx = np.arange(0,self.Rflabel.max()+1)
        self.Rfvec = self.Rfidx * binSize
        self.PSF0prof = ndimage.mean(self.PSF0.get(), labels=self.Rflabel, index=self.Rfidx)

        
        if self.VISU:
            self.show_psf_and_profile(self.PSF0.get(), self.PSF0prof, im_display_size=1000)
            print('Strehl ratio @ %.1f um of diff-limited PSF: %.2f'%(self.lim*1e6,np.max(self.PSF0)))
    
    
    
    
    def runAll(self,mode = 'all',dbg = False, figsize = (15,5)):
        self.tid.tic()
        self.M2_KL_modes(figsize = figsize)
        self.propagatePyramid(figsize = figsize)
        if mode == 'all':
            self.KL_modes_exit_pupil(100,10**-9,figsize = figsize)
            self.pyr_display_signals(figsize = figsize)
        
        
        self.getOrCalibrateIM(figsize = figsize)
        self.extractIMRequestedModes(figsize = figsize)
        self.applySignalMasks(figsize = figsize)
        self.showIMSignalsPrep(figsize = figsize)
        if mode == 'all':
            self.showIMSignals(6, 100,figsize = figsize)
            self.showIMSignals(0, 0,figsize = figsize)
        
        self.removeCentralPiston(figsize = figsize)
        self.doIMsvd(figsize = figsize)
        if mode == 'all':
            self.showEigenMode(202,figsize = figsize)
        
        self.generalizedIMInverse(figsize = figsize)
        self.addRowsOfRemovedModes(figsize = figsize)
        if self.spp_rec_type == 'MMSE':
            self.regularizedSegmentPistonReconstructor(figsize = figsize)
        if self.secondChannelType ==  'idealpistonsensor':
            self.idealPistonSensor(figsize = figsize)
        elif self.secondChannelType == 'lift':
            self.doubleChan2[0].CalibAndRM(self.gmt,figsize = figsize)
            self.doubleChan2[1].CalibAndRM(self.gmt,figsize = figsize)
        else:
            self.chan2.CalibAndRM(self.gmt,figsize = figsize)
        
        self.doOTGLsimul(figsize = figsize)
        self.simulTurbulence(figsize = figsize)
        self.modalPerfEval(figsize = figsize)
        if mode == 'all':
            self.showSignalsModeSegAmp(6, 20, -25e-9,figsize = figsize)
            self.islandMode(6,figsize = figsize)
        
        self.globalTTModes(figsize = figsize)
        self.phaseAveraging(figsize = figsize)
        if mode == 'all':
            self.showDL_PSF(figsize = figsize)
        self.tid.toc()
        self.housekeep_initTime = self.tid.elapsedTime*1e-3
        self.housekeep_stepTime = []
        self.closedLoopSimul(dbg = dbg, figsize = figsize)
        
            
    def closedLoopSimul(self,figsize = (15,5),dbg = False):
        self.KL0_idx = self.n_mode*np.array([0,1,2,3,4,5])
        ### Reset objects
        self.gs.reset()
        self.wfs.reset()
        self.gmt.reset()
        
        if self.do_Phase_integration:
            self.imgs.reset()
            
            
        if self.simul_M1polish:

            self.gmt.M1.modes.a[:,0] = 1   # coefficient of 1.0 means nominal values
            self.gmt.M1.modes.update()

            if self.VISU:
                self.gs.reset()
                self.gmt.propagate(self.gs)
                fig, ax1 = plt.subplots()
                fig.set_size_inches(figsize)
                InitScr = self.gs.phase.host(units='nm')-self.ph_fda_on

                imm = ax1.imshow(InitScr, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('on-axis WF')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)

                print('WF RMS: %.1f nm'%(self.gs.phaseRms()*1e9))
                print('SPP RMS: %.1f nm'%(np.std(self.gs.piston('segments'))*1e9))
                

        # if self.seg_pist_scramble:
        #     self.pist_scramble_rms=10e-9   # in m RMS SURF

        if self.M2_modes_scramble:
            self.M2modes_scramble_rms = 5e-9  # in m RMS SURF        
            

        if self.seg_pist_scramble:
            # Generate piston scramble
            rng = np.random.default_rng(self.scramble_seed)
            pistscramble  = rng.uniform(-2.0*self.gs.wavelength,+2.0*self.gs.wavelength,size=self.nseg-1)/2
            # pistscramble *= self.pist_scramble_rms/np.std(pistscramble)
            # pistscramble[6] *= 0
            # pistscramble -= np.mean(pistscramble)
            # pistscramble -= pistscramble[6]  # relative to central segment
            
            # Apply it to M2
            self.gmt.M2.motion_CS.origin[:-1,2] = pistscramble
            self.gmt.M2.motion_CS.update()
            
        if self.seg_pist_scramble == 'byhand':
            pistscramble = self.pist_scramble_rms
            self.gmt.M2.motion_CS.origin[:,2] = pistscramble
            self.gmt.M2.motion_CS.update()

        if self.M2_modes_scramble:
            self.M2modes_scramble = np.random.normal(loc=0.0, scale=1, size=(self.nseg,self.n_mode))
            self.M2modes_scramble *= self.M2modes_scramble_rms/np.std(self.M2modes_scramble)
            self.M2modes_scramble[:,0] = 0
            #--- If you wanted to introduce just one mode over one segment....
            #self.M2modes_scramble = np.zeros((7,self.n_mode))
            #self.M2modes_scramble[0,1] = 100e-9
            self.gmt.M2.modes.a[:,self.z_first_mode:] = self.M2modes_scramble
            self.gmt.M2.modes.update()
        else:
            self.M2modes_scramble = np.zeros((self.nseg,self.n_mode))
            
            
        if self.VISU:
            self.gs.reset()
            self.gmt.propagate(self.gs)
            fig, ax1 = plt.subplots()
            fig.set_size_inches(figsize)
            imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
            ax1.set_title('on-axis WF')
            ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
            clb.set_label('$nm$ WF', fontsize=12)
            clb.ax.tick_params(labelsize=12)
            print(' WF RMS: %.1f nm'%(self.gs.phaseRms()*1e9))
            print('SPP RMS: %.1f nm'%(np.std(self.gs.piston('segments'))*1e9))

        AOinit = np.round(self.AOinitTime/self.Tsim) #round(0.10/Tsim)               # close the AO loop

        if not self.simul_turb:
            #--- Timing when no turbulence is simulated
            # Tsim= 1e-3                         # simulation sampling time [s]
            self.totSimulIter = round(self.totSimulTime/self.Tsim)                  # simulation duration [iterations]
            # totSimulTime = self.totSimulIter*self.Tsim   # simulation duration [in seconds]
            self.totSimulInit = 30                  # start PSF integration at this iteration
            # self.SPPctrlInit  = float('inf')        # start controlling SPP
            # self.SPP2ndChInit = float('inf')        # Apply 2nd channel correction (with ideal SPS)
            self.SPPctrlInit  = AOinit+round(self.SPPctrlInitTime/self.Tsim)+1       # start controlling SPP
            self.SPP2ndChInit = AOinit+round(self.SPP2ndChInitTime/self.Tsim)        # start applying 2nd channel correction (with ideal SPS)
            self.SPP2ndCh_Ts  = round(self.chan2.exposure_time/self.Tsim)       # 2nd channel sampling time (in number of iterations)
            self.SPP2ndCh_count = 0
            
        else:
            #--- Timing when turbulence is simulated (longer exposures)
            # Tsim = 1e-3                             # simulation sampling time [s]
            # totSimulTime = self.totSimulTime        # simulation duration [in seconds]
            self.totSimulIter = round(self.totSimulTime/self.Tsim) # simulation duration [iterations]

            AOinit = np.round(self.AOinitTime/self.Tsim) #round(0.10/Tsim)               # close the AO loop
            self.SPPctrlInit  = AOinit+round(self.SPPctrlInitTime/self.Tsim)+1       # start controlling SPP
            self.SPP2ndChInit = AOinit+round(self.SPP2ndChInitTime/self.Tsim)        # start applying 2nd channel correction (with ideal SPS)
            self.SPP2ndCh_Ts  = round(self.chan2.exposure_time/self.Tsim)       # 2nd channel sampling time (in number of iterations)
            self.SPP2ndCh_count = 0

            self.totSimulInit = AOinit+1000         # start PSF integration at this iteration

            if self.do_Phase_integration:
                PhIntInit = AOinit+1000  # start averaging the phase maps
                PhInt_Ts = 200
                PhInt_count = 0

        if self.simul_windload:
            self.wlstartInit = 0      

        if self.ogtl_simul:
            probeSigInit = AOinit+round(self.ogtl_probeInjectionStarts/self.Tsim) #SPPctrlInit+9          # Start injection probe signals on selected modes 
            # ogtl_Ts_1 = 500e-3 # sampling during bootstrap [s]
            ogtl_sz = np.int(self.ogtl_Ts /self.Tsim)
            ogtl_count = 0
            # self.ogtl_detrend_deg = 5   # detrend time series
            nprobes = len(self.ogtl_probe_params)
            self.ogtl_radord_deg  = 3 #nprobes-1   # fit OG vs radial order curve 
            # self.ogtl_gain = 0.3

            self.ogtl_timevec = np.arange(0,ogtl_sz)*self.Tsim
            self.cosFn = np.array([np.cos(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])
            self.sinFn = np.array([np.sin(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])

            #ogtl_Ts_2 = 500e-3 #sampling after convergence [s]
            self.ogtl_reconfig_iter =float('inf')#probeSigInit + np.int(18*ogtl_Ts_1/self.Tsim)  # make sure this coincides with an OGTL iteration.

            self.ogtl_ogc_probes = np.ones(nprobes)
            #self.ogtl_ogc_allmodes = np.ones(self.n_mode)
            OGC_all  = cp.ones((self.nseg,self.n_mode))
            self.ogtl_ogeff_probes_iter = []
            #ogtl_ogeff_iter = []
            #ogtl_ogc_iter = []
            self.ogtl_ogc_centr_iter = []
            self.ogtl_ogc_outer_iter = []
            self.ogtl_ogc_probes_iter = []
            self.ogtl_ticks = []
            # ogtl_Ts = ogtl_Ts_1 # select sampling to be used first
        else:
            OGC_all  = cp.ones((self.nseg,self.n_mode))
            
        #use_presaved_ogtl = True

        if self.use_presaved_ogtl:
            ogtl_fname = self.dir_calib+'ogc_r0-{:0.1f}cm_mag{:.0f}' \
                            .format(self.r0*1e2, self.mag)+'_v1.npz'
            #ogtl_fname = 'ogc_r0-12.8cm_mag12_v1.npz'
            print(ogtl_fname)
            ogtl_fdata = dict(np.load(ogtl_fname))
            for this_seg in range(6):
                OGC_all[this_seg,:] = cp.asarray(ogtl_fdata['ogtl_ogc_outer'])
            OGC_all[6,:] = cp.asarray(ogtl_fdata['ogtl_ogc_centr'])
            self.ogtl_ogc_probes = OGC_all[0,self.mode_list].get()

        if self.VISU:
            plt.plot(OGC_all[0,:].get())
            plt.plot(OGC_all[6,:].get())
            
        #--- optimized gain integrator (pre-calibrated gains)
        if self.use_presaved_omgi:
            gain_fname = self.dir_calib+'omgi_r0-{:0.1f}cm_mag{:.0f}' \
                            .format(self.r0*1e2, self.mag)+'_v1.npz'
            #gain_fname = 'omgi_r0-12.8cm_mag12_v1.npz'
            print(gain_fname)
            optgain_data = dict(np.load(gain_fname))
            gAO = cp.zeros((self.nseg,self.n_mode))
            gAO[0:6,:] = cp.asarray(optgain_data['optgain_outer'])
            gAO[6,:]   = cp.asarray(optgain_data['optgain_centr'])
        else:
            gAO = cp.ones((self.nseg,self.n_mode)) * 0.5
            #gAO[:,0:200] = 0.8
            #gAO[:,0] = 0.5  #SPP gain
        
        gAO = cp.reshape(gAO, (self.ntotmodes,1))
        g_pist = 1.0 # gain of regularized SPP command
        omgi_can_update=False
        optgain_buffer = np.zeros((self.ntotmodes,2))
        optgain_buffer[:,0] = gAO[:,0].get()
        self.optgain_iter = [optgain_buffer[:,0]]
        self.omgi_ticks = []
        self.ffAO = cp.ones(self.ntotmodes) # Forgetting factors; placeholder only
        
        #---- delay simulation
        self.tot_delay = 2   # in number of frames
        delay  = self.tot_delay-1 
        Mdecal = cp.zeros((self.tot_delay, self.tot_delay))
        Mdecal[0, 0] = 1 
        if delay >= 0:
            Mdecal[0:delay,1:self.tot_delay] = cp.identity(delay) 
        Vinc   = cp.zeros((1,self.tot_delay))
        Vinc[0, 0]   = 1 
        
        comm_buffer = cp.zeros((self.ntotmodes,self.tot_delay))
        myAOest1    = cp.zeros((self.ntotmodes,1))

        pist_buffer = cp.zeros((self.nseg,self.tot_delay))

        #---- Time histories to save in results
        self.a_M2_iter   = np.zeros((self.nseg,self.n_mode,self.totSimulIter))   # integrated M2 modal commands
        self.da_M2_iter  = np.zeros((self.nseg,self.n_mode,self.totSimulIter))   # delta M2 modal commands
        self.a_OLCL_iter = np.zeros((self.nseg,self.n_mode,self.totSimulIter))
        
        if self.save_telemetry:
            wfs_meas_iter = np.zeros((self.wfs.get_measurement_size(),self.totSimulIter)) # pyramid WFS signals
            self.wfe_gs_iter = np.zeros(self.totSimulIter)              # on-axis WFE
            self.spp_gs_iter = np.zeros((self.nseg,self.totSimulIter))          # on-axis (differential) segment phase piston error
            self.seg_wfe_gs_iter = np.zeros((self.nseg,self.totSimulIter))      # on-axis WFE over each segment
            wfgrad_iter = np.zeros((2,self.totSimulIter))          # on-axis WF gradient (estimate of image motion) in mas
            seg_wfgrad_iter = np.zeros((14,self.totSimulIter))     # on-axis segment WF gradients [alpha_x, alpha_y]
        if self.save_telemetry and self.do_psf_le:
            sr_iter = np.zeros(self.totSimulIter)
            im_centr_iter = np.zeros((2,self.totSimulIter))
        if self.save_telemetry and self.simul_variable_seeing:
            self.r0_iter = np.zeros(self.totSimulIter)

        if self.do_crazy_spp:
            self.crazy_spp_iter = np.zeros((self.nseg,self.totSimulIter))

        if self.eval_perf_modal:
            seg_aRes_gs_iter = np.zeros((self.nseg,self.gmt.M2.modes.n_mode,self.totSimulIter)) # seg KL modal coefficients buffer     
            #ZresIter = np.zeros((nzern,self.totSimulIter))

        if self.eval_perf_modal_turb:
            seg_aTur_gs_iter = np.zeros((self.nseg,self.gmt.M2.modes.n_mode,self.totSimulIter)) # seg KL modal coefficients buffer 
            #ZturIter = np.zeros((nzern,self.totSimulIter))

        if self.simul_windload:
            pistjumps_com=np.zeros(self.nseg)

        if self.do_Phase_integration:
            PhInt_iter = []   #integrated phase maps
            PhInst_iter = []  #instantaneous phase realization
        
        if self.analyse_island_effect:
            Isliter = np.zeros((self.nisl,self.totSimulIter))
            Islwfe = np.zeros(self.totSimulIter)
        
        if self.save_phres_decimated:
            save_ph_Ts = 10e-3
            save_ph_niter = np.int(save_ph_Ts / self.Tsim)
            save_ph_count=0
            maskfile = './savedir%d/GMTmask.npz'%self.GPUnum
            np.savez_compressed(maskfile, GMTmask=self.GMTmask, nPx=self.nPx) 
            
            
        if self.save_psfse_decimated:
            save_psf_Ts = 10e-3
            save_psf_niter = np.int(save_psf_Ts / self.Tsim)
            save_psf_count=0
            psf_range_mas = np.array([-900, 900]) # +/- 1 arcsec
            psf_range_pix = np.rint(psf_range_mas/self.fp_pixscale + self.nPx*self.npad/2).astype(int)
            
        self.ogtl_fitcoef = []
        if self.secondChannelType =='lift':
            self.doubleChan2[0].pistEstList = []
            self.doubleChan2[0].correctionList = []
            if dbg:
                
                self.doubleChan2[0].debugframe = []
            self.doubleChan2[1].pistEstList = []
            
            self.doubleChan2[1].correctionList = []
            if dbg:
                self.doubleChan2[1].debugframe = []
        else:
            self.chan2.pistEstList = []
            if dbg:
                self.chan2.correctionList = []
                self.chan2.debugframe = []
            
        
        for jj in range(self.totSimulIter):
            self.tid.tic()
            self.gs.reset()
            if self.secondChannelType == 'lift':
                self.doubleChan2[0].gs.reset()
                self.doubleChan2[1].gs.reset()
            elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                self.chan2.gs.reset()
                

            #----- Update Turbulence --------------------------------------------
            if self.simul_turb:
                
                if self.simul_variable_seeing:
                    self.atm.r0 = self.update_r0(jj*self.Tsim)
                    if self.save_telemetry:
                        self.r0_iter[jj] = self.atm.r0
                self.atm.ray_tracing(self.gs, self.pixelSize,self.nPx,self.pixelSize,
                                     self.nPx, self.atm_t0+(jj*self.Tsim))
                if self.secondChannelType == 'lift':
                    self.atm.ray_tracing(self.doubleChan2[0].gs, self.doubleChan2[0].pixelSize,
                                         self.doubleChan2[0].nPx,self.doubleChan2[0].pixelSize,
                                         self.doubleChan2[0].nPx, self.atm_t0+(jj*self.Tsim))
                    self.atm.ray_tracing(self.doubleChan2[1].gs, self.doubleChan2[1].pixelSize,
                                         self.doubleChan2[1].nPx,self.doubleChan2[1].pixelSize,
                                         self.doubleChan2[1].nPx, self.atm_t0+(jj*self.Tsim))

                elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                    self.atm.ray_tracing(self.chan2.gs, self.chan2.pixelSize,
                                         self.chan2.nPx,self.chan2.pixelSize,
                                         self.chan2.nPx, self.atm_t0+(jj*self.Tsim))

                if self.do_Phase_integration:
                    if jj >= PhIntInit:
                        self.atm.ray_tracing(self.imgs, self.pixelSize,self.nPx,
                                             self.pixelSize,self.nPx, self.atm_t0+(jj*self.Tsim))

                if self.eval_perf_modal_turb:
                    PhaseTur = np.squeeze(self.gs.wavefront.phase.host()) * self.GMTmask
                    PhaseTur[self.GMTmask] -= np.mean(PhaseTur[self.GMTmask])  # global piston removed
                    #ZturIter[:,jj] = inv_zmat @ PhaseTur[GMTmask]

                    for segId in range(self.nseg):
                        seg_aTur_gs_iter[segId,0:self.n_valid_modes[segId],jj] = np.dot(self.inv_segKLmat[segId], PhaseTur[self.P[segId,:]])

            #----- Introduce wind load effects as M1 and M2 RBM perturbations ---
#            if self.simul_windload:
#                self.gmt.M1.motion_CS.origin[:] = np.ascontiguousarray(wldata['Data']['M1 RBM'][jj*int(self.Tsim/Twl)+wlstartInit,:,:3])
#                self.gmt.M1.motion_CS.euler_angles[:] = np.ascontiguousarray(wldata['Data']['M1 RBM'][jj*int(self.Tsim/Twl)+wlstartInit,:,3:])
#                self.gmt.M1.motion_CS.update()

#                self.gmt.M2.motion_CS.origin[:] = np.ascontiguousarray(wldata['Data']['M2 RBM'][jj*int(self.Tsim/Twl)+wlstartInit,:,:3])
#                #gmt.M2.motion_CS.origin[:,2] += np.ascontiguousarray(pistjumps_com)
#                self.gmt.M2.motion_CS.euler_angles[:] = np.ascontiguousarray(wldata['Data']['M2 RBM'][jj*int(self.Tsim/Twl)+wlstartInit,:,3:])
#                self.gmt.M2.motion_CS.update()

            #----- Apply AO command, taking into account the simulated delay --------------------
            if self.forcePhased:
                comm_buffer[self.n_mode*np.arange(7)] = cp.asarray(
                    self.gs.piston(where='segments')[0,0:7,np.newaxis]*np.ones((7,2)))
                if self.forceseg:
                    comm_buffer[self.n_mode*np.array(self.forceseg)] -= \
                        cp.asarray(np.array(self.forceto)[:,np.newaxis])
            
            
            nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
            self.M2modes_command = cp.asnumpy(comm_buffer[0:nall,delay].reshape((self.nseg,-1)))
            # print('comm_buffer first occurence when jj = {}'.format(jj))
            # print(comm_buffer)
            if self.spp_rec_type=='MMSE' and jj <= self.SPPctrlInit:
                self.M2modes_command[:,0] += cp.asnumpy(pist_buffer[:,delay])

            #-> Apply probe signal command
            if self.ogtl_simul:
                if jj >= probeSigInit:
                    self.probe_signal(1/self.Tsim,jj-probeSigInit)
                    self.M2modes_command += self.ogtl_probe_vec
                    ogtl_count+=1

            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = self.M2modes_scramble - self.M2modes_command
            self.a_M2_iter[:,:,jj] = self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode]
            self.gmt.M2.modes.update()

            #----- On-axis WFS measurement ---------------------------------------------------
            self.gmt.propagate(self.gs)
            # hdu = fits.PrimaryHDU(self.gs.phase.host())
            # hdul = fits.HDUList(hdu)
            # hdul.writeto('/raid1/gmt_data/CEO/turb_phasescreens/seeing0-968/atmresidual-{}.fits'
            #              .format(jj))
            if self.secondChannelType == 'lift':
                self.gmt.propagate(self.doubleChan2[0].gs)
                self.gmt.propagate(self.doubleChan2[1].gs)
            
            elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                self.gmt.propagate(self.chan2.gs)
                # hdu = fits.PrimaryHDU(self.chan2[0].gs.phase.host())
                # hdul = fits.HDUList(hdu)
                # hdul.writeto('/raid1/gmt_data/CEO/turb_phasescreens/seeing0-968Chan2-nLenslet{:.0f}/atmresidual-{}.fits'
                #              .format(self.chan2[0].nLenslet,jj))

            if self.do_Phase_integration:
                if jj >= PhIntInit:
                    self.gmt.propagate(self.imgs)
                    PhInt_count+=1

                    if PhInt_count==PhInt_Ts:
                        PhInt_iter.append(self.imgs.phase.host()/PhInt_Ts)
                        PhInst_iter.append(self.gs.phase.host())
                        self.imgs.reset()
                        PhInt_count=0

            # save phase screens every N iterations
            if self.save_phres_decimated: #and jj >= totSimulInit:
                if save_ph_count == save_ph_niter-1:
                    PhaseMap = np.squeeze(self.gs.wavefront.phase.host())
                    PhaseMap[self.GMTmask] -= np.mean(PhaseMap[self.GMTmask])  # global piston removed
                    phres_temp = dict(phres = PhaseMap[self.GMTmask], timeStamp=jj*self.Tsim)
                    phres_fname='./savedir%d/phres_%04d'%(self.GPUnum,jj)
                    #savemat(phres_fname, phres_temp)
                    np.savez_compressed(phres_fname, **phres_temp)
                    save_ph_count = 0
                else:
                    save_ph_count+=1

            # Project residual phase maps onto segment KL modes
            if self.eval_perf_modal:
                PhaseRes = np.squeeze(self.gs.wavefront.phase.host()) * self.GMTmask
                PhaseRes[self.GMTmask] -= np.mean(PhaseRes[self.GMTmask])  # global piston removed
                #ZresIter[:,jj] = inv_zmat @ PhaseRes[self.GMTmask]

                for segId in range(self.nseg):
                    seg_aRes_gs_iter[segId,0:self.n_valid_modes[segId],jj] = np.dot(self.inv_segKLmat[segId], PhaseRes[self.P[segId,:]])  

            PhaseRes = np.squeeze(self.gs.wavefront.phase.host()) * self.GMTmask
            if self.analyse_island_effect:
                Isliter[:,jj] = self.inv_IslMat @ PhaseRes[self.P[6,:]]
                IslPh = self.IslMat @ Isliter[:,jj]
                Islwfe[jj] = np.std(IslPh)

            self.wfs.reset()
            if self.simul_wfs_noise:
                self.wfs.propagate(self.gs)
                #self.wfs.camera.readOut(self.Tsim, self.RONval, 0, self.excess_noise)
                self.wfs.readOut(self.Tsim, RON=self.RONval)
                self.wfs.process()
            else:
                self.wfs.analyze(self.gs)   # This simulates a noise-less read-out.


            AOmeasvec = self.wfs.get_measurement()
            if self.save_telemetry:
                wfs_meas_iter[:,jj] = AOmeasvec

            #------ WF Reconstruction and command computation --------------------------------------------
            if jj >= AOinit:
                if self.rec_type=='LS':
                    myAOest1[:,0] = OGC_all.ravel() * (self.R_AO @ cp.asarray(AOmeasvec))


            if jj < self.SPPctrlInit:   # Do not control segment piston
                myAOest1[self.n_mode*np.arange(self.nseg),0] = 0
            elif jj == self.SPPctrlInit and (self.rec_type=='LS' and self.spp_rec_type=='MMSE'):
                #g_pist=0.0
                comm_buffer[self.n_mode*np.arange(self.nseg),:] = pist_buffer

            self.da_M2_iter[:,:,jj]  = cp.asnumpy(myAOest1.ravel()).reshape((self.nseg,self.n_mode))
            self.a_OLCL_iter[:,:,jj] = self.M2modes_command + self.da_M2_iter[:,:,jj]
                

            #------ Second phasing channel measurement and command -------------------------------
            if jj == self.SPP2ndChInit: # Apply a one-time 2nd NGWS channel segment piston correction
                if self.fake_fast_spp_convergence and self.secondChannelType=='idealpistonsensor':
                    self.onps.reset()
                    self.onps.analyze(self.gs)
                    pistjumps_est = self.onps.get_measurement()
                    pistjumps_est -= pistjumps_est[6]
                    pistjumps_com = np.dot(self.R_M2_PSideal,pistjumps_est)
                    comm_buffer[self.KL0_idx,:] += cp.asarray(pistjumps_com[0:6,np.newaxis])
                    self.onps.reset()
                    # print('comm_buffer when jj == self.SPP2ndChInit')
                    # print(comm_buffer)
                if self.secondChannelType == 'lift':
                    self.doubleChan2[0].wfs.reset()
                    self.doubleChan2[1].wfs.reset()
                    self.doubleChan2[0].propagate()
                    self.doubleChan2[1].propagate()
                elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                    self.chan2.wfs.reset()
                    self.chan2.propagate()

                    #if simul_windload==False:
                    #    gmt.M2.motion_CS.origin[:,2] = np.ascontiguousarray(pistjumps_com)
                    #    gmt.M2.motion_CS.update()

            elif jj > self.SPP2ndChInit:
                if self.SPP2ndCh_count == self.SPP2ndCh_Ts-1:# every time the counter reaches the exposure time for the second channel
                    if self.secondChannelType == 'idealpistonsensor':
                        self.onps.analyze(self.gs)
                        pistjumps_est = self.onps.get_measurement()
                        pistjumps_est -= pistjumps_est[6]
                        pistjumps_2nd = np.zeros(self.nseg)
                        pistjumps_2nd = np.where(pistjumps_est >  self.SPP2ndCh_thr,  self.gs.wavelength, pistjumps_2nd )
                        pistjumps_2nd = np.where(pistjumps_est < -self.SPP2ndCh_thr, -self.gs.wavelength, pistjumps_2nd )                               
                        pistjumps_com = np.dot(self.R_M2_PSideal,pistjumps_2nd)
                        #if simul_windload==False:
                        #    gmt.M2.motion_CS.origin[:,2] = np.ascontiguousarray(pistjumps_com)
                        #    gmt.M2.motion_CS.update()
                        comm_buffer[self.KL0_idx,:] += cp.asarray(pistjumps_com[0:6,np.newaxis])
                        # print('comm_buffer when jj > self.SPP2ndChInit and jj = {}'.format(jj))
                        # print(comm_buffer)

                        self.onps.reset()
                    if self.secondChannelType == 'lift':
                        self.doubleChan2[0].process(dbg = dbg,figsize=figsize)
                        self.doubleChan2[0].pistEstTime.append(jj)
                        self.doubleChan2[0].pistEstList.append(self.doubleChan2[0].piston_estimate)
                        self.doubleChan2[1].process(dbg = dbg,figsize=figsize)
                        self.doubleChan2[1].pistEstTime.append(jj)
                        self.doubleChan2[1].pistEstList.append(self.doubleChan2[1].piston_estimate)

                        # chan1converge = np.array([a*self.gs.wavelength for a in range(-7,8)])
                        # tolerence = 50*10**-9
                        # aprio= np.array([[a-tolerence,a+tolerence] for a in chan1converge])
                        for ss in range(self.nseg-1):
                            self.doubleChan2[0].forCorrection[ss] = self.doubleChan2[0].piston_est4(self.gs.wavelength,
                                                                                  self.doubleChan2[0].cwl,
                                                                                  self.doubleChan2[1].cwl,
                                                                                  self.doubleChan2[0].piston_estimate[ss],
                                                                                  self.doubleChan2[1].piston_estimate[ss],
                                                                                  apriori = self.chan2.aprio.copy(),
                                                                                  amp_conf = self.chan2.amp_conf
                                                                                  )
                            if np.abs(self.doubleChan2[0].forCorrection[ss])<= \
                                self.doubleChan2[0].aprioriTolerence:
                                self.doubleChan2[0].forCorrection[ss] *=0
                        
                        if self.chan2.active_corr:
                            comm_buffer[self.KL0_idx,:] += cp.asarray(self.doubleChan2[0].forCorrection[0:6,np.newaxis])

                        self.doubleChan2[0].correctionList.append(self.doubleChan2[0].forCorrection.copy())
                        self.doubleChan2[0].wfs.reset()
                        self.doubleChan2[1].wfs.reset()

                    elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                        #TODO
                        self.chan2.process(dbg = dbg,figsize=figsize)
                        self.chan2.pistEstTime.append(jj)
                        self.chan2.pistEstList.append(self.chan2.piston_estimate)

                        if dbg and self.chan2.sensorType.lower() == PYRAMID_SENSOR:
                            sx2d, sy2d = self.pyr_display_signals_base(self.chan2.wfs,
                                                        *self.chan2.wfs.get_measurement(out_format='list' ),
                                                        title = ['sx measured by channel 2', 'sy measured by channel 2' ],
                                                        figsize=figsize)
                            self.chan2.debugframe.append([sx2d,sy2d])

                        comm_buffer[self.KL0_idx,:] -= cp.asarray(self.chan2.forCorrection[0:6,np.newaxis])
                        self.chan2.wfs.reset()

                    self.SPP2ndCh_count=0
                else:
                    if self.secondChannelType == 'idealpistonsensor':
                        self.onps.propagate(self.gs)
                    elif self.secondChannelType == 'lift':
                        self.doubleChan2[0].propagate()
                        self.doubleChan2[1].propagate()
                    elif self.secondChannelType in [PYRAMID_SENSOR,PHASE_CONTRAST_SENSOR]:
                        self.chan2.propagate()

                    self.SPP2ndCh_count+=1

            myAOest1 *= gAO
            comm_buffer[:,0] *= self.ffAO #apply forgetting factors
            comm_buffer =  cp.dot(comm_buffer, Mdecal) + cp.dot(myAOest1,Vinc)  # handle time delay
            
            #---- PIston-reguliarized command
            if self.rec_type=='LS' and self.spp_rec_type=='MMSE': 
                a_hh_OL = np.reshape(self.a_OLCL_iter[:,:,jj],(nall,1))[self.allhh_idx]
                a_p_OL = self.R_p @ a_hh_OL
                a_p_OL -= a_p_OL[-1]
                mypist1 = cp.asarray(g_pist*a_p_OL)
                pist_buffer = cp.dot(pist_buffer,Mdecal)
                pist_buffer[:,0] = mypist1[:,0]

                #----- Optical Gain Tracking Loop
            if self.ogtl_simul:
                if ogtl_count == ogtl_sz:
                    #---- Find effective optical gain in closed-loop operation
                    probes_in  = [ self.a_M2_iter[ss,mm,jj-ogtl_sz:jj] 
                                  for (ss,mm) in zip(self.seg_list,self.mode_list)]
                    probes_out = [self.da_M2_iter[ss,mm,jj-ogtl_sz:jj] 
                                  for (ss,mm) in zip(self.seg_list,self.mode_list)]
                    OGeff = self.optical_gain_cl(probes_in,probes_out)
                    
                    # self.OG_cube['{}'.format(jj)]=list(OGeff)
                    self.ogtl_ogeff_probes_iter.append(OGeff.copy())

                    #---- Compute optical gain compensation (OGC) coefficients for probes
                    self.ogtl_ogc_probes = 0.995*self.ogtl_ogc_probes + self.ogtl_gain * (1-OGeff)
                    self.ogtl_ogc_probes_iter.append(self.ogtl_ogc_probes.copy())
                    
                    #---- Fit polynomial to radial order vs OGC 
                    ogc_fitcoefs = np.polyfit(self.probe_radord, self.ogtl_ogc_probes, self.ogtl_radord_deg)
                    self.ogtl_ogc_allmodes_outer = poly_nomial(self.radord_all_outer,ogc_fitcoefs)
                    self.ogtl_ogc_allmodes_centr = poly_nomial(self.radord_all_centr,ogc_fitcoefs)
                    self.ogtl_ogc_outer_iter.append(self.ogtl_ogc_allmodes_outer.copy())
                    self.ogtl_ogc_centr_iter.append(self.ogtl_ogc_allmodes_centr.copy())
            
                    #---- Fit cubic spline to radial order vs OGC (Descoped)
                    #ogc_spline = CubicSpline(self.probe_radord, self.ogtl_ogc_probes, 
                    #                         bc_type='natural', extrapolate=True)
                    #self.ogtl_ogc_allmodes_outer = ogc_spline(self.radord_all_outer)
                    #self.ogtl_ogc_outer_iter.append(self.ogtl_ogc_allmodes_outer.copy())
                    #self.ogtl_ogc_allmodes_centr = ogc_spline(self.radord_all_centr)
                    #self.ogtl_ogc_centr_iter.append(self.ogtl_ogc_allmodes_centr.copy())
                    
                    #passing the ogc to the second channel for extrapolation
                    if self.secondChannelType == 'lift':
                        self.doubleChan2[0].OGTL(self.ogtl_ogc_allmodes_outer,self.gs.wavelength)
                        self.doubleChan2[1].OGTL(self.ogtl_ogc_allmodes_outer,self.gs.wavelength)
                    else:
                        self.chan2.OGTL(self.ogtl_ogc_allmodes_outer,self.gs.wavelength)

                    

                    #---- Update K_OGC matrix
                    if self.omgi:
                        OGC_all_previous = OGC_all.copy()
                        omgi_can_update = True
                        #omgi_can_update *= ogtl_reconfigured
                    for this_seg in range(6):
                        OGC_all[this_seg,:] = cp.asarray(self.ogtl_ogc_allmodes_outer)
                    OGC_all[6,:] = cp.asarray(self.ogtl_ogc_allmodes_centr)

                    #---- Time stamp of OGTL correction
                    self.ogtl_ticks.append(jj*self.Tsim)
                    ogtl_count = 0

                # #---- Increase OGTL sampling frequency after intial convergence, for more stability
                # if jj == self.ogtl_reconfig_iter:
                #     ogtl_Ts = ogtl_Ts_2 # in seconds
                #     ogtl_sz = np.int(ogtl_Ts /self.Tsim)
                #     self.ogtl_detrend_deg = 5
                #     self.ogtl_timevec = np.arange(0,ogtl_sz)*self.Tsim
                #     self.cosFn = np.array([np.cos(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])
                #     self.sinFn = np.array([np.sin(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])

                
            if omgi_can_update==True:
                print("OMGI updated!")
                fvec, a_OLCL_psd = signal.welch(self.a_OLCL_iter[:,:,jj-ogtl_sz:jj], 1/self.Tsim, nperseg=ogtl_sz//2)
                optgain_new = self.dessenne_gain(fvec, a_OLCL_psd).ravel()

                optgain_buffer = np.roll(optgain_buffer,1)
                self.omgi_lpf = 0.5
                optgain_buffer[:,0] = (1-self.omgi_lpf)*optgain_buffer[:,1] + self.omgi_lpf*optgain_new
                self.optgain_iter.append(optgain_buffer[:,0])
                gAO[:,0] = cp.asarray(optgain_buffer[:,0]) * (OGC_all_previous/OGC_all).ravel()
                omgi_can_update=False
                self.omgi_ticks.append(jj*self.Tsim)                
            




            
            #----- Save telemetry data ------------------------------------------
            if self.save_telemetry:
                self.wfe_gs_iter[jj] = self.gs.wavefront.rms()
                self.spp_gs_iter[:,jj] = self.gs.piston(where='segments')
                #this_spp = gs.piston(where='segments')
                #spp_gs_iter[:,jj] = this_spp[0,0:6] - this_spp[0,6]
                self.seg_wfe_gs_iter[:,jj] = self.gs.phaseRms(where='segments')
                wfgrad_iter[:,jj] = self.gs.wavefront.gradientAverageFast(self.D)*ceo.constants.RAD2MAS - self.wfgrad_ref
                seg_wfgrad_iter[:,jj] = np.squeeze(self.gs.segmentsWavefrontGradient()*ceo.constants.RAD2MAS  - self.seg_wfgrad_ref)

            #---- Crazy estimate of segment piston
            if self.do_crazy_spp:
                OPD = np.squeeze(self.gs.wavefront.phase.host())
                OPD[self.GMTmask] -= np.mean(OPD[self.GMTmask])  # global piston removed

                zTTc = self.inv_wfgradmat @ (self.gs.wavefront.gradientAverageFast(self.D)*ceo.constants.RAD2MAS - self.wfgrad_ref)

                OPDztt = np.zeros(self.nPx**2)
                OPDztt[self.GMTmask] = self.PTTmato[:,1:] @ zTTc
                OPDr = OPD - OPDztt
                self.crazy_spp_iter[:,jj] = np.array([np.sum(OPDr[self.P[segId,:]])/self.npseg[segId] for segId in range(self.nseg)])


            #----- Compute short-exposure PSFs --------------------------------

            if self.coro_psf:
                if not self.do_wo_coro_too:
                    PSFcse = self.perfect_coro_psf_se(shifted=True, norm_factor=self.norm_pup)
                else:
                    PSFse, PSFcse = self.perfect_coro_psf_se(shifted=True, norm_factor=self.norm_pup, do_wo_coro_too=True)
                    
                if jj == self.totSimulInit:
                    print('\nStart accumulating coro PSFs\n')
                    PSFcle = PSFcse.copy()
                    if self.do_wo_coro_too:
                        PSFle = PSFse.copy()
                elif jj > self.totSimulInit:
                    PSFcle += PSFcse
                    if self.do_wo_coro_too:
                        PSFle += PSFse
                        
            elif self.do_psf_le:
                PSFse  = self.psf_se(self.complex_amplitude(FT=True), shifted=True, norm_factor=self.norm_pup)
                
                if jj == self.totSimulInit:
                    print('\nStart accumulating PSFs\n')
                    PSFle = PSFse.copy()
                elif jj > self.totSimulInit:
                    PSFle += PSFse
                    
            if self.save_psfse_decimated:
                if save_psf_count == save_psf_niter-1:
                    psftemp = PSFse[psf_range_pix[0]:psf_range_pix[1],psf_range_pix[0]:psf_range_pix[1]]
                    psfres_temp = dict(psfse = psftemp.get(), timeStamp=jj*self.Tsim)
                    psfres_fname='./savedir%d/psfres_%04d'%(self.GPUnum,jj)
                    #savemat(phres_fname, phres_temp)
                    np.savez_compressed(psfres_fname, **psfres_temp)
                    save_psf_count = 0
                else:
                    save_psf_count+=1
                    
            if self.save_telemetry and self.do_psf_le:
                srse, centrse = self.sr_and_centroid(PSFse.get()) 
                sr_iter[jj] = srse #self.strehl_ratio(PSFse.get())
                im_centr_iter[:,jj] = centrse #self.psf_centroid(PSFse.get())
                
            self.tid.toc()
            sys.stdout.write("\r iter: %d/%d, ET: %.2f s, on-axis WF RMS [nm]: %.1f"%(jj, self.totSimulIter, 
                                                                                      self.tid.elapsedTime*1e-3, 
                                                                                      self.gs.phaseRms()*1e9))
            self.housekeep_stepTime.append(self.tid.elapsedTime*1e-3)
            sys.stdout.flush() 
            
            #---------- End of closed-loop iterations!!!
        self.tid.tic()
        #-- show final on-axis residual map
        if self.VISU:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(figsize)
            
            imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
            ax1.set_title('on-axis WF')
            ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            clb = fig.colorbar(imm, ax=ax1, format="%.1f", fraction=0.012, pad=0.03)
            clb.set_label('$nm$ WF', fontsize=12)
            clb.ax.tick_params(labelsize=12)
            
        if self.coro_psf:
            PSFcle /= (self.totSimulIter-self.totSimulInit)
            del PSFcse
            if self.do_wo_coro_too:
                PSFle /= (self.totSimulIter-self.totSimulInit)
                del PSFse
        elif self.do_psf_le:
            PSFle /= (self.totSimulIter-self.totSimulInit)
            del PSFse
                        
        # Compute contrast and long-exposure SR
        if self.coro_psf:
            PSFcleprof = ndimage.mean(PSFcle.get(), labels=self.Rflabel, index=self.Rfidx)
            inten_query = self.intensity_query(PSFcleprof, self.Rfvec, self.sep_req)
            print("Coronagraphic PSF contrast at %.1f mas: %.2e"%(self.sep_req,inten_query))
            
            if self.do_wo_coro_too:
                PSFleprof = ndimage.mean(PSFle.get(), labels=self.Rflabel, index=self.Rfidx)
                print("Long-exposure PSF Strehl Ratio @ %1.2f um: %1.3f"%(self.lim*1e6,self.strehl_ratio(PSFle)))
                
        elif self.do_psf_le:
            PSFleprof = ndimage.mean(PSFle.get(), labels=self.Rflabel, index=self.Rfidx)
            srle = self.strehl_ratio(PSFle.get())
            print("Long-exposure PSF Strehl Ratio @ %1.2f um: %1.3f"%(self.lim*1e6,srle))
            
            
        if self.do_psf_le and self.VISU:
            self.show_psf_and_profile(PSFle.get(), PSFleprof, im_display_size=1000,clim=[-5,0])
        if self.coro_psf and self.VISU:
            self.show_psf_and_profile(PSFcle.get(), PSFcleprof, im_display_size=1000)   
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                self.show_psf_and_profile(PSFle.get(), PSFleprof, im_display_size=1000)
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                self.show_two_psfs(PSFle.get(), PSFcle.get(), im_display_size=1000, log=True, clim=[-8,0])
        ####### Compare PSFS with/without M1 aberrations
        if self.simul_M1polish and not self.simul_turb and self.VISU:
            print('PSFc does not exist, commented code')
            # self.show_two_psfs(PSFc.get(), PSFle.get(), im_display_size=1000, log=True, clim=[-9,-4])
        # Overplot radial profiles
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                fig, ax2 = plt.subplots()
                fig.set_size_inches( figsize)
                ax2.loglog(self.Rfvec, PSFleprof, 'b', linewidth=3, label='w/o PC')
                ax2.loglog(self.Rfvec, PSFcleprof, 'r', linewidth=3, label=' w/ PC')
                ax2.plot(np.array([self.sep_req, self.sep_req]), np.array([1e-10,1]), 'k--', linewidth=1.5)
                ax2.plot(np.array([1,1e4]), np.array([inten_query, inten_query]), 'k--', linewidth=1.5)
                #ax2.plot(Rfvec, PSF1prof, 'k--', label='DL')
                ax2.set_xlim([1,1e4])
                ax2.set_ylim([1e-8,1e-0])
                ax2.grid()
                ax2.tick_params(labelsize=12)
                ax2.set_xlabel('radial distance [mas]', fontsize=14)
                ax2.set_ylabel('normalized intensity', fontsize=14)
                ax2.legend(fontsize=12)
                
                
        tosave = dict(D=self.D, nPx=self.nPx, PupilArea=self.PupilArea, Tsim=self.Tsim, 
                      totSimulTime=self.totSimulTime, totSimulInit=self.totSimulInit,totSimulIter=self.totSimulIter,
                      simul_turb=self.simul_turb, simul_M1polish=self.simul_M1polish, 
                      simul_onaxis_AO=self.simul_onaxis_AO, simul_windload=self.simul_windload,
                      eval_perf_modal=self.eval_perf_modal, eval_perf_modal_turb=self.eval_perf_modal_turb, 
                      simul_truss_mask=self.simul_truss_mask)

        if self.simul_turb:
            tosave.update(dict( seeing=self.seeing, 
                               r0=self.r0, L0=self.L0, tau0=self.tau0, 
                               wind_speed=self.wind_speed, zen_angle=self.zen_angle, 
                               atm_fname=self.atm_fullname))
            if self.simul_variable_seeing:
                tosave.update(dict(r0_iter=self.r0_iter))

        if self.eval_perf_modal:#False
            tosave.update(dict(seg_aRes_gs_iter=seg_aRes_gs_iter))

        if self.eval_perf_modal_turb:#False
            tosave.update(dict(seg_aTur_gs_iter=seg_aTur_gs_iter))

        if self.simul_windload:
            print('windload case not fully implemented, code commented')
            # tosave.update(dict(wldir=wldir, wlfn=wlfn, wlstart=wlstartInit, remove_quasi_statics=remove_quasi_statics,
            #                 wlstart0=wlstart0, zen=zen, az=az, wl_wind_speed = wl_wind_speed, vents=vents,
            #                 wind_screen=wind_screen))

        if self.simul_onaxis_AO:
            tosave.update(dict(band=self.band, lwfs=self.gs.wavelength, 
                               mag=self.mag, e0=self.e0, nLenslet=self.nLenslet, 
                               M2_n_modes=self.M2_n_modes, IMfile=self.IMfnameFull,
                               pixelSize=self.pixelSize, tot_delay=self.tot_delay, 
                               SPPctrlInit=self.SPPctrlInit, SPP2ndChInit=self.SPP2ndChInit, 
                               SPP2ndCh_Ts=self.SPP2ndCh_Ts, AOinit=AOinit))

            if self.omgi:
                tosave.update(dict(gAO=self.optgain_iter, omgi_lpf=self.omgi_lpf, omgi_ticks=self.omgi_ticks))
            else:
                tosave.update(dict(gAO=gAO.get()))         

            if self.save_telemetry:
                tosave.update(dict(a_M2_iter=self.a_M2_iter, da_M2_iter=self.da_M2_iter, 
                                   wfe_gs_iter=self.wfe_gs_iter, spp_gs_iter=self.spp_gs_iter, 
                                   seg_wfe_gs_iter=self.seg_wfe_gs_iter,wfs_meas_iter=wfs_meas_iter, 
                                   wfgrad_iter=wfgrad_iter, seg_wfgrad_iter=seg_wfgrad_iter, 
                                   opdr_spp_iter=self.spp_gs_iter, a_OLCL_iter=self.a_OLCL_iter))
                    # opdr_spp_iter=self.crazy_spp_iter))

            # Pyramid WFS parameters
            tosave.update(dict(pyr_modulation=self.pyr_modulation, pyr_angle=self.pyr_angle, 
                               pyr_thr=self.pyr_thr, percent_extra_subaps=self.percent_extra_subaps,
                               throughput=self.throughput, pyr_fov=self.pyr_fov, RONval=self.RONval,
                               excess_noise=self.excess_noise, simul_wfs_noise=self.simul_wfs_noise))

            if self.ogtl_simul:
                tosave.update(dict(ogtl_probe_params=self.ogtl_probe_params, 
                                   ogtl_detrend_deg=self.ogtl_detrend_deg, 
                                   ogtl_gain=self.ogtl_gain,
                                   probe_radord=self.probe_radord,
                                   ogtl_ogc_outer_iter=self.ogtl_ogc_outer_iter,
                                   ogtl_ogc_centr_iter=self.ogtl_ogc_centr_iter,
                                   ogtl_radord_deg=self.ogtl_radord_deg, 
                                   ogtl_Ts=[self.ogtl_Ts,self.ogtl_Ts], 
                                   probeSigInit=probeSigInit,
                                   ogtl_reconfig_iter=self.ogtl_reconfig_iter, 
                                   ogtl_ticks=self.ogtl_ticks, 
                                   ogtl_ogeff_probes_iter=self.ogtl_ogeff_probes_iter,
                                   ogtl_ogc_probes_iter=self.ogtl_ogc_probes_iter))
                
        #save second channel data
        notsaveKeys = ['chan1wl','debugframe','ogc','VISU', 'simul_truss_mask',
                       'wfs','gs','ph_fda_on','D2m','R2m', 'piston_estimate', 
                       'forCorrection', 'gs2phase', 'gs2inpup' ]
        dic2save = { x : self.chan2.__dict__[x] for x in self.chan2.__dict__.keys() if x not in notsaveKeys}
        tmpkeys = copy.deepcopy(dic2save)
        for k in tmpkeys.keys():
            dic2save['chan2'+k] = dic2save[k]
            del dic2save[k]
        tosave.update(dic2save)
        
        if self.secondChannelType=='lift':
            dic2save = { x : self.doubleChan2[1].__dict__[x] for x in self.doubleChan2[1].__dict__.keys() if x not in notsaveKeys}
            tmpkeys = copy.deepcopy(dic2save)
            for k in tmpkeys.keys():
                dic2save['chan2_sec'+k] = dic2save[k]
                del dic2save[k]
            tosave.update(dic2save)
        
        

        if self.do_psf_le:
            tosave.update(dict(srle=srle, lim=self.lim))
            if self.save_telemetry:
                tosave.update(dict(sr_iter=sr_iter, im_centr_iter=im_centr_iter))
                
        # filename='./savedir'+str(self.GPUnum)+'/simul_results.npz'
        filename = self.tempFolder+'/simul_results.npz'
        np.savez_compressed(filename, **tosave)
        
        #PSF stuff:
        if self.coro_psf or self.do_psf_le:
            tosavePSF = dict(lim=self.lim, norm_pup=self.norm_pup, fp_pixscale=self.fp_pixscale, npad=self.npad, coro_psf=self.coro_psf, centr_ref=self.centr_ref)
            if self.coro_psf:
                tosavePSF.update(dict(PSFcle=PSFcle.get(), do_wo_coro_too=self.do_wo_coro_too))
                if self.do_wo_coro_too:
                    tosavePSF.update(dict(PSFle=PSFle.get()))    
            elif self.do_psf_le:
                tosavePSF.update(dict(PSFle=PSFle.get()))

        if self.coro_psf or self.do_psf_le:
            filename=self.tempFolder+'/psf_results.npz'
            np.savez_compressed(filename, **tosavePSF)
        self.tid.toc()
        self.housekeep_saveTime = self.tid.elapsedTime*1e-3
    
    # just copy the files in ./savedir to a newly generated TN
    def saveTN(self,inputFolder,paramaterFileName):
        self.tid.tic()
        # tnString = datetime.today().strftime('%Y%m%d_%H%M%S')
        # tnpath = os.path.join(self.TN_dir)
        os.makedirs(self.TN_dir,exist_ok=True)
        shutil.move(self.tempFolder, self.TN_dir)
        # os.system('cp ./savedir' + str(self.GPUnum)+ '/* ' + tnpath )  
        print('\n saved in :{}'.format(self.TN_dir+self.tnString))
        self.tid.toc()
        self.housekeep_mvSave = self.tid.elapsedTime*1e-3
        return self.TN_dir+self.tnString

        
    def loadTN(self, tnString):
        tnpath1 = os.path.join(self.TN_dir, tnString, 'simul_results.npz')
        with np.load(tnpath1, allow_pickle=True) as data:
            self.D = data['D']
            self.nPx = data['nPx']
            self.PupilArea = data['PupilArea']
            self.Tsim = data['Tsim']
            self.totSimulTime = data['totSimulTime']
            self.totSimulInit = data['totSimulInit']
            self.simul_turb = data['simul_turb']
            self.simul_M1polish = data['simul_M1polish']
            self.simul_onaxis_AO = data['simul_onaxis_AO']
            self.simul_windload = data['simul_windload']
            self.eval_perf_modal = data['eval_perf_modal']
            self.eval_perf_modal_turb = data['eval_perf_modal_turb']
            self.simul_truss_mask = data['simul_truss_mask']            
            self.totSimulIter = data['totSimulIter']
            if self.simul_turb:
                self.seeing = data['seeing']
                self.r0 = data['r0']
                self.L0 = data['L0']
                self.tau0 = data['tau0']
                self.wind_speed = data['wind_speed']
                self.zen_angle = data['zen_angle']
                if 'atm_fullname' in data:
                    self.atm_fullname = data['atm_fullname']
                if 'r0_iter' in data:
                    self.r0_iter = data['r0_iter']
            if 'seg_aRes_gs_iter' in data:
                self.seg_aRes_gs_iter = data['seg_aRes_gs_iter']
            if 'seg_aTur_gs_iter' in data:
                self.seg_aTur_gs_iter = data['seg_aTur_gs_iter']
            if 'wldir' in data:
                self.wldir = data['wldir']
                self.wlfn = data['wlfn']
                self.wlstartInit = data['wlstartInit']
                self.remove_quasi_statics = data['remove_quasi_statics']
                self.wlstart0 = data['wlstart0']
                self.zen = data['zen']
                self.az = data['az']
                self.wl_wind_speed = data['wl_wind_speed']
                self.vents = data['vents']
                self.wind_screen = data['wind_screen']
            
            self.band = data['band']
            # cannot rewrite this
#            self.gs.wavelength = data['lwfs']
            self.mag = data['mag']
            self.e0 = data['e0']
            self.nLenslet = data['nLenslet']
            self.M2_n_modes = data['M2_n_modes']
            self.IMfnameFull = data['IMfile']
            self.pixelSize = data['pixelSize']
            self.gAO = cp.array( data['gAO'] )
            self.tot_delay = data['tot_delay']
            
            self.SPP2ndCh_Ts = data['SPP2ndCh_Ts']
            self.SPPctrlInit = data['SPPctrlInit']
            self.SPP2ndChInit = data['SPP2ndChInit']
                        
            self.AOinit = data['AOinit']
            self.a_M2_iter = data['a_M2_iter']
            # self.da_M2_iter = data['da_M2_iter']
            self.a_OLCL_iter = data['a_OLCL_iter']
            self.wfe_gs_iter = data['wfe_gs_iter']
            self.spp_gs_iter = data['spp_gs_iter']
            self.seg_wfe_gs_iter = data['seg_wfe_gs_iter']
            # self.wfs_meas_iter = data['wfs_meas_iter']
            self.wfgrad_iter = data['wfgrad_iter']
            self.seg_wfgrad_iter = data['seg_wfgrad_iter']

            if 'opdr_spp_iter' in data:
                self.crazy_spp_iter = data['opdr_spp_iter']
            self.pyr_modulation = data['pyr_modulation']
            self.pyr_angle = data['pyr_angle']
            self.pyr_thr = data['pyr_thr']
            self.percent_extra_subaps = data['percent_extra_subaps']
            self.throughput = data['throughput']
            self.pyr_fov = data['pyr_fov']
            self.RONval = data['RONval']
            self.excess_noise = data['excess_noise']
            self.simul_wfs_noise = data['simul_wfs_noise']
            self.ogtl_probe_params = data['ogtl_probe_params']
            self.ogtl_detrend_deg = data['ogtl_detrend_deg']
            self.ogtl_ogc_outer_iter = data['ogtl_ogc_outer_iter']
            self.ogtl_ogc_centr_iter = data['ogtl_ogc_centr_iter']
            self.ogtl_radord_deg = data['ogtl_radord_deg']
            self.ogtl_Ts = data['ogtl_Ts']
            self.probeSigInit = data['probeSigInit']
            self.ogtl_reconfig_iter = data['ogtl_reconfig_iter']
            self.ogtl_ticks = data['ogtl_ticks']
            self.ogtl_ogeff_probes_iter = data['ogtl_ogeff_probes_iter']
            self.ogtl_ogc_probes_iter = data['ogtl_ogc_probes_iter']
            self.optgain_iter = data['gAO']
            self.omgi_ticks = data['omgi_ticks']
            self.chan2.pistEstTime = data['chan2pistEstTime']
            self.chan2.pistEstList = data['chan2pistEstList']
            self.chan2.ogc_iter = data['chan2ogc_iter']
            self.chan2.correctionList = data['chan2correctionList' ]
            self.chan2.gs2segPistErr = data['chan2gs2segPistErr']
            self.chan2.gs2wfe = data['chan2gs2wfe']
            if self.secondChannelType == 'lift':
                self.doubleChan2[1].pistEstTime = data['chan2_secpistEstTime']
                self.doubleChan2[1].pistEstList = data['chan2_secpistEstList']
                self.doubleChan2[1].ogc_iter = data['chan2_secogc_iter']
                self.doubleChan2[1].gs2segPistErr = data['chan2_secgs2segPistErr']
                self.doubleChan2[1].gs2wfe = data['chan2_secgs2wfe']

            if 'srle' in data:
                self.srle = data['srle']
                self.lim = data['lim']
            if 'sr_iter' in data:
                self.sr_iter = data['sr_iter']
                self.im_centr_iter = data['im_centr_iter']

        tnpath2 = os.path.join(self.TN_dir, tnString, 'psf_results.npz')
        with np.load(tnpath1) as data:
            if 'lim' in data:
                self.lim = data['lim']
                self.norm_pup = data['norm_pup']
                self.fp_pixscale = data['fp_pixscale']
                self.npad = data['npad']
                self.coro_psf = data['coro_psf']
                self.centr_ref = data['centr_ref']
                self.PSFcle = cp.array(data['PSFcle'])
                self.do_wo_coro_too = data['do_wo_coro_too']
                self.PSFle = cp.array(data['PSFle'])

    
    def analyseSimulation(self,ogtl_iter_ogc = 0,figsize = (15,5)):

        
        self.totSimulInit =500
        
        self.propagatePyramid()
        self.M2_KL_modes()
        self.getOrCalibrateIM()
#        self.doOTGLsimul()
        
        nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
        self.n_mode = nall//self.nseg

        # init        
        niter = self.totSimulIter-self.totSimulInit
        timeVec = np.arange(self.totSimulIter)*self.Tsim
                        
        if self.simul_onaxis_AO:
            #-- Ticks will be set at multiples of WFS central wavelength
            lwfs = np.round(self.gs.wavelength*1e9)

        if self.SPP2ndCh_Ts < niter:   
            self.SPP2ndCh_sampling_time = self.Tsim*self.SPP2ndCh_Ts
            #-- Ticks will be set at multiples of AGWS sampling Time
            xticksrange = np.round(np.array([0,timeVec[-1]]) / self.SPP2ndCh_sampling_time)
            self.SPP2ndCH_ticks = np.arange(xticksrange[0],xticksrange[1]+1)*self.SPP2ndCh_sampling_time


        # WFE and SPP RMS  
        fig, (ax1,ax2) = plt.subplots(num = 'WFE and SPP RMS', ncols=2)
        fig.set_size_inches(figsize)

        #### on-axis WFE vs. iteration number
        ax1.plot(self.wfe_gs_iter*1e9)
        ax1.grid()
        ax1.set_xlabel('Iteration number')
        ax1.set_ylabel('WF RMS [nm]')
        # ax1.tick_params(labelsize=15)

        #### on-axis segment phase piston (SPP) RMS vs. iteration number
        ax2.plot(np.std(self.spp_gs_iter,0)*1e9, label='OPD')
        # ax2.plot(np.std(self.crazy_spp_iter,0)*1e9, label='OPDr')
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel('Iteration number')
        ax2.set_ylabel('SPP [nm WF]')
        # ax2.tick_params(labelsize=15)
        #ax2.set_xlim([40,60])
        plt.tight_layout()
        plt.show()

        wfe_final = np.mean(self.wfe_gs_iter[self.totSimulInit:])
        spp_final  = np.mean((np.std(self.spp_gs_iter,   0))[self.totSimulInit:])
        # spp_final2 = np.mean((np.std(self.crazy_spp_iter,0))[self.totSimulInit:])
        SR_marechal_final = np.exp(-(wfe_final*2*np.pi/self.lim)**2)

        print ('      Final WFE [nm RMS]: %3.1f'%(wfe_final*1e9))
        print ('      Final SPP RMS [nm]: %3.1f'%(spp_final*1e9))
        # print ('      Final SPP RMS [nm]: %3.1f'%(spp_final2*1e9))
        print ('Equivalent SR @ %2.2f um: %1.4f'%(self.lim*1e6,SR_marechal_final))


        # WFE and SPP
        #-- Differential segment piston w.r.t. central segment
        diff_spp_iter =self.spp_gs_iter - self.spp_gs_iter[6,:]
        # diff_spp_iter2 =self.crazy_spp_iter - self.crazy_spp_iter[6,:]
        #### on-axis segment WFE vs. iteration number
        fig, (ax1,ax2) = plt.subplots(num='piston WFE',ncols=2)
        fig.set_size_inches(figsize)
        ax1.plot(timeVec,self.seg_wfe_gs_iter.T*1e9)#, '-+')
        ax1.grid()
        ax1.set_xlabel('elapsed time [s]')
        ax1.set_ylabel('WF RMS [nm]')
        # ax1.tick_params(labelsize=13)
        #ax1.set_ylim([0,2000])
        #ax1.set_xlim([0,20])
        #### on-axis SPP vs. iteration number
        ax2.plot(timeVec,diff_spp_iter.T*1e9)#, '-+')
        ax2.grid()
        ax2.set_xlabel('elapsed time [s]')
        ax2.set_ylabel('SPP [nm WF]')
        # ax2.tick_params(labelsize=13)
        #ax2.set_ylim([-4000,4000])
        #ax2.set_xlim([0,20])

        lwfs = np.round(self.gs.wavelength*1e9)
        yticksrange = np.round((np.min(diff_spp_iter*1e9), np.max(diff_spp_iter*1e9)) / lwfs)
        lwfs_ticks = np.arange(yticksrange[0]-1,yticksrange[1]+1)*lwfs
        ax2.set_yticks(lwfs_ticks) # yTicks will be set at multiples of the WFS central wavelength"""
        #ax2.set_xticks(SPP2ndCH_ticks)
        #ax2.set_xlim([0,2])
        #ax1.set_xlim([0,2])
        plt.tight_layout()
        plt.show()
        
        
        self.n_mode = 500
        
        daM2mv  = np.zeros((self.n_mode,self.nseg))
        daM2var = np.zeros((self.n_mode,self.nseg))
        for segId in range(self.nseg):
            daM2mv[:,segId] = np.mean(self.da_M2_iter[segId,:,self.totSimulInit:], axis=1)
            daM2var[:,segId] = np.var(self.da_M2_iter[segId,:,self.totSimulInit:], axis=1)
        aM2mv  = np.zeros((self.n_mode,self.nseg))
        aM2var = np.zeros((self.n_mode,self.nseg))
        for segId in range(self.nseg):
            aM2mv[:,segId] = np.mean(self.a_M2_iter[segId,:,self.totSimulInit:], axis=1)
            aM2var[:,segId] = np.var(self.a_M2_iter[segId,:,self.totSimulInit:], axis=1)
        
        # M2 commands modal plot
        f1, ax = plt.subplots(num='command modal plot',ncols=3, nrows=3,sharex=True)
        f1.set_size_inches(20,15)
        for jj in range(self.nseg):
            thisax = (ax.ravel())[jj]
            thisax.loglog(np.arange(self.n_mode)+1,aM2var[:,jj]*(1e9**2))
            thisax.loglog(np.arange(self.n_mode)+1,daM2var[:,jj]*(1e9**2))
            thisax.grid()
            thisax.set_title('Segment #%d'%(jj+1))
            # thisax.set_xlabel('mode number + 1')
            # thisax.set_ylabel('variance [nm$^2$ WF]', size='large')
            #thisax.set_ylim([-30,100])
        for k in range(self.nseg,9):
            (ax.ravel())[k].axis('off')
            
        f1.text(0.5, 0.04, 'mode number + 1', ha='center')
        f1.text(0.04, 0.5, 'variance [nm$^2$ WF]', va='center', rotation='vertical')
        # plt.tight_layout()
        plt.show()

        # Optical Gain
        fig, (ax,ax2) = plt.subplots(num= 'optical gain probe signals', ncols=2)
        fig.set_size_inches(figsize)
        ax.plot(self.ogtl_ticks, np.array(self.ogtl_ogeff_probes_iter), drawstyle='steps-post')
        label = ['KL#%d'%kk['mode'] for kk in self.ogtl_probe_params]
        #ax.set_xticks(OGTL_ticks[1:-1:4])
        ax.legend(label,ncol=2)
        ax.grid()
        ax.set_xlabel('elapsed time [s]', size='large')
        ax.set_ylabel('effective OG $\equiv D(\Delta z) / D(z)$', size='large')
        #ax.set_xlim([0,2.05])

        imm = ax2.plot(self.ogtl_ticks, 1-np.array(self.ogtl_ogeff_probes_iter), drawstyle='steps-post')
        ax2.grid()
        ax2.set_xlabel('elapsed time [s]', size='large')
        ax2.set_ylabel('1 - effective OG', size='large')
        plt.tight_layout()
        plt.show()

        fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(num='optical gain extrapolation', ncols=2, nrows=2)
        fig.set_size_inches(14,11)
        ax.plot(self.ogtl_ticks, self.ogtl_ogc_outer_iter, drawstyle='steps-post')
        #ax.set_xticks(OGTL_ticks[1:-1:4])
        ax.grid()
        # ax.set_xlabel('elapsed time [s]', size='large')
        ax.set_ylabel('OGC coefficient $\equiv$ K', size='large')
        ax.set_title('outer segments', fontsize=13)
        #ax.set_xlim([0,2.05])

        radord_data = dict(np.load(self.dir_calib+'ASM_fittedKLs_S7OC04184_675kls_radord.npz'))
        self.radord_all_outer = radord_data['outer_radord'][0:self.n_mode]
        self.radord_all_centr = radord_data['centr_radord'][0:self.n_mode]
        outer_max_radord = np.max(self.radord_all_outer)
        self.probe_radord = np.round(np.arange(6)*outer_max_radord/6+3).astype('int')
        self.probe_outer = np.zeros(6, dtype='int')
        for jj in range(6):
            self.probe_outer[jj] = np.argwhere(radord_data['outer_radord'] == self.probe_radord[jj])[1]
        # self.probe_outer = np.insert(self.probe_outer, 0, 0)
        # self.probe_radord = np.insert(self.probe_radord, 0, 1)

        ax2.plot(self.radord_all_outer, self.ogtl_ogc_outer_iter[ogtl_iter_ogc], '+')
        ax2.plot(self.probe_radord, self.ogtl_ogc_probes_iter[ogtl_iter_ogc], 'o')
        ax2.set_xlabel('KL group', size='large')
        # ax2.set_ylabel('OGC coefficient', size='large')

        ax2.grid()
        ax2.legend(['OGTL iteration %d'%ogtl_iter_ogc])

        ax3.plot(self.ogtl_ticks, np.array(self.ogtl_ogc_centr_iter), drawstyle='steps-post')
        #a3x.set_xticks(OGTL_ticks[1:-1:4])
        ax3.grid()
        ax3.set_xlabel('elapsed time [s]')
        ax3.set_ylabel('OGC coefficient')
        ax3.set_title('central segments')
        #ax3.legend()
        #ax3.set_xlim([0,2.05])

        ax4.plot(self.radord_all_centr, self.ogtl_ogc_centr_iter[ogtl_iter_ogc],'+')
        ax4.plot(self.probe_radord, self.ogtl_ogc_probes_iter[ogtl_iter_ogc], 'o')
        ax4.set_xlabel('KL group')
        ax4.set_ylabel('OGC coefficient')
        #ax4.set_xlim([480,500])
        ax4.grid()
        ax4.legend(['OGTL iteration %d'%ogtl_iter_ogc])
        plt.tight_layout()
        
        plt.figure('measurement of the wavefront sensor')
        plt.clf()
        y = self.chan2.pistEstList
        x = self.chan2.pistEstTime
        # y = [a*10**6 for b ib ]
        y = np.array(y)*10**6
        plt.plot(x,y ,'+-' )
        plt.title('channel 2 measurement')
        plt.xticks(np.arange(-50,int(self.totSimulTime/ self.Tsim),150),rotation = 90)
        plt.xlim([0,int(self.totSimulTime/ self.Tsim)])
        
        plt.grid()
        plt.legend(['S1','S2','S3','S4','S5', 'S6', 'S0' ])
        plt.xlabel('iteration number' )
        plt.ylabel('segment piston detected [μm]')
        plt.tight_layout()
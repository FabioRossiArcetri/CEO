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

#----- Math and Scientific Computing
import math
import numpy as np
import cupy as cp
from scipy import ndimage
from scipy import signal 
from scipy.interpolate import CubicSpline

#----- Visualization
import matplotlib.pyplot as plt

# .ini file parsing
from configparser import ConfigParser
from datetime import datetime

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

class NGAO(object):
    def __init__(self, path, parametersFile):
        parser = ConfigParser()
        parser.read(path + parametersFile + '.ini')        
        print(path + parametersFile + '.ini')
        self.GPUnum = eval(parser.get('general', 'GPUnum'))
        self.dir_calib = eval(parser.get('general', 'dir_calib'))
        self.atm_dir = eval(parser.get('general', 'atm_dir'))
        self.TN_dir = eval(parser.get('general', 'TN_dir'))
        self.simul_onaxis_AO = eval(parser.get('general', 'simul_onaxis_AO'))
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
        self.M2_modes_scramble = eval(parser.get('general', 'M2_modes_scramble'))
        self.save_telemetry = eval(parser.get('general', 'save_telemetry'))        
        self.do_psf_le = eval(parser.get('general', 'do_psf_le'))        
        self.do_crazy_spp = eval(parser.get('general', 'do_crazy_spp'))        
        self.fake_fast_spp_convergence = eval(parser.get('general', 'fake_fast_spp_convergence'))
        self.save_phres_decimated = eval(parser.get('general', 'save_phres_decimated'))
        self.save_psfse_decimated = eval(parser.get('general', 'save_psfse_decimated'))        

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

        if self.simul_onaxis_AO:

            self.band = eval(parser.get('pyramid', 'band'))
            self.mag = eval(parser.get('pyramid', 'mag'))        
            # ph/s in R+I band over the GMT pupil (correction factor applied)
            self.e0 = 9.00e12 /368. * self.PupilArea
            self.bkgd_mag = eval(parser.get('pyramid', 'bkgd_mag'))
            self.nLenslet = eval(parser.get('pyramid', 'nLenslet'))
            self.nPx = eval(parser.get('pyramid', 'nPx_fact')) * self.nLenslet
            self.pixelSize = self.D/self.nPx
            self.pyr_separation = eval(parser.get('pyramid', 'pyr_separation'))
            self.pyr_modulation = eval(parser.get('pyramid', 'pyr_modulation'))
            self.pyr_angle = eval(parser.get('pyramid', 'pyr_angle'))
            self.pyr_thr = eval(parser.get('pyramid', 'pyr_thr'))
            self.percent_extra_subaps = eval(parser.get('pyramid', 'percent_extra_subaps'))
            self.throughput = self.tel_throughput * eval(parser.get('pyramid', 'throughput_factor'))
            self.pyr_fov = eval(parser.get('pyramid', 'pyr_fov'))
            self.RONval = eval(parser.get('pyramid', 'RONval'))
            
            # TODO: here?
            self.Zstroke = eval(parser.get('pyramid', 'Zstroke'))
            self.z_first_mode = eval(parser.get('pyramid', 'z_first_mode'))
            self.last_mode = eval(parser.get('pyramid', 'last_mode'))
            
            self.seg_pist_sig_masked = eval(parser.get('pyramid', 'seg_pist_sig_masked'))
            self.seg_sig_masked = eval(parser.get('pyramid', 'seg_sig_masked'))
            self.remove_seg_piston = eval(parser.get('pyramid', 'remove_seg_piston'))

            self.rec_type = eval(parser.get('pyramid', 'rec_type'))
            self.ao_thr = eval(parser.get('pyramid', 'ao_thr'))
            
            self.spp_rec_type = eval(parser.get('pyramid', 'spp_rec_type'))

            self.excess_noise = eval(parser.get('pyramid', 'excess_noise'))

            
            self.PSstroke = eval(parser.get('pistonSensor', 'PSstroke'))
            self.wvl_fraction = eval(parser.get('pistonSensor', 'wvl_fraction'))

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
            
                        
        if self.simul_turb:
            #TODOFR: Put r0 as parameter and derive the seeing value
            self.wind_scale = eval(parser.get('turbolence', 'wind_scale'))
            self.zen_angle = eval(parser.get('turbolence', 'zen_angle'))
            self.seeing = eval(parser.get('turbolence', 'seeing'))
            self.L0 = eval(parser.get('turbolence', 'L0'))
            self.altitude = np.array(eval(parser.get('turbolence', 'altitude')))
            self.xi0 = np.array(eval(parser.get('turbolence', 'xi0')))
            self.r0a  = 0.9759 * 500e-9/(self.seeing * ceo.constants.ARCSEC2RAD)  # Fried parameter at zenith [m]
            self.r0   = self.r0a * np.cos( self.zen_angle*np.pi/180 )**(3./5.)
            self.wind_speed = self.wind_scale[0] * np.array(eval(parser.get('turbolence', 'wind_speed')))
            self.wind_direction = np.array(eval(parser.get('turbolence', 'wind_direction')))
            self.meanV = np.sum(self.wind_speed**(5.0/3.0)*self.xi0)**(3./5.)
            self.tau0 = 0.314*self.r0/self.meanV
            self.simul_variable_seeing = eval(parser.get('turbolence', 'simul_variable_seeing'))
                        

    def M2_KL_modes(self):
        
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
        self.Roc = self.gmt.M2_baffle / self.segmentD #TODOFR
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
            fig.set_size_inches(15,5)
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
        
    def propagatePyramid(self):
        
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
                fig.set_size_inches(15,5)
                imm1 = ax1.imshow( np.sum(self.wfs.indpup, axis=0, dtype='bool')[extr:-1-extr, extr:-1-extr] )
                imm2 = ax2.imshow(self.wfs.ccd_frame[extr:-1-extr, extr:-1-extr])
                fig.colorbar(imm2, ax=ax2)
                plt.show()
                

    def KL_modes_exit_pupil(self):
                
        #show segment KL modes in the exit pupil
        this_kl = 100
        amp = 10e-9
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


    def pyr_display_signals_base(self, sx, sy, title=None):
        sx2d = self.wfs.get_sx2d(this_sx=sx)
        sy2d = self.wfs.get_sy2d(this_sy=sy)
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(12,4)
        if title==None:
            title = ['Sx', 'Sy']
        ax1.set_title(title[0])
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        imm = ax1.imshow(sx2d, interpolation='None')#,origin='lower', vmin=-1, vmax=1)
        clb = fig.colorbar(imm, ax=ax1, format="%.4f")
        clb.ax.tick_params(labelsize=12)    
        ax2.set_title(title[1])
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        imm2 = ax2.imshow(sy2d, interpolation='None')#,origin='lower', vmin=-1, vmax=1)
        clb2 = fig.colorbar(imm2, ax=ax2, format="%.4f")  
        clb2.ax.tick_params(labelsize=12)    
        return (sx2d,sy2d)
            
    def pyr_display_signals(self, title=None):                
        ## Visualize reference slope vector in 2D (for a flat WF)
        if self.simul_onaxis_AO and self.VISU:
            sx,sy = self.wfs.get_ref_measurement(out_format='list')
            return self.pyr_display_signals_base(sx, sy)
    
    def getOrCalibrateIM(self):
    
        if self.simul_onaxis_AO==True:
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
                tosave = dict(D_M2=D_M2_MODES, first_mode=self.z_first_mode, Stroke=self.Zstroke)
                np.savez(self.IMfnameFull, **tosave)
            else: 
                print(u'Reading file: '+self.IMfnameFull)
                ftemp = np.load(self.IMfnameFull)
                self.D_M2_MODES = ftemp.f.D_M2
                ftemp.close()

            nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
            self.n_mode = nall//7
            print(u'AO WFS - M2 Segment Modal IM:')
            print(self.D_M2_MODES.shape)
            
    def extractIMRequestedModes(self):
        if self.simul_onaxis_AO==True:
            self.D_M2_split = []
            for kk in range(7):
                Dtemp = self.D_M2_MODES[:, kk*self.n_mode : self.n_mode*(kk+1)]
                self.D_M2_split.append(Dtemp[:, 0:self.last_mode-self.z_first_mode])
            self.D_M2_MODES = np.concatenate(self.D_M2_split[0:7], axis=1) 
            nall = (self.D_M2_MODES.shape)[1]  ## number of KL DoFs calibrated
            self.n_mode = nall // 7
            print('AO WFS - M2 Segment Modal IM:')
            print(self.D_M2_MODES.shape)
            # free memory
            del self.D_M2_split
            del Dtemp
            del self.last_mode
                        
    def applySignalMasks(self):
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
                self.D_AO = gmt.NGWS_apply_segment_piston_mask(self.D_AO, segpist_signal_mask['mask'])
                self.wfs.segpist_signal_mask = segpist_signal_mask
                print("Segment piston signal masks applied.")                
            if self.simul_onaxis_AO and self.seg_pist_sig_masked and self.seg_sig_masked and self.VISU:
                segmSigMaskAll = np.sum(self.wfs.segpist_signal_mask['mask'][0:6],axis=0)
                pistSigMaskAll = np.sum(self.wfs.segment_signal_mask['mask'], axis=0)
                (sx2dtemp, sy2dtemp) = self.pyr_display_signals_base(segmSigMaskAll, pistSigMaskAll, 
                                           title=['segment piston masks', 'segment masks'])

                
    def showIMSignalsPrep(self):        
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
        (sx_temp,sy_temp) = self.pyr_display_signals_base(*ll)
        
        
    def showIMSignals(self, segment, mode):
        if self.simul_onaxis_AO==True and self.VISU==True:
            this_segment = segment   # from 0 to 6. Central segment: 6
            this_kl = mode
            this_mode = self.n_mode*this_segment+this_kl
            sx = self.D_AO[0:self.wfs.n_sspp,this_mode] / 1e9
            sy = self.D_AO[self.wfs.n_sspp:,this_mode]  / 1e9
            sx2d, sy2d = self.pyr_display_signals_base(sx,sy)
            
    def removeCentralPiston(self):            
        if self.simul_onaxis_AO:
            # Remove segment piston:
            if self.remove_seg_piston:
                self.segpist_idx = self.n_mode*np.array([6])
                self.D_AO = np.delete(self.D_AO, self.segpist_idx, axis=1) 
                
                
    def doIMsvd(self):
        if self.simul_onaxis_AO and self.rec_type=='LS':
            print('Condition number: %f'%np.linalg.cond(self.D_AO))
            self.Uz, self.Sz, self.Vz =np.linalg.svd(self.D_AO)
            if self.VISU:
                fig, ax = plt.subplots()
                fig.set_size_inches(7,5)
                ax.semilogy(self.Sz/np.max(self.Sz), 'o-', )
                ax.grid()
                ax.tick_params(labelsize=14)
                ax.set_xlabel('eigenmode number', fontsize=14)
                ax.set_ylabel('normalized singular value', fontsize=14)
                
                
    def showEigenMode(self, this_eigen):
        
        if self.simul_onaxis_AO and self.rec_type=='LS' and self.VISU:
            eigenmodevec = np.copy(self.Vz[this_eigen,:])
            if self.remove_seg_piston == True:
                for idx in self.segpist_idx:
                    eigenmodevec = np.insert(eigenmodevec,idx,0)
            self.gmt.reset()
            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = np.ascontiguousarray(eigenmodevec.reshape((7,-1))) *1e-6
            self.gmt.M2.modes.update()
            self.gs.reset()
            self.gmt.propagate(self.gs)

            fig, (ax1,ax2) = plt.subplots(ncols=2)
            fig.set_size_inches(15,5)
            imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation='None',cmap='RdYlBu',origin='lower')
            clb = fig.colorbar(imm, ax=ax1)  #, fraction=0.012, pad=0.03,format="%.1f")
            clb.set_label('nm WF', fontsize=12)
            ax2.plot(eigenmodevec, '+--')
            tx=ax2.set_xlabel('KL mode number')
            tx=ax2.set_ylabel('coefficient [a.u.]')
            plt.show()


    def genelarizedIMInverse(self):
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

    def addRowsOfRemovedModes(self):
        if self.simul_onaxis_AO:
            if self.remove_seg_piston:
                for idx in self.segpist_idx:
                    self.R_AO = np.insert(self.R_AO, idx, 0, axis=0)
   
                print('AO SH WFS - M2 Segment Modal Rec:')
                print(self.R_AO.shape)

                self.R_AO = cp.asarray(self.R_AO)
                self.ntotmodes = self.R_AO.shape[0]
                
                
    def idealPistonSensor(self):
        if self.simul_onaxis_AO:    
            # idealized SPS initialization
            self.onps = ceo.IdealSegmentPistonSensor(self.D, self.nPx, segment='full')
            self.gs.reset()
            self.gmt.reset()
            self.gmt.propagate(self.gs)
            self.onps.calibrate(self.gs)  # computes reference vector
            #-----> Ideal SPS - M2 segment piston IM calibration
            print("KL0 - SPS IM:")
            self.D_M2_PSideal = self.gmt.calibrate(self.onps, self.gs, mirror="M2", mode="Karhunen-Loeve", stroke=self.PSstroke, first_mode=0, last_mode=1)
            # index to KL0 in the command vector (to update controller state with 2nd NGWS command)
            self.KL0_idx = self.n_mode*np.array([0,1,2,3,4,5])
            if self.VISU:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(7,5)
                imm = ax1.pcolor(self.D_M2_PSideal)
                ax1.grid()
                fig.colorbar(imm, ax=ax1)#, fraction=0.012) 
                
            self.R_M2_PSideal = np.linalg.pinv(self.D_M2_PSideal)
            self.SPP2ndCh_thr = self.gs.wavelength * self.wvl_fraction

            
    def probe_signal(self, fs, kk):
        for pp in self.ogtl_probe_params:
            self.ogtl_probe_vec[pp['seg'], pp['mode']] = pp['amp']*np.sin(2*np.pi*pp['freq']/fs*kk)


    def optical_gain_cl(self, probes_in,probes_out):    
        probe_in_fitcoef = [np.polyfit(self.ogtl_timevec,this_probe,self.ogtl_detrend_deg) for this_probe in probes_in]
        probe_in_fit = [poly_nomial(self.ogtl_timevec,this_coeff) for this_coeff in probe_in_fitcoef]
        probes_in = np.array([pr-prf for (pr,prf) in zip(probes_in,probe_in_fit)])
        A_in = 2*np.sqrt(np.mean(probes_in * self.cosFn, axis=1)**2 + np.mean(probes_in * self.sinFn, axis=1)**2)

        probe_out_fitcoef = [np.polyfit(self.ogtl_timevec,this_probe,self.ogtl_detrend_deg) for this_probe in probes_out]
        probe_out_fit = [poly_nomial(self.ogtl_timevec,this_coeff) for this_coeff in probe_out_fitcoef]
        probes_out = np.array([pr-prf for (pr,prf) in zip(probes_out,probe_out_fit)])
        A_out = 2*np.sqrt(np.mean(probes_out * self.cosFn, axis=1)**2 + np.mean(probes_out * self.sinFn, axis=1)**2)

        OG = A_out / A_in   

        for kk in range(len(self.seg_list)):
            print('  Optical Gain of mode S%d KL%d: %0.3f'%(self.seg_list[kk],self.mode_list[kk],OG[kk]))

        return OG
        
    def doOTGLsimul(self):
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
            print(self.probe_outer)
            print(self.probe_radord)
            
            # Define probe signal parameters [segment#, mode#, amplitude, frequency]
            self.ogtl_probe_params = []
            self.ogtl_probe_params.append(dict(seg=0, mode=self.probe_outer[0], amp=2.5e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=1, mode=self.probe_outer[1], amp=2.5e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=2, mode=self.probe_outer[2], amp=2.5e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=3, mode=self.probe_outer[3], amp=2.0e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=4, mode=self.probe_outer[4], amp=2.0e-9, freq=310))
            self.ogtl_probe_params.append(dict(seg=5, mode=self.probe_outer[5], amp=2.0e-9, freq=310))
            #self.ogtl_probe_params.append(dict(seg=6, mode=probe_centr, amp=2.5e-9, freq=310))

            self.ogtl_probe_vec = np.zeros((7,self.n_mode))

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
                fig.set_size_inches(20,5)
                imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('probe modes')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f", fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)
                plt.show()


    def update_r0(self, t):
        self.varseeing_iter = (self.seeing_max-self.seeing_init)*np.sin(2*np.pi*t/self.vs_T)+ self.seeing_init
        return 0.9759 * 500e-9/(self.varseeing_iter*ceo.constants.ARCSEC2RAD) 
    
    def simulTurbolence(self):        
        if self.simul_turb:
            print('       Mean wind speed : %2.1f m/s' % self.meanV)
            print('                  tau0 : %2.2f ms'%(self.tau0*1e3))
            print('    r0 @ 500nm @ %d deg: %2.1f cm'%(self.zen_angle,self.r0*1e2))
            print('seeing @ 500nm @ %d deg: %2.2f arcsec'%(self.zen_angle,0.9759*500e-9/self.r0*ceo.constants.RAD2ARCSEC))

            #atm_fname = 'gmtAtmosphere_median_1min.bin'
            #atm_fullname = os.path.normpath(os.path.join(atm_dir,atm_fname))
            self.atm_fullname=None

            self.atm = ceo.Atmosphere(self.r0,self.L0,len(self.altitude),self.altitude,self.xi0,self.wind_speed,self.wind_direction,
                             L=26,NXY_PUPIL=346,fov=0.0*ceo.constants.ARCMIN2RAD, filename=self.atm_fullname, duration=5.0)
                                 #duration=5.0, N_DURATION=6)
            
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
                (sx_ref_2d,sy_ref_2d) = self.pyr_display_signals_base(*ll)
                
                
                
    def modalPerfEval(self, this_mode=401):
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
            self.segKLmat = [np.zeros((self.npseg[segId], self.gmt.M2.modes.n_mode)) for segId in range(7) ]
            KLamp = 5e-9
            for jj in range(self.gmt.M2.modes.n_mode):
                self.gmt.reset()
                self.gmt.M2.modes.a[:, jj] = KLamp
                self.gmt.M2.modes.update()
                self.gs.reset()
                self.gmt.propagate(self.gs)
                for segId in range(7):
                    self.segKLmat[segId][:,jj] = np.ravel(self.gs.phase.host()-self.ph_fda_on*1e-9)[self.P[segId,:]] / (KLamp) #normalize to unitary RMS WF
            # Compute KL RMS (should be unitary, but ray tracing to exit pupil may change slightly that...)
            # We will use the RMS to identify "zero" modes (e.g. in central segment w.r.t. outer segments)
            klrms = [np.sqrt(np.sum(self.segKLmat[segId]**2, axis=0)/self.npseg[segId]) for segId in range(7)]
            self.n_valid_modes = [np.max(np.where(klrms[segId] > 0.5))+1 for segId in range(7)]

            for segId in range(7):
                if self.VISU:
                    plt.semilogx(np.arange(self.gmt.M2.modes.n_mode)+1,klrms[segId], '+--', label='seg%d'%(segId+1))
                    plt.legend()
                    plt.show()
                    
            self.inv_segKLmat = [np.linalg.pinv(self.segKLmat[segId][:,0:self.n_valid_modes[segId]]) for segId in range(7)]
            if self.VISU:
                # Show a particular mode on all segments, just for fun
                caca = np.full(self.nPx**2, np.nan)
                for segId in range(7):
                    caca[self.P[segId,:]] = self.segKLmat[segId][:,this_mode]
                plt.imshow(caca.reshape((self.nPx,self.nPx)), interpolation='None')#, origin='lower')
                plt.title('KL#%d'%this_mode)
                plt.colorbar()
            
            
    def showSignalsModeSegAmp(self, this_seg, this_kl, this_amp ):
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
            fig.set_size_inches(12,4)
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
            [sx2dtemp, sy2dtemp] = self.pyr_display_signals_base(sxtemp,sytemp)
            
            
    def islandMode(self, index):
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
                fig.set_size_inches(14,5)

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
            self.gmt.propagate(gs)

            if self.VISU:
                fig, (ax1,ax2) = plt.subplots(ncols=2)
                fig.set_size_inches(14,5)

                imm = ax1.imshow(gs.phase.host(units='nm')-ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('on-axis WF')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)
                ax1.set_xlim([nPx/2-300,nPx/2+300])
                ax1.set_ylim([nPx/2-300,nPx/2+300])

                ax2.semilogx(np.arange(first_kl,self.n_mode)+1,this_comm*1e9)
                ax2.grid()
                ax2.set_xlabel('segment KL number + 1')
                ax2.set_ylabel('coeff amp [nm RMS]')
                plt.show()
                
                
    def globalTTModes(self):
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


    def phaseAveraging(self):
        if self.do_Phase_integration:
            self.imgs = ceo.Source('H', magnitude=8, zenith=0.,azimuth=0., rays_box_size=D, 
                    rays_box_sampling=self.nPx, rays_origin=[0.0,0.0,25])
            self.imgs.rays.rot_angle = self.pyr_angle*np.pi/180
            
            
    #------------------- Complex Amplitude computation.
    #     Inputs: None. It uses current amplitude and phase of gs object
    #     Keywords: FT: if True, returns the FT of the complex amplitude
    def complex_amplitude(self, FT=False):
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
    def psf_se(self, W_ft, shifted=False, norm_factor=1):
        if shifted==True:
            return cp.fft.fftshift( cp.abs(W_ft)**2 ) / norm_factor
        else:
            return cp.abs(W_ft)**2 / norm_factor

    # ------------------ Perfect coronagraph PSF computation
    def perfect_coro_psf_se(self, shifted=False, norm_factor=1):
        W1_ft = self.complex_amplitude(FT=True)
        SR_se = cp.exp(cp.array(-(self.knumber*self.gs.phaseRms())**2))  # Marechal approximation
        Wc_ft = W1_ft - cp.sqrt(SR_se)*self.Rf
        if self.do_wo_coro_too:
            return self.psf_se(W1_ft, shifted=shifted, norm_factor=norm_factor), self.psf_se(Wc_ft, shifted=shifted, norm_factor=norm_factor)
        else:
            return self.psf_se(Wc_ft, shifted=shifted, norm_factor=norm_factor) 


    #----------------- Computes the image displacement w.r.t to the reference position [in mas]
    def psf_centroid(self, myPSF, centr_ref):
        return np.array(ndimage.center_of_mass(myPSF) - centr_ref)*self.fp_pixscale

    #------------------ Estimates the SR (intensity at center of image from a normalized PSF)
    def strehl_ratio(self, myPSF, centr_ref):
        return myPSF[tuple(np.round(centr_ref).astype('int'))]

    def sr_and_centroid(self, myPSF):
        return self.strehl_ratio(myPSF), self.psf_centroid(myPSF)

    # ----------------- Estimate intensity at given separation
    def intensity_query(self, PSFprof, Rfvec, sep_query):
        ss = np.where(Rfvec > sep_query)[0][0]  # index of first distance value larger than sep_query
        int_req = PSFprof[ss] + \
            (PSFprof[ss]-PSFprof[ss-1]) / (Rfvec[ss]-Rfvec[ss-1]) * (sep_query-Rfvec[ss])
        return int_req
    
    #------------------- Funcdtion to show PSFs side by side
    #   im_display_size: +/- mas from center
    def show_two_psfs(self, myPSF1, myPSF2, im_display_size=150, log=True, clim=None):
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(14,5)

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
    def show_psf(self, myPSF, im_display_size=150, clim=[-8,0], fig=None,ax1=None):
        if ax1==None:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(7,5)

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
    def show_psf_and_profile(self, myPSF, myProf, im_display_size=150, clim=[-8,0]):
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(14,5)
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
        
    def showDL_PSF(self):
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
    
    
    def runAll(self):
        self.M2_KL_modes()
        self.propagatePyramid()
        self.KL_modes_exit_pupil()
        self.pyr_display_signals()
        self.getOrCalibrateIM()
        self.extractIMRequestedModes()
        self.applySignalMasks()
        self.showIMSignalsPrep()
        self.showIMSignals(6, 100)
        self.showIMSignals(0, 0)
        self.removeCentralPiston()
        self.doIMsvd()
        self.showEigenMode(202)
        self.genelarizedIMInverse()
        self.addRowsOfRemovedModes()
        self.idealPistonSensor()
        self.doOTGLsimul()
        self.simulTurbolence()
        self.modalPerfEval()
        self.showSignalsModeSegAmp(6, 20, -25e-9)
        self.islandMode(6)
        self.globalTTModes()
        self.phaseAveraging()
        self.showDL_PSF()
        self.closedLoopSimul()

            
    def closedLoopSimul(self):
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
                fig.set_size_inches(12,4)
                InitScr = gs.phase.host(units='nm')-self.ph_fda_on

                imm = ax1.imshow(InitScr, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
                ax1.set_title('on-axis WF')
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
                clb.set_label('$nm$ WF', fontsize=12)
                clb.ax.tick_params(labelsize=12)

                print('WF RMS: %.1f nm'%(self.gs.phaseRms()*1e9))
                print('SPP RMS: %.1f nm'%(np.std(self.gs.piston('segments'))*1e9))
                

        if self.seg_pist_scramble:
            self.pist_scramble_rms=10e-9   # in m RMS SURF

        if self.M2_modes_scramble:
            self.M2modes_scramble_rms = 5e-9  # in m RMS SURF        
            

        if self.seg_pist_scramble:
            # Generate piston scramble
            pistscramble  = np.random.normal(loc=0.0, scale=1, size=7)
            pistscramble *= pist_scramble_rms/np.std(pistscramble)
            pistscramble -= np.mean(pistscramble)
            pistscramble -= pistscramble[6]  # relative to central segment
            # Apply it to M2
            self.gmt.M2.motion_CS.origin[:,2] = pistscramble
            self.gmt.M2.motion_CS.update()

        if self.M2_modes_scramble:
            self.M2modes_scramble = np.random.normal(loc=0.0, scale=1, size=(7,self.n_mode))
            self.M2modes_scramble *= self.M2modes_scramble_rms/np.std(self.M2modes_scramble)
            self.M2modes_scramble[:,0] = 0
            #--- If you wanted to introduce just one mode over one segment....
            #self.M2modes_scramble = np.zeros((7,self.n_mode))
            #self.M2modes_scramble[0,1] = 100e-9
            self.gmt.M2.modes.a[:,self.z_first_mode:] = self.M2modes_scramble
            self.gmt.M2.modes.update()
        else:
            self.M2modes_scramble = np.zeros((7,self.n_mode))
            
            
        if self.VISU:
            self.gs.reset()
            self.gmt.propagate(self.gs)
            fig, ax1 = plt.subplots()
            fig.set_size_inches(12,4)
            imm = ax1.imshow(self.gs.phase.host(units='nm')-self.ph_fda_on, interpolation=None,cmap=plt.cm.gist_earth_r, origin='lower')#, vmin=-25, vmax=25)
            ax1.set_title('on-axis WF')
            ax1.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            clb = fig.colorbar(imm, ax=ax1, format="%.1f")#, fraction=0.012, pad=0.03)
            clb.set_label('$nm$ WF', fontsize=12)
            clb.ax.tick_params(labelsize=12)
            print(' WF RMS: %.1f nm'%(self.gs.phaseRms()*1e9))
            print('SPP RMS: %.1f nm'%(np.std(self.gs.piston('segments'))*1e9))

        AOinit = 0 #round(0.10/Tsim)               # close the AO loop

        if not self.simul_turb:
            #--- Timing when no turbulence is simulated
            Tsim= 1e-3                         # simulation sampling time [s]
            self.totSimulIter = 50                  # simulation duration [iterations]
            totSimulTime = self.totSimulIter*Tsim   # simulation duration [in seconds]
            self.totSimulInit = 30                  # start PSF integration at this iteration
            SPPctrlInit  = float('inf')        # start controlling SPP
            SPP2ndChInit = float('inf')        # Apply 2nd channel correction (with ideal SPS)
        else:
            #--- Timing when turbulence is simulated (longer exposures)
            Tsim = 1e-3                             # simulation sampling time [s]
            totSimulTime = self.totSimulTime        # simulation duration [in seconds]
            self.totSimulIter = round(totSimulTime/Tsim) # simulation duration [iterations]

            AOinit = 0 #round(0.10/Tsim)               # close the AO loop
            self.SPPctrlInit  = AOinit+round(0.05/Tsim)+1       # start controlling SPP
            self.SPP2ndChInit = AOinit+round(0.1/Tsim)        # start applying 2nd channel correction (with ideal SPS)
            self.SPP2ndCh_Ts  = round(0.150/Tsim)       # 2nd channel sampling time (in number of iterations)
            self.SPP2ndCh_count = 0

            self.totSimulInit = AOinit+1000         # start PSF integration at this iteration

            if self.do_Phase_integration:
                PhIntInit = AOinit+1000  # start averaging the phase maps
                PhInt_Ts = 200
                PhInt_count = 0

        if self.simul_windload:
            wlstartInit = 0      

        if self.ogtl_simul:
            probeSigInit = AOinit+self.SPP2ndChInit #SPPctrlInit+9          # Start injection probe signals on selected modes 
            ogtl_Ts_1 = 50e-3 # sampling during bootstrap [s]
            ogtl_sz = np.int(ogtl_Ts_1 /Tsim)
            ogtl_count = 0
            self.ogtl_detrend_deg = 3   # detrend time series
            nprobes = len(self.ogtl_probe_params)
            self.ogtl_radord_deg  = nprobes-1   # fit OG vs radial order curve 
            self.ogtl_gain = 0.3

            self.ogtl_timevec = np.arange(0,ogtl_sz)*Tsim
            self.cosFn = np.array([np.cos(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])
            self.sinFn = np.array([np.sin(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])

            ogtl_Ts_2 = 500e-3 #sampling after convergence [s]
            self.ogtl_reconfig_iter =probeSigInit + np.int(18*ogtl_Ts_1/Tsim)  # make sure this coincides with an OGTL iteration.

            self.ogtl_ogc_probes = np.ones(nprobes)
            self.ogtl_ogc_allmodes = np.ones(self.n_mode)
            OGC_all  = cp.ones((7,self.n_mode))
            self.ogtl_ogeff_probes_iter = []
            #ogtl_ogeff_iter = []
            #ogtl_ogc_iter = []
            self.ogtl_ogc_centr_iter = []
            self.ogtl_ogc_outer_iter = []
            self.ogtl_ogc_probes_iter = []
            self.ogtl_ticks = []
            ogtl_Ts = ogtl_Ts_1 # select sampling to be used first
        else:
            OGC_all  = cp.ones((7,self.n_mode))
            
        #---- Integrator's gain
        gAO = cp.ones((7,self.n_mode)) * 0.5
        #gAO[:,0:200] = 0.8
        #gAO[:,0] = 0.5  #SPP gain
        gAO = cp.reshape(gAO, (self.ntotmodes,1))
        g_pist = 1.0 # gain of regularized SPP command

        #---- delay simulation
        tot_delay = 2   # in number of frames
        delay  = tot_delay-1 
        Mdecal = cp.zeros((tot_delay, tot_delay))
        Mdecal[0, 0] = 1 
        if delay >= 0:
            Mdecal[0:delay,1:tot_delay] = cp.identity(delay) 
        Vinc   = cp.zeros((1,tot_delay))
        Vinc[0, 0]   = 1 
        
        comm_buffer = cp.zeros((self.ntotmodes,tot_delay))
        myAOest1    = cp.zeros((self.ntotmodes,1))

        pist_buffer = cp.zeros((7,tot_delay))

        #---- Time histories to save in results
        if self.save_telemetry or self.ogtl_simul:
            self.a_M2_iter   = np.zeros((7,self.n_mode,self.totSimulIter))   # integrated M2 modal commands
            self.da_M2_iter  = np.zeros((7,self.n_mode,self.totSimulIter))   # delta M2 modal commands
        if self.save_telemetry:
            a_OLCL_iter = np.zeros((7,self.n_mode,self.totSimulIter))
            wfs_meas_iter = np.zeros((self.wfs.get_measurement_size(),self.totSimulIter)) # pyramid WFS signals
            self.wfe_gs_iter = np.zeros(self.totSimulIter)              # on-axis WFE
            self.spp_gs_iter = np.zeros((7,self.totSimulIter))          # on-axis (differential) segment phase piston error
            self.seg_wfe_gs_iter = np.zeros((7,self.totSimulIter))      # on-axis WFE over each segment
            wfgrad_iter = np.zeros((2,self.totSimulIter))          # on-axis WF gradient (estimate of image motion) in mas
            seg_wfgrad_iter = np.zeros((14,self.totSimulIter))     # on-axis segment WF gradients [alpha_x, alpha_y]
        if self.save_telemetry and self.do_psf_le:
            sr_iter = np.zeros(self.totSimulIter)
            im_centr_iter = np.zeros((2,self.totSimulIter))
        if self.save_telemetry and self.simul_variable_seeing:
            self.r0_iter = np.zeros(self.totSimulIter)

        if self.do_crazy_spp:
            self.crazy_spp_iter = np.zeros((7,self.totSimulIter))

        if self.eval_perf_modal:
            seg_aRes_gs_iter = np.zeros((7,self.gmt.M2.modes.n_mode,self.totSimulIter)) # seg KL modal coefficients buffer     
            #ZresIter = np.zeros((nzern,self.totSimulIter))

        if self.eval_perf_modal_turb:
            seg_aTur_gs_iter = np.zeros((7,self.gmt.M2.modes.n_mode,self.totSimulIter)) # seg KL modal coefficients buffer 
            #ZturIter = np.zeros((nzern,self.totSimulIter))

        if self.simul_windload:
            pistjumps_com=np.zeros(7)

        if self.do_Phase_integration:
            PhInt_iter = []   #integrated phase maps
            PhInst_iter = []  #instantaneous phase realization
            
        Isliter = np.zeros((self.nisl,self.totSimulIter))
        Islwfe = np.zeros(self.totSimulIter)
        
        if self.save_phres_decimated:
            save_ph_Ts = 10e-3
            save_ph_niter = np.int(save_ph_Ts / Tsim)
            save_ph_count=0
            maskfile = './savedir%d/GMTmask.npz'%self.GPUnum
            np.savez_compressed(maskfile, GMTmask=self.GMTmask, nPx=self.nPx) 
            
            
        if self.save_psfse_decimated:
            save_psf_Ts = 10e-3
            save_psf_niter = np.int(save_psf_Ts / Tsim)
            save_psf_count=0
            psf_range_mas = np.array([-900, 900]) # +/- 1 arcsec
            psf_range_pix = np.rint(psf_range_mas/self.fp_pixscale + self.nPx*self.npad/2).astype(int)
            
            
        for jj in range(self.totSimulIter):
            self.tid.tic()
            self.gs.reset()

            #----- Update Turbulence --------------------------------------------
            if self.simul_turb:

                if self.simul_variable_seeing:
                    self.atm.r0 = self.update_r0(jj*Tsim)
                    if self.save_telemetry:
                        self.r0_iter[jj] = self.atm.r0
                self.atm.ray_tracing(self.gs, self.pixelSize,self.nPx,self.pixelSize,self.nPx, jj*Tsim)

                if self.do_Phase_integration:
                    if jj >= PhIntInit:
                        self.atm.ray_tracing(self.imgs, self.pixelSize,self.nPx,self.pixelSize,self.nPx, jj*Tsim)

                if self.eval_perf_modal_turb:
                    PhaseTur = np.squeeze(self.gs.wavefront.phase.host()) * self.GMTmask
                    PhaseTur[self.GMTmask] -= np.mean(PhaseTur[self.GMTmask])  # global piston removed
                    #ZturIter[:,jj] = inv_zmat @ PhaseTur[GMTmask]

                    for segId in range(7):
                        seg_aTur_gs_iter[segId,0:self.n_valid_modes[segId],jj] = np.dot(inv_segKLmat[segId], PhaseTur[P[segId,:]])

            #----- Introduce wind load effects as M1 and M2 RBM perturbations ---
#            if self.simul_windload:
#                self.gmt.M1.motion_CS.origin[:] = np.ascontiguousarray(wldata['Data']['M1 RBM'][jj*int(Tsim/Twl)+wlstartInit,:,:3])
#                self.gmt.M1.motion_CS.euler_angles[:] = np.ascontiguousarray(wldata['Data']['M1 RBM'][jj*int(Tsim/Twl)+wlstartInit,:,3:])
#                self.gmt.M1.motion_CS.update()

#                self.gmt.M2.motion_CS.origin[:] = np.ascontiguousarray(wldata['Data']['M2 RBM'][jj*int(Tsim/Twl)+wlstartInit,:,:3])
#                #gmt.M2.motion_CS.origin[:,2] += np.ascontiguousarray(pistjumps_com)
#                self.gmt.M2.motion_CS.euler_angles[:] = np.ascontiguousarray(wldata['Data']['M2 RBM'][jj*int(Tsim/Twl)+wlstartInit,:,3:])
#                self.gmt.M2.motion_CS.update()

            #----- Apply AO command, taking into account the simulated delay --------------------

            nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
            self.M2modes_command = cp.asnumpy(comm_buffer[0:nall,delay].reshape((7,-1)))
            if self.spp_rec_type=='MMSE' and jj <= self.SPPctrlInit:
                self.M2modes_command[:,0] += cp.asnumpy(pist_buffer[:,delay])

            #-> Apply probe signal command
            if self.ogtl_simul:
                if jj >= probeSigInit:
                    self.probe_signal(1/Tsim,jj-probeSigInit)
                    self.M2modes_command += self.ogtl_probe_vec
                    ogtl_count+=1

            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = self.M2modes_scramble - self.M2modes_command
            if self.save_telemetry or self.ogtl_simul:
                self.a_M2_iter[:,:,jj] = self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode]
            self.gmt.M2.modes.update()

            #----- On-axis WFS measurement ---------------------------------------------------
            self.gmt.propagate(self.gs)

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
                    PhaseMap = np.squeeze(gs.wavefront.phase.host())
                    PhaseMap[GMTmask] -= np.mean(PhaseMap[GMTmask])  # global piston removed
                    phres_temp = dict(phres = PhaseMap[GMTmask], timeStamp=jj*Tsim)
                    phres_fname='./savedir%d/phres_%04d'%(GPUnum,jj)
                    #savemat(phres_fname, phres_temp)
                    np.savez_compressed(phres_fname, **phres_temp)
                    save_ph_count = 0
                else:
                    save_ph_count+=1

            # Project residual phase maps onto segment KL modes
            if self.eval_perf_modal:
                PhaseRes = np.squeeze(gs.wavefront.phase.host()) * GMTmask
                PhaseRes[GMTmask] -= np.mean(PhaseRes[GMTmask])  # global piston removed
                #ZresIter[:,jj] = inv_zmat @ PhaseRes[GMTmask]

                for segId in range(7):
                    seg_aRes_gs_iter[segId,0:self.n_valid_modes[segId],jj] = np.dot(inv_segKLmat[segId], PhaseRes[P[segId,:]])  

            PhaseRes = np.squeeze(self.gs.wavefront.phase.host()) * self.GMTmask
            Isliter[:,jj] = self.inv_IslMat @ PhaseRes[self.P[6,:]]
            IslPh = self.IslMat @ Isliter[:,jj]
            Islwfe[jj] = np.std(IslPh)

            self.wfs.reset()
            if self.simul_wfs_noise:
                self.wfs.propagate(self.gs)
                self.wfs.camera.readOut(Tsim, self.RONval, 0, self.excess_noise)
                self.wfs.process()
            else:
                self.wfs.analyze(self.gs)   # This simulates a noise-less read-out.


            AOmeasvec = self.wfs.get_measurement()
            if self.save_telemetry:
                wfs_meas_iter[:,jj] = AOmeasvec

            #------ WF Reconstruction and command computation --------------------------------------------
            if self.rec_type=='LS':

                if jj >= AOinit:
                    myAOest1[:,0] = OGC_all.ravel() * (self.R_AO @ cp.asarray(AOmeasvec))


                if jj < self.SPPctrlInit:   # Do not control segment piston
                    myAOest1[self.n_mode*np.arange(7),0] = 0
                elif jj == self.SPPctrlInit:
                    #g_pist=0.0
                    comm_buffer[self.n_mode*np.arange(7),:] = pist_buffer

                if self.save_telemetry or self.ogtl_simul:
                    self.da_M2_iter[:,:,jj]  = cp.asnumpy(myAOest1.ravel()).reshape((7,self.n_mode))
                if self.save_telemetry:
                    a_OLCL_iter[:,:,jj] = self.M2modes_command + self.da_M2_iter[:,:,jj] 

                myAOest1 *= gAO

                if jj == self.SPP2ndChInit: # Apply a one-time 2nd NGWS channel segment piston correction
                    if self.fake_fast_spp_convergence:
                        self.onps.reset()
                        self.onps.analyze(self.gs)
                        pistjumps_est = self.onps.get_measurement()
                        pistjumps_est -= pistjumps_est[6]
                        pistjumps_com = np.dot(self.R_M2_PSideal,pistjumps_est)
                        comm_buffer[self.KL0_idx,:] += cp.asarray(pistjumps_com[0:6,np.newaxis])

                        #if simul_windload==False:
                        #    gmt.M2.motion_CS.origin[:,2] = np.ascontiguousarray(pistjumps_com)
                        #    gmt.M2.motion_CS.update()
                        self.onps.reset()
                elif jj > self.SPP2ndChInit:
                    if self.SPP2ndCh_count == self.SPP2ndCh_Ts-1:
                        self.onps.analyze(self.gs)
                        pistjumps_est = self.onps.get_measurement()
                        pistjumps_est -= pistjumps_est[6]
                        pistjumps_2nd = np.zeros(7)
                        pistjumps_2nd = np.where(pistjumps_est >  self.SPP2ndCh_thr,  self.gs.wavelength, pistjumps_2nd )
                        pistjumps_2nd = np.where(pistjumps_est < -self.SPP2ndCh_thr, -self.gs.wavelength, pistjumps_2nd )                               
                        pistjumps_com = np.dot(self.R_M2_PSideal,pistjumps_2nd)
                        #if simul_windload==False:
                        #    gmt.M2.motion_CS.origin[:,2] = np.ascontiguousarray(pistjumps_com)
                        #    gmt.M2.motion_CS.update()
                        comm_buffer[self.KL0_idx,:] += cp.asarray(pistjumps_com[0:6,np.newaxis])
                        self.SPP2ndCh_count=0
                        self.onps.reset()                
                    else:
                        self.onps.propagate(self.gs)
                        self.SPP2ndCh_count+=1

                comm_buffer =  cp.dot(comm_buffer, Mdecal) + cp.dot(myAOest1,Vinc)  # handle time delay

            if self.spp_rec_type=='MMSE':
                print('code removed')


            #----- Optical Gain Tracking Loop
            if self.ogtl_simul:
                if ogtl_count == ogtl_sz-1:
                    #---- Find effective optical gain in closed-loop operation
                    probes_in  = [ self.a_M2_iter[ss,mm,jj-ogtl_sz:jj] for (ss,mm) in zip(self.seg_list,self.mode_list)]
                    probes_out = [self.da_M2_iter[ss,mm,jj-ogtl_sz:jj] for (ss,mm) in zip(self.seg_list,self.mode_list)]
                    OGeff = self.optical_gain_cl(probes_in,probes_out)
                    self.ogtl_ogeff_probes_iter.append(OGeff.copy())

                    #---- Compute optical gain compensation (OGC) coefficients for probes
                    self.ogtl_ogc_probes += self.ogtl_gain * (1-OGeff)
                    self.ogtl_ogc_probes_iter.append(self.ogtl_ogc_probes.copy())

                    #---- Fit cubic spline to radial order vs OGC
                    ogc_spline = CubicSpline(self.probe_radord, self.ogtl_ogc_probes, bc_type='natural', extrapolate=True)
                    self.ogtl_ogc_allmodes_outer = ogc_spline(self.radord_all_outer)
                    self.ogtl_ogc_outer_iter.append(self.ogtl_ogc_allmodes_outer.copy())
                    self.ogtl_ogc_allmodes_centr = ogc_spline(self.radord_all_centr)
                    self.ogtl_ogc_centr_iter.append(self.ogtl_ogc_allmodes_centr.copy())

                    #---- Update K_OGC matrix
                    for this_seg in range(6):
                        OGC_all[this_seg,:] = cp.asarray(self.ogtl_ogc_allmodes_outer)
                    OGC_all[6,:] = cp.asarray(self.ogtl_ogc_allmodes_centr)

                    #---- Time stamp of OGTL correction
                    self.ogtl_ticks.append(jj*Tsim)
                    ogtl_count = 0

                #---- Increase OGTL sampling frequency after intial convergence, for more stability
                if jj == self.ogtl_reconfig_iter:
                    ogtl_Ts = ogtl_Ts_2 # in seconds
                    ogtl_sz = np.int(ogtl_Ts /Tsim)
                    self.ogtl_detrend_deg = 5
                    self.ogtl_timevec = np.arange(0,ogtl_sz)*Tsim
                    self.cosFn = np.array([np.cos(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])
                    self.sinFn = np.array([np.sin(2*np.pi*kk['freq']*self.ogtl_timevec) for kk in self.ogtl_probe_params])

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
                self.crazy_spp_iter[:,jj] = np.array([np.sum(OPDr[self.P[segId,:]])/self.npseg[segId] for segId in range(7)])


            #----- Compute short-exposure PSFs --------------------------------

            if self.coro_psf:
                if not do_wo_coro_too:
                    PSFcse = perfect_coro_psf_se(shifted=True, norm_factor=norm_pup)
                else:
                    PSFse, PSFcse = perfect_coro_psf_se(shifted=True, norm_factor=norm_pup, do_wo_coro_too=True)

                if jj == self.totSimulInit:
                    print('\nStart accumulating coro PSFs\n')
                    PSFcle = PSFcse.copy()
                    if do_wo_coro_too:
                        PSFle = PSFse.copy()
                elif jj > self.totSimulInit:
                    PSFcle += PSFcse
                    if do_wo_coro_too:
                        PSFle += PSFse

            elif self.do_psf_le:
                PSFse  = psf_se(complex_amplitude(FT=True), shifted=True, norm_factor=norm_pup)

                if jj == self.totSimulInit:
                    print('\nStart accumulating PSFs\n')
                    PSFle = PSFse.copy()
                elif jj > self.totSimulInit:
                    PSFle += PSFse

            if self.save_psfse_decimated:
                if save_psf_count == save_psf_niter-1:
                    psftemp = PSFse[psf_range_pix[0]:psf_range_pix[1],psf_range_pix[0]:psf_range_pix[1]]
                    psfres_temp = dict(psfse = psftemp.get(), timeStamp=jj*Tsim)
                    psfres_fname='./savedir%d/psfres_%04d'%(GPUnum,jj)
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
            sys.stdout.write("\r iter: %d/%d, ET: %.1f s, on-axis WF RMS [nm]: %.1f"%(jj, self.totSimulIter, 
                                                                                      self.tid.elapsedTime*1e-3, self.gs.phaseRms()*1e9))
            sys.stdout.flush() 

            #---------- End of closed-loop iterations!!!

        #-- show final on-axis residual map
        if self.VISU:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(20,5)

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
            inten_query = self.intensity_query(PSFcleprof, Rfvec, sep_req)
            print("Coronagraphic PSF contrast at %.1f mas: %.2e"%(sep_req,inten_query))

            if self.do_wo_coro_too:
                PSFleprof = ndimage.mean(PSFle.get(), labels=self.Rflabel, index=self.Rfidx)
                print("Long-exposure PSF Strehl Ratio @ %1.2f um: %1.3f"%(self.lim*1e6,self.strehl_ratio(PSFle)))

        elif self.do_psf_le:
            PSFleprof = ndimage.mean(PSFle.get(), labels=self.Rflabel, index=self.Rfidx)
            srle = self.strehl_ratio(PSFle.get())
            print("Long-exposure PSF Strehl Ratio @ %1.2f um: %1.3f"%(self.lim*1e6,srle))
            
            
        if self.do_psf_le and self.VISU:
            show_psf_and_profile(PSFle.get(), PSFleprof, im_display_size=1000,clim=[-5,0])
        if self.coro_psf and self.VISU:
            show_psf_and_profile(PSFcle.get(), PSFcleprof, im_display_size=1000)   
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                self.show_psf_and_profile(PSFle.get(), PSFleprof, im_display_size=1000)
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                self.show_two_psfs(PSFle.get(), PSFcle.get(), im_display_size=1000, log=True, clim=[-8,0])
        ####### Compare PSFS with/without M1 aberrations
        if self.simul_M1polish and not self.simul_turb and self.VISU:
            self.show_two_psfs(PSFc.get(), PSFle.get(), im_display_size=1000, log=True, clim=[-9,-4])
        # Overplot radial profiles
        if self.coro_psf and self.VISU:
            if self.do_wo_coro_too:
                fig, ax2 = plt.subplots()
                fig.set_size_inches(10,5)
                ax2.loglog(Rfvec, PSFleprof, 'b', linewidth=3, label='w/o PC')
                ax2.loglog(Rfvec, PSFcleprof, 'r', linewidth=3, label=' w/ PC')
                ax2.plot(np.array([sep_req, sep_req]), np.array([1e-10,1]), 'k--', linewidth=1.5)
                ax2.plot(np.array([1,1e4]), np.array([inten_query, inten_query]), 'k--', linewidth=1.5)
                #ax2.plot(Rfvec, PSF1prof, 'k--', label='DL')
                ax2.set_xlim([1,1e4])
                ax2.set_ylim([1e-8,1e-0])
                ax2.grid()
                ax2.tick_params(labelsize=12)
                ax2.set_xlabel('radial distance [mas]', fontsize=14)
                ax2.set_ylabel('normalized intensity', fontsize=14)
                ax2.legend(fontsize=12)
                
                
        tosave = dict(D=self.D, nPx=self.nPx, PupilArea=self.PupilArea, Tsim=Tsim, totSimulTime=totSimulTime, totSimulInit=self.totSimulInit,
        simul_turb=self.simul_turb, simul_M1polish=self.simul_M1polish, simul_onaxis_AO=self.simul_onaxis_AO, simul_windload=self.simul_windload,
        eval_perf_modal=self.eval_perf_modal, eval_perf_modal_turb=self.eval_perf_modal_turb, simul_truss_mask=self.simul_truss_mask)

        if self.simul_turb:
            tosave.update(dict(totSimulIter=self.totSimulIter, seeing=self.seeing, r0=self.r0, L0=self.L0, tau0=self.tau0, wind_speed=self.wind_speed, zen_angle=self.zen_angle, atm_fname=self.atm_fullname))
            if self.simul_variable_seeing:
                tosave.update(dict(r0_iter=self.r0_iter))

        if self.eval_perf_modal:
            tosave.update(dict(seg_aRes_gs_iter=seg_aRes_gs_iter))

        if self.eval_perf_modal_turb:
            tosave.update(dict(seg_aTur_gs_iter=seg_aTur_gs_iter))

        if self.simul_windload:
            tosave.update(dict(wldir=wldir, wlfn=wlfn, wlstart=wlstartInit, remove_quasi_statics=remove_quasi_statics,
                            wlstart0=wlstart0, zen=zen, az=az, wl_wind_speed = wl_wind_speed, vents=vents,
                            wind_screen=wind_screen))

        if self.simul_onaxis_AO:
            tosave.update(dict(band=self.band, lwfs=self.gs.wavelength, mag=self.mag, e0=self.e0, nLenslet=self.nLenslet, M2_n_modes=self.M2_n_modes, IMfile=self.IMfnameFull,
                pixelSize=self.pixelSize, gAO=gAO.get(), tot_delay=tot_delay, SPPctrlInit=self.SPPctrlInit, SPP2ndChInit=self.SPP2ndChInit, SPP2ndCh_Ts=self.SPP2ndCh_Ts, AOinit=AOinit))

            if self.save_telemetry:
                tosave.update(dict(a_M2_iter=self.a_M2_iter, da_M2_iter=self.da_M2_iter, wfe_gs_iter=self.wfe_gs_iter, spp_gs_iter=self.spp_gs_iter, 
                    seg_wfe_gs_iter=self.seg_wfe_gs_iter,wfs_meas_iter=wfs_meas_iter, wfgrad_iter=wfgrad_iter, seg_wfgrad_iter=seg_wfgrad_iter, 
                                   opdr_spp_iter=self.crazy_spp_iter))

            # Pyramid WFS parameters
            tosave.update(dict(pyr_modulation=self.pyr_modulation, pyr_angle=self.pyr_angle, pyr_thr=self.pyr_thr, 
               percent_extra_subaps=self.percent_extra_subaps, throughput=self.throughput, pyr_fov=self.pyr_fov, RONval=self.RONval,
               excess_noise=self.excess_noise, simul_wfs_noise=self.simul_wfs_noise))

            if self.ogtl_simul:
                tosave.update(dict(ogtl_probe_params=self.ogtl_probe_params, ogtl_detrend_deg=self.ogtl_detrend_deg, ogtl_gain=self.ogtl_gain, #ogtl_ogc_iter=ogtl_ogc_iter, #ogtl_ogeff_iter=ogtl_ogeff_iter,
                                  ogtl_ogc_outer_iter=self.ogtl_ogc_outer_iter,
                                  ogtl_ogc_centr_iter=self.ogtl_ogc_centr_iter,
                                  ogtl_radord_deg=self.ogtl_radord_deg, ogtl_Ts=[ogtl_Ts_1,ogtl_Ts_2], probeSigInit=probeSigInit,
                                  ogtl_reconfig_iter=self.ogtl_reconfig_iter, ogtl_ticks=self.ogtl_ticks, ogtl_ogeff_probes_iter=self.ogtl_ogeff_probes_iter,ogtl_ogc_probes_iter=self.ogtl_ogc_probes_iter))

        if self.do_psf_le:
            tosave.update(dict(srle=srle, lim=self.lim))
            if self.save_telemetry:
                tosave.update(dict(sr_iter=sr_iter, im_centr_iter=im_centr_iter))
                
        filename='./savedir'+str(self.GPUnum)+'/simul_results.npz'
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
            filename='./savedir'+str(self.GPUnum)+'/psf_results.npz'
            np.savez_compressed(filename, **tosavePSF)
                      
    
    # just copy the files in ./savedir to a newly generated TN
    def saveTN(self):
        tnString = datetime.today().strftime('%Y%m%d_%H%M%S')
        tnpath = os.path.join(self.TN_dir, tnString)
        os.makedirs(tnpath,exist_ok=True)
        os.system('cp ./savedir' + str(self.GPUnum)+ '/* ' + tnpath )  

        
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
            self.da_M2_iter = data['da_M2_iter']
            self.wfe_gs_iter = data['wfe_gs_iter']
            self.spp_gs_iter = data['spp_gs_iter']
            self.seg_wfe_gs_iter = data['seg_wfe_gs_iter']
            self.wfs_meas_iter = data['wfs_meas_iter']
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

    
    def analyseSimulation(self):

        
        self.totSimulInit =500
        
        self.propagatePyramid()
        self.M2_KL_modes()
        self.getOrCalibrateIM()
#        self.doOTGLsimul()
        
        nall = (self.D_M2_MODES.shape)[1]  ## number of modes calibrated
        self.n_mode = nall//7

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
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(15,5)

        #### on-axis WFE vs. iteration number
        ax1.plot(self.wfe_gs_iter*1e9)
        ax1.grid()
        ax1.set_xlabel('Iteration number', fontsize=20)
        ax1.set_ylabel('WF RMS [nm]', fontsize=20)
        ax1.tick_params(labelsize=15)

        #### on-axis segment phase piston (SPP) RMS vs. iteration number
        ax2.plot(np.std(self.spp_gs_iter,0)*1e9, label='OPD')
        ax2.plot(np.std(self.crazy_spp_iter,0)*1e9, label='OPDr')
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel('Iteration number', fontsize=20)
        ax2.set_ylabel('SPP [nm WF]', fontsize=20)
        ax2.tick_params(labelsize=15)
        #ax2.set_xlim([40,60])

        plt.show()

        wfe_final = np.mean(self.wfe_gs_iter[self.totSimulInit:])
        spp_final  = np.mean((np.std(self.spp_gs_iter,   0))[self.totSimulInit:])
        spp_final2 = np.mean((np.std(self.crazy_spp_iter,0))[self.totSimulInit:])
        SR_marechal_final = np.exp(-(wfe_final*2*np.pi/self.lim)**2)

        print ('      Final WFE [nm RMS]: %3.1f'%(wfe_final*1e9))
        #print ('      Final SPP RMS [nm]: %3.1f'%(spp_final*1e9))
        print ('      Final SPP RMS [nm]: %3.1f'%(spp_final2*1e9))
        print ('Equivalent SR @ %2.2f um: %1.4f'%(self.lim*1e6,SR_marechal_final))


        # WFE and SPP
        #-- Differential segment piston w.r.t. central segment
        diff_spp_iter =self.spp_gs_iter - self.spp_gs_iter[6,:]
        diff_spp_iter2 =self.crazy_spp_iter - self.crazy_spp_iter[6,:]
        #### on-axis segment WFE vs. iteration number
        fig, (ax1,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(15,5)
        ax1.plot(timeVec,self.seg_wfe_gs_iter.T*1e9)#, '-+')
        ax1.grid()
        ax1.set_xlabel('elapsed time [s]', fontsize=15)
        ax1.set_ylabel('WF RMS [nm]', fontsize=15)
        ax1.tick_params(labelsize=13)
        #ax1.set_ylim([0,2000])
        #ax1.set_xlim([0,20])
        #### on-axis SPP vs. iteration number
        ax2.plot(timeVec,diff_spp_iter2.T*1e9)#, '-+')
        ax2.grid()
        ax2.set_xlabel('elapsed time [s]', fontsize=15)
        ax2.set_ylabel('SPP [nm WF]', fontsize=15)
        ax2.tick_params(labelsize=13)
        #ax2.set_ylim([-4000,4000])
        #ax2.set_xlim([0,20])

        lwfs = np.round(self.gs.wavelength*1e9)
        yticksrange = np.round((np.min(diff_spp_iter*1e9), np.max(diff_spp_iter*1e9)) / lwfs)
        lwfs_ticks = np.arange(yticksrange[0]-1,yticksrange[1]+1)*lwfs
        ax2.set_yticks(lwfs_ticks) # yTicks will be set at multiples of the WFS central wavelength"""
        #ax2.set_xticks(SPP2ndCH_ticks)
        #ax2.set_xlim([0,2])
        #ax1.set_xlim([0,2])

        plt.show()
        
        
        self.n_mode = 500
        
        daM2mv  = np.zeros((self.n_mode,7))
        daM2var = np.zeros((self.n_mode,7))
        for segId in range(7):
            daM2mv[:,segId] = np.mean(self.da_M2_iter[segId,:,self.totSimulInit:], axis=1)
            daM2var[:,segId] = np.var(self.da_M2_iter[segId,:,self.totSimulInit:], axis=1)
        aM2mv  = np.zeros((self.n_mode,7))
        aM2var = np.zeros((self.n_mode,7))
        for segId in range(7):
            aM2mv[:,segId] = np.mean(self.a_M2_iter[segId,:,self.totSimulInit:], axis=1)
            aM2var[:,segId] = np.var(self.a_M2_iter[segId,:,self.totSimulInit:], axis=1)
        
        # M2 commands modal plot
        f1, ax = plt.subplots(ncols=3, nrows=3)
        f1.set_size_inches(20,15)
        for jj in range(7):
            thisax = (ax.ravel())[jj]
            thisax.loglog(np.arange(self.n_mode)+1,aM2var[:,jj]*(1e9**2))
            thisax.loglog(np.arange(self.n_mode)+1,daM2var[:,jj]*(1e9**2))
            thisax.grid()
            thisax.set_title('Segment #%d'%(jj+1), size='large')
            thisax.set_xlabel('mode number + 1')
            thisax.set_ylabel('variance [nm$^2$ WF]', size='large')
            #thisax.set_ylim([-30,100])
        for k in range(7,9):
            (ax.ravel())[k].axis('off')

        plt.show()

        # Optical Gain
        fig, (ax,ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(12,4)
        ax.plot(self.ogtl_ticks, np.array(self.ogtl_ogeff_probes_iter), drawstyle='steps-post')
        label = ['KL#%d'%kk['mode'] for kk in self.ogtl_probe_params]
        #ax.set_xticks(OGTL_ticks[1:-1:4])
        ax.legend(label)
        ax.grid()
        ax.set_xlabel('elapsed time [s]', size='large')
        ax.set_ylabel('effective OG', size='large')
        #ax.set_xlim([0,2.05])

        caca = ax2.plot(self.ogtl_ticks, 1-np.array(self.ogtl_ogeff_probes_iter), drawstyle='steps-post')
        ax2.grid()
        ax2.set_xlabel('elapsed time [s]', size='large')
        ax2.set_ylabel('1 - effective OG', size='large')
        
        plt.show()

        fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2)
        fig.set_size_inches(14,11)
        ax.plot(self.ogtl_ticks, self.ogtl_ogc_outer_iter, drawstyle='steps-post')
        #ax.set_xticks(OGTL_ticks[1:-1:4])
        ax.grid()
        ax.set_xlabel('elapsed time [s]', size='large')
        ax.set_ylabel('OGC coefficient', size='large')
        ax.set_title('outer segments', fontsize=13)
        #ax.set_xlim([0,2.05])

        this_iter = 15

        radord_data = dict(np.load(self.dir_calib+'ASM_fittedKLs_S7OC04184_675kls_radord.npz'))
        self.radord_all_outer = radord_data['outer_radord'][0:self.n_mode]
        self.radord_all_centr = radord_data['centr_radord'][0:self.n_mode]
        outer_max_radord = np.max(self.radord_all_outer)
        self.probe_radord = np.round(np.arange(6)*outer_max_radord/6+3).astype('int')
        self.probe_outer = np.zeros(6, dtype='int')
        for jj in range(6):
            self.probe_outer[jj] = np.argwhere(radord_data['outer_radord'] == self.probe_radord[jj])[1]

        ax2.plot(self.radord_all_outer, self.ogtl_ogc_outer_iter[this_iter], '+')
        ax2.plot(self.probe_radord, self.ogtl_ogc_probes_iter[this_iter], 'o')
        ax2.set_xlabel('KL group', size='large')
        ax2.set_ylabel('OGC coefficient', size='large')

        ax2.grid()
        ax2.legend(['OGTL iteration %d'%this_iter])

        ax3.plot(self.ogtl_ticks, np.array(self.ogtl_ogc_centr_iter), drawstyle='steps-post')
        #a3x.set_xticks(OGTL_ticks[1:-1:4])
        ax3.grid()
        ax3.set_xlabel('elapsed time [s]', size='large')
        ax3.set_ylabel('OGC coefficient', size='large')
        ax3.set_title('central segments', fontsize=13)
        #ax3.legend()
        #ax3.set_xlim([0,2.05])

        ax4.plot(self.radord_all_centr, self.ogtl_ogc_centr_iter[this_iter],'+')
        ax4.plot(self.probe_radord, self.ogtl_ogc_probes_iter[this_iter], 'o')
        ax4.set_xlabel('KL group', size='large')
        ax4.set_ylabel('OGC coefficient', size='large')
        #ax4.set_xlim([480,500])
        ax4.grid()
        ax4.legend(['OGTL iteration %d'%this_iter])
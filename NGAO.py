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
        
        #Select M1 map
        if self.simul_M1polish:
            # self.M1_map = u'M1_print_through'     # Print-through errors.
            self.M1_map = u'M1_meeting_proposed_spec'  # M1 polishing errors.
        else:
            self.M1_map = ''
            
        self.do_psf_le = eval(parser.get('general', 'do_psf_le'))
        self.coro_psf = eval(parser.get('general', 'coro_psf'))
        self.ogtl_simul = eval(parser.get('general', 'ogtl_simul'))
        self.eval_perf_modal = eval(parser.get('general', 'eval_perf_modal'))
        self.eval_perf_modal_turb = eval(parser.get('general', 'eval_perf_modal')) and self.simul_turb
        self.VISU = eval(parser.get('general', 'VISU'))
            
        self.do_wo_coro_too = self.coro_psf  # If True, compute both w/coro and w/o coro in the same simulation
        
        tid = ceo.StopWatch()  # Keep the time
        
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

        """self.gmt.M2.modes.a[0,9] = 5e-9
        self.gmt.M2.modes.a[1,105] = 5e-9
        self.gmt.M2.modes.a[2,300] = 5e-9
        self.gmt.M2.modes.a[3,400] = 5e-9
        self.gmt.M2.modes.a[4,500] = 5e-9
        self.gmt.M2.modes.a[5,674] = 5e-9"""

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
            fnameFull = os.path.normpath(os.path.join(RECdir,fname))
            print(fnameFull)
            #TODOFR: Flag to generate or not the file
            if not os.path.isfile(fnameFull):  
                self.D_M2_MODES = self.gmt.NGWS_calibrate(self.wfs,self.gs, stroke=self.Zstroke)
                tosave = dict(D_M2=D_M2_MODES, first_mode=self.z_first_mode, Stroke=self.Zstroke)
                np.savez(fnameFull, **tosave)
            else: 
                print(u'Reading file: '+fnameFull)
                ftemp = np.load(fnameFull)
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
            KL0_idx = self.n_mode*np.array([0,1,2,3,4,5])
            if self.VISU:
                fig, ax1 = plt.subplots()
                fig.set_size_inches(7,5)
                imm = ax1.pcolor(self.D_M2_PSideal)
                ax1.grid()
                fig.colorbar(imm, ax=ax1)#, fraction=0.012) 
                
            self.R_M2_PSideal = np.linalg.pinv(self.D_M2_PSideal)
            self.SPP2ndCh_thr = self.gs.wavelength * self.wvl_fraction

            
    def probe_signal(self, fs, kk):
        for pp in ogtl_probe_params:
            ogtl_probe_vec[pp['seg'], pp['mode']] = pp['amp']*np.sin(2*np.pi*pp['freq']/fs*kk)
            
                        
    def optical_gain_cl(self, probes_in,probes_out):    
        probe_in_fitcoef = [np.polyfit(ogtl_timevec,this_probe,ogtl_detrend_deg) for this_probe in probes_in]
        probe_in_fit = [poly_nomial(ogtl_timevec,this_coeff) for this_coeff in probe_in_fitcoef]
        probes_in = np.array([pr-prf for (pr,prf) in zip(probes_in,probe_in_fit)])
        A_in = 2*np.sqrt(np.mean(probes_in * cosFn, axis=1)**2 + np.mean(probes_in * sinFn, axis=1)**2)

        probe_out_fitcoef = [np.polyfit(ogtl_timevec,this_probe,ogtl_detrend_deg) for this_probe in probes_out]
        probe_out_fit = [poly_nomial(ogtl_timevec,this_coeff) for this_coeff in probe_out_fitcoef]
        probes_out = np.array([pr-prf for (pr,prf) in zip(probes_out,probe_out_fit)])
        A_out = 2*np.sqrt(np.mean(probes_out * cosFn, axis=1)**2 + np.mean(probes_out * sinFn, axis=1)**2)

        OG = A_out / A_in   

        for kk in range(len(seg_list)):
            print('  Optical Gain of mode S%d KL%d: %0.3f'%(seg_list[kk],mode_list[kk],OG[kk]))

        return OG
        
    def doOTGLsimul(self):
        if self.ogtl_simul:
            # Warning! hard-code below. It assues the basis used is ASM_fittedKLs_S7OC04184_675kls.ceo
            radord_data = dict(np.load(self.dir_calib+'ASM_fittedKLs_S7OC04184_675kls_radord.npz'))
            radord_all_outer = radord_data['outer_radord'][0:self.n_mode]
            radord_all_centr = radord_data['centr_radord'][0:self.n_mode]
            outer_max_radord = np.max(radord_all_outer)
            probe_radord = np.round(np.arange(6)*outer_max_radord/6+3).astype('int')
            probe_outer = np.zeros(6, dtype='int')
            for jj in range(6):
                probe_outer[jj] = np.argwhere(radord_data['outer_radord'] == probe_radord[jj])[1]
            print(probe_outer)
            print(probe_radord)
            
            # Define probe signal parameters [segment#, mode#, amplitude, frequency]
            ogtl_probe_params = []
            ogtl_probe_params.append(dict(seg=0, mode=probe_outer[0], amp=2.5e-9, freq=310))
            ogtl_probe_params.append(dict(seg=1, mode=probe_outer[1], amp=2.5e-9, freq=310))
            ogtl_probe_params.append(dict(seg=2, mode=probe_outer[2], amp=2.5e-9, freq=310))
            ogtl_probe_params.append(dict(seg=3, mode=probe_outer[3], amp=2.0e-9, freq=310))
            ogtl_probe_params.append(dict(seg=4, mode=probe_outer[4], amp=2.0e-9, freq=310))
            ogtl_probe_params.append(dict(seg=5, mode=probe_outer[5], amp=2.0e-9, freq=310))
            #ogtl_probe_params.append(dict(seg=6, mode=probe_centr, amp=2.5e-9, freq=310))

            ogtl_probe_vec = np.zeros((7,self.n_mode))

            seg_list  = [kk['seg']  for kk in ogtl_probe_params]
            mode_list = [kk['mode'] for kk in ogtl_probe_params]
            freq_list = [kk['freq'] for kk in ogtl_probe_params]
            
            self.gmt.reset()
            self.gs.reset()
            
            for pp in ogtl_probe_params:
                ogtl_probe_vec[pp['seg'], pp['mode']] = pp['amp']
            self.gmt.M2.modes.a[:,self.z_first_mode:self.n_mode] = ogtl_probe_vec
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
        nisl = len(islands)
        IslMat = np.zeros((self.npseg[index],nisl))
        for jj in range(nisl):
            IslMat[:,jj] = islands[jj].ravel()[self.P[6,:]]

        inv_IslMat = np.linalg.pinv(IslMat)
        
        
        #---- Island modes as linear combination of segment KL modes
        if self.eval_perf_modal == True:
            Isl2segKLmat = self.inv_segKLmat[index] @ IslMat

            inv_S7KLmat_pistfree = np.linalg.pinv(self.segKLmat[index][:,1:n_mode])
            Isl2segKLmat_pistfree = inv_S7KLmat_pistfree @ IslMat
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
            self.gmt.M2.modes.a[index,first_kl:n_mode] = this_comm
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

                ax2.semilogx(np.arange(first_kl,n_mode)+1,this_comm*1e9)
                ax2.grid()
                ax2.set_xlabel('segment KL number + 1')
                ax2.set_ylabel('coeff amp [nm RMS]')
                plt.show()
                
                

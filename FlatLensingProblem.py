# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:40:58 2014

@author: jpbreuer
"""
from __future__ import division

from nifty import *                                                   # version 0.6.8
from nifty.nifty_tools import *
from nifty.nifty_explicit import *

#---For Maksim's code--->
from vector_field import *
from derivatives import *

#--- Other --->
#from itertools import *

#--- Response Operator--->
from FlatLensingResponse import LensingResponse as LR

about.warnings.on()
about.infos.on() 


###########
test = False       # if True, then testconfig class will run
###########

class config():
    """
    Defines spaces and known Covariances, separated into several segments:
        Preliminaries: Change input/output paths, !!! keep self.test here False !!!
        
        Specify Space Characteristics: Edit the required parameters for the domain needed
    
        Prior Knowledge of Primary CMB: Edit the conditions for the CMB to be lensed
    
        Prior Knowledge of Lensing Potential: Edit the conditions for the lensing potential
    
        Noise: Should be obvious
    """
    def __init__(self):
#--- Preliminaries --->
    #--- Input Directory --->
        self.filename='/CMBLens_Implemenations/PowerSpectra/CAMBOutput/nlinear/lmax25000_nonlinear_scalcls_new.dat' 
    #--- Output Directory --->
        self.path = '/output'
    #--- For WienerFilter_flat --->
        self.WF = 'signalspace'
        self.test = False #Do not change this!
        
        self.NFW_profile = False # if True, will use the NFW dark matter halo profile as a lensing potential
########################################
#--- Specify Space Charactersitics --->
########################################
        self.space ='rg'
        self.pixels = 300        #np.sqrt(12*2048**2).astype(int)
        self.distance = 1
        self.naxes = 2

        if (self.space == 'rg'):
    #--- s_space ---> DOMAIN
            self.domain = rg_space(num=self.pixels, naxes=self.naxes, dist=self.distance, zerocenter=True)
            print(self.domain)
            self.spixsize = self.distance**2
            print("Number of Pixels (s_space): %d" %(self.domain.dim()))
    #--- k_space ---> CODOMAIN
            self.codomain = self.domain.get_codomain()
            print(self.codomain)
#            for i in self.codomain.get_power_indices():
#                print("Kvalues: ", i)
    #--- Check K-values --->
            self.codomain.get_power_indices()
            k = self.codomain.power_indices["kindex"]
            self.p = self.codomain.power_indices["pindex"]
#            print(self.p)
            
#########################################
#--- Prior Knowledge of Primary CMB --->
#########################################
#--- Power Spectra from CAMB --->
    #--- Load File --->
            CAMBSpectra = np.loadtxt(self.filename, dtype=float)
    #--- Specify Maxima and Range --->
            upperbound = len(k) - 1#(np.sqrt(self.codomain.dof()))*self.naxes-1 #6200 #CONFIG
            l = np.array(CAMBSpectra[0:upperbound, 0])
    #--- Power Spectrum --->
            T_Power_full = np.array(CAMBSpectra[0:upperbound, 1])
            T_Power_full = T_Power_full/(l*(l+1))*2*np.pi #l(l+1)C_l/2pi        
    #--- Insert Some Monopol --->
            T_Power_full = np.insert(T_Power_full, 0, 1000)
    #--- Rescale (if needed) --->
            self.T_Power = T_Power_full
    #--- Covariance Matrix --->
            self.C_T = power_operator(self.codomain, spec=self.T_Power, bare=True) #CONFIG

###############################################
#--- Prior Knowledge of Lensing Potential --->
###############################################
            if (self.NFW_profile == False):
    #--- CAMB Normalisation --->
            #--- Outputscale --->
                CMB_outputscale = 7.4311e12 #CONFIG
            #--- Power Spectrum --->
                Psi_Power_full = CAMBSpectra[0:upperbound, 4] / CMB_outputscale / (l**4)
            #--- Insert Some Monopol --->
                Psi_Power_full = np.insert(Psi_Power_full, 0, 3e-6) #8
            #--- Rescale --->
                self.Psi_Power = 10e21*Psi_Power_full #15
            #--- Covariance Matrix --->
                self.C_Psi = power_operator(domain=self.codomain, target=self.codomain, spec=self.Psi_Power, bare=True)
            else:
    #--- NFW Profile --->
            #--- Basic Parameters --->
                G_N = 6.67e-11                          # Newtonian Gravitational constant
                Hubble = 0.704                          # Scale factor for Hubble expansion parameter
                omega_c = 0.227                         # CDM Density
                omega_b = 0.0456                        # Baryonic Density
                omega_lambda = 0.728                    # Dark Energy Density
                omega_total = omega_c + omega_b
            #--- Specific Parameters --->
                delta_c = 200                           # Characteristic Overdensity
                M = 3.5e13                              # Mass
                c = 1                                   # Concentration            
#                Hubble_0 = 100*h / 3.086e19
#                rho_crit = ((3 * Hubble_0**2) / (8*pi*G_N))
                rho_crit = 2.775e11                     # Units: h^2 M⊙ Mpc^−3
                R_s = (3*M / (4*pi*omega_total*rho_crit) * (1/delta_c))**(1./3.)
            #--- NFW Halo Density --->
                self.Rho = (lambda r: rho_crit / ((r*c/R_s) * (1 + (r*c/R_s))**2))
            #--- Lensing Potential Phi --->
                self.Phi = (lambda r: (((-1.*G_N*M)/(r + 0.0001)) * np.log(1 + r/R_s)))
            #--- Field --->
                r = self.Find_r()
                self.phi_r = np.asarray(map(self.Phi, r))
#                self.phi_r[self.pixels/2, self.pixels/2] = 0
                self.lensing_potential = field(self.domain, target=self.codomain, val=self.phi_r)
            

################
#--- Noise --->
################
    #--- Variance --->
            self.nvar = 1e-20 #CONFIG
            self.N = diagonal_operator(self.domain, diag=self.nvar)
            
            
        else:
            print("Pick another space.")

    def Find_r(self):
        "Outputs an array where the distances (r) from center pixel are calculated"
    #--- Define Space --->
        zeroarray = np.zeros([self.pixels, self.pixels]).astype(float)
    #--- For X --->
        xarray = np.array(range(self.pixels)).astype(float)
        xpositions = zeroarray + xarray[None,:]
    #--- New Coordinates X --->
        newx = (xpositions - (self.pixels/2))*self.distance
    #--- For Y --->
        yarray = np.array(range(self.pixels)).astype(float)
        ypositions = zeroarray + yarray[:,None]
    #--- New Coordinates Y --->
        newy = (ypositions - (self.pixels/2))*self.distance
    #--- Grid --->
        grid = zeroarray
        for i in np.arange(self.pixels):
            for j in np.arange(self.pixels):
                grid[i][j] = np.sqrt(((newx[i][j])**2) + ((newy[i][j])**2))
        return grid
        

class MockProblem():
    "Creates mock data, then makes plots"
    def __init__(self, config):
        self.config = config
        self.R = LR(config)
    #--- Creation of Mock Data --->
        self.d = self.createData()
        self.d_power = self.d.power()
        self.makePlots(PowerOnly=False)
    
    def createData(self):
        """
        Creates randomly drawn fields for the signal and noise from the given power spectrums in the config class
        Returns Data, the lensing response given the signal + noise
        """
        print("Creating Data...")
    #--- Draw CMB_T Random Field S --->
        self.s = self.config.C_T.get_random_field(domain=self.config.domain)
#        print(self.s)
        self.s_power = self.s.power()
    #--- Draw Noise --->
        self.n = self.config.N.get_random_field(domain=self.config.domain)
    #--- Create Data: Signal to Lensing Response --->
        self.Rs = self.R(self.s)
        d = self.Rs + self.n
        return d
#        return self.Rs

    def makePlots(self, PowerOnly):
        "Creates and saves plots of the fields to the output path"
        print("Making Plots...")
        path = self.config.path
    #--- Optional: Signal Map --->
        if (PowerOnly == False):
            self.s.plot(title="Signal", save=("%s/signal.png" %path))
#--- Power Spectra of Signal, Data and Noise --->
        fig = pl.figure()
	  #num=None,figsize=(6.4,4.8),dpi=None,facecolor=None,edgecolor=None,frameon=False,FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        ax0.loglog(self.s_power, label='Signal')
        ax0.loglog(self.d_power, label='Data')
        #ax0.loglog(self.n.power(),label='Noise')
        ax0.legend(loc='lower left')
        #ax0.set_xlim(0,3000)
        #ax0.set_ylim(bottom=10e-6)
        ax0.set_xlabel('l')
        ax0.set_ylabel('l(2l+1) Delta C_l ')
        #fig.canvas.draw()
        fig.savefig("%s/Psignal.png" %(path))
        pl.close(fig)
    #--- Optional: Data Map --->
        if (PowerOnly == False):
            self.d.plot(title="Data", save=("%s/data.png" %path))

#--- Relative Differences --->
    #--- Data - Signal / Signal --->
        fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor=None, edgecolor=None, frameon=False, FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        frac_change = (self.d_power - self.s_power) / self.s_power
        ax0.plot(frac_change,'k')
        #ax0.set_xlim(0,3000)
        #ax0.set_ylim(-8,8)
        ax0.set_title('Fractional Change in Power Spectrum d-s/s')
        ax0.set_xlabel('l')
        ax0.set_ylabel('l(2l+1) Delta C_l ')
        fig.savefig("%s/Frac_Change_d-s.png" %(path))
        pl.close(fig)       
        
#--- Lensing Potential Underlying Response --->
    #--- Optional: Underlying Lensing Potential Map --->
        if (PowerOnly == False):
            self.R.lensing_potential.transform(domain=self.config.domain).plot(title="Underlying Lensing Potential", save=("%s/Phi.png" %path))
        
    #--- Power Spectrum --->
#        self.R.lensing_potential.plot(title="Underlying Lensing Potential", power=True, mono=False, other=(self.config.Psi_Power), save=("%s/PPhi.png" %(path)))

class Testconfig():
    "Defines space, creates test grid, makes basic plots"
    def __init__(self):
    #--- Test --->
        self.test = True #Do not change
    #--- Output Directory --->
        self.path = '/Volumes/UltraZombie/users/jbreuer/Desktop/mpa-project/final/testoutput'
    #--- Define Space Characteristics --->
        self.space ='rg'
        self.pixels = 12        #np.sqrt(12*2048**2).astype(int) # must be odd!
        self.distance = 1
        self.naxes = 2
        self.NFW_profile = False
    #--- Domain --->        
        self.domain = rg_space(num=self.pixels, naxes=self.naxes, dist=self.distance, zerocenter=True)
        print(self.domain)
    #--- Codomain --->
        self.codomain = self.domain.get_codomain()
        print(self.codomain)
    #--- Check K-values --->
        self.codomain.get_power_indices()   #print(self.codomain.get_power_indices())
        k = self.codomain.power_indices["kindex"]
        self.p = self.codomain.power_indices["pindex"]
    #--- Test Non-Random Field --->
        self.testfield = np.zeros(shape = (self.pixels, self.pixels)).astype(int)
        self.testfield[0::2, 0::2] = 1
        self.testfield[1::2, 1::2] = 1
        self.testfield[1::2, 0::2] = -1
        self.testfield[0::2, 1::2] = -1
#        self.testfield[::, 1::3] = 0
        self.testfield = field(self.domain, val=self.testfield)
        self.testfield.plot(title="Signal", save=("%s/signal.png" %self.path))
    #--- Test Lensing Field --->
        x = np.ones(None)
        self.testlensing_potential = np.zeros(shape = (self.pixels, self.pixels)).astype(int)
        self.testlensing_potential[::, 0::3] = x #Test X direction
        self.testlensing_potential[0::3, ::] = x #Test Y direction ----- Use both for a party
        self.testlensing_potential = field(self.domain, target=self.codomain, val=self.testlensing_potential)
        self.testlensing_potential.plot(title="Underlying Lensing Potential", save=("%s/Phi.png" %self.path))
        
class TestProblem():
    "Does basic lensing using the given parameters from the Testconfig"
    def __init__(self, Testconfig):
        self.config = Testconfig
        self.lensing_potential = self.config.testlensing_potential
#        else:
#            self.lensing_potential
        self.R = LR(self.config, self.config.testlensing_potential)
        print("Shifting...")
        result = self.R(self.config.testfield)
        result.plot(title="Result", save="%s/result.png"%self.config.path)

        

if (test == False):
    if(__name__=="__main__"):
        MockProblem(config())
        print("Finished!")
else:
    if(__name__=="__main__"):        
        TestProblem(Testconfig())
        print("Finished!")
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:07:44 2014

@author: jbreuer
"""

from __future__ import division

from FlatLensingProblem import MockProblem
from FlatLensingProblem import config
import time

from nifty import *                                                   # version 0.6.8
from nifty.nifty_tools import *
from nifty.nifty_explicit import *

#---For Maksim's code--->
from vector_field import *
from derivatives import *

#--- Other --->
#from joblib import 

#--- Response Operator--->
from MyLensingResponse import LensingResponse as LR

class WienerFilter():
    "Computes mean field in Wiener Filter Theory"
    def __init__(self, config):
        #@profile
    #--- Creates Mock Data --->
#        if (config.mock == 1):
#            self.Mock = MockProblem(config)
#            self.d = self.Mock.d
#            self.R = self.Mock.R
            #self.Mock.makePlots(PowerOnly=True)
    #--- Reads in Real Data --->
#        elif (config.mock == 0):
#            self.d = ImportData().d
#        else:
#            print("Invalid Input Configuration")
        self.Mock = MockProblem(config)
        self.d = self.Mock.d
        self.R = self.Mock.R
        
        self.N = config.N

        self.S = config.C_T
        
        self.path = config.path
        
        self.domain = config.domain
        self.codomain = config.codomain
        
        self.WF = config.WF
        
        
    #--- Stores Results Before M is Computed --->
        
        m = field(self.domain)
        self.m_power = m.power()
        self.save_results(m)
        
    
#--- WIENER FILTER --->
    def solve(self):
        if (self.WF == 'signalspace'):
            print("Initialising Propagator...")
            self.D = propagator_operator(S=self.S, N=self.N, R=self.R)
            print(self.D.domain)
        
            print("Initialising Information Source...")
            j = self.R.adjoint_times(self.N.inverse_times(self.d))
#            if (self.Prec == True):
#                Prec = self.compute_Prec()
#                print(Prec)
#                print("Starting Wiener Filtering")
#                begin_WF = time.time()
#                m = self.D.times(j, W=Prec)
#                end_WF = time.time()
#            else:
            print("Starting Wiener Filtering...")
            begin_WF = time.time()
            m = self.D.times(j, note=True) 
            end_WF = time.time()
        elif (self.WF == 'dataspace'):
            pass
        
        self.m_power = m.power()
    #--- Storing Results --->
        print("Wiener Filtering took %s" %(end_WF - begin_WF))
    #--- Arrays of Power Spectra Saved to File --->
        print("Saving results...")
        begin_saving = time.time()
        self.save_results(m)
        print("Saving results took %s" %(time.time() - begin_saving))
        return m
        
    def makePlots_M(self, PowerOnly):
        """
        Makes plots
        """
    #--- Optional: Reconstructed Map --->
        if (PowerOnly == False):
            m.plot(title="Reconstructed Map", save=("%s/m.png" %self.path))
    #--- M Power Spectrum Together with Signal --->
        fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor=None, edgecolor=None, frameon=False, FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        ax0.loglog(self.m_power, label='Reconstructed Map')
        ax0.loglog(self.Mock.s_power, label='Signal')
        ax0.legend(loc='lower left')
        #ax0.set_xlim(0, 3000)
        #ax0.set_ylim(bottom = 10e-6)
        ax0.set_xlabel('l')
        ax0.set_ylabel('l(2l+1) Delta C_l ')
        fig.savefig("%s/Pm_Ps.png" %(self.path))
        pl.close(fig)
    #--- Optional: M Power Spectrum Together with Data and s_rec (Signal Reconstructed?) --->
        fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor=None, edgecolor=None, frameon=False, FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        ax0.loglog(self.m_power, label='Reconstructed Map')
        ax0.loglog(self.Mock.d_power, label='Data')
        #ax0.loglog(self.Mock.s_rec.power(), label = 'R^T(Rs)') #CONFIG
        ax0.legend(loc='lower left')
        #ax0.set_xlim(0, 3000)
        #ax0.set_ylim(bottom = 10e-6)
        ax0.set_xlabel('l')
        ax0.set_ylabel('l(2l+1) Delta C_l ')
        fig.savefig("%s/Pm.png" %(self.path))
        pl.close(fig)
        
#--- Relative Differences --->
    #--- Reconstructed Map - Signal / Signal --->
        fig = pl.figure(num=None, figsize=(6.4, 4.8), dpi=None, facecolor=None, edgecolor=None, frameon=False, FigureClass=pl.Figure)
        ax0 = fig.add_axes([0.12, 0.12, 0.82, 0.76])
        frac_change = (self.m_power - self.Mock.s_power) / self.Mock.s_power
        #ax0.set_xlim(0, 3000)
        #ax0.set_ylim(-10, 10)
        ax0.plot(frac_change,'k')
        ax0.set_title('Fractional Change in Power Spectrum m-s/s')
        ax0.set_xlabel('l')
        ax0.set_ylabel('l(2l+1) Delta C_l ')
        #fig.canvas.draw()
        fig.savefig("%s/Frac_Change_m-s.png" %(self.path))
        pl.close(fig)              
     
    def save_results(self, m):
        f = file("%s/outputs_power.bin" %(self.path), "wb")
        np.save(f, self.Mock.s_power)
        #np.save(f, self.Mock.n.power())
        np.save(f, self.Mock.d_power)
        np.save(f, self.m_power)
        #np.save(f, self.R.LensPot_lm.power())
        f.close()
        #f = file("%s/outputs.bin" %(self.path), "wb")
        #np.save(f, self.Mock.s)
        #np.save(f, self.Mock.n)
        #np.save(f, self.d)
        #np.save(f, m)
        #f.close()

        
if(__name__=="__main__"):
    start = time.time()
    m = WienerFilter(config()).solve()
    print("Finished! Time taken: %.1f seconds" %(time.time() - start))

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:50:04 2014

@author: jpbreuer
"""

#--- Response Operator CMB Lensing --->

from __future__ import division
from nifty import *

#--- For Maksim's Derivatives Implementation --->
from vector_field import *
from derivatives import *

#--- Other --->


note = notification()

#--- CMB Lensing Response --->
class LensingResponse(response_operator):
    def __init__(self, config, lensing_potential=None):
        """
        Parameters
        -----------
        lensing_potential : field | default = None
            default takes a fixed and explicit potential
            
            otherwise, takes a power spectrum from the config as input,
            and creates a realisation of a potential from it

        """
        #@profile
        if (not isinstance (config.domain, space)):
            raise TypeError(about._errors.cstring("ERROR: invalid input."))
        self.domain = config.domain
        self.target = config.domain
        self.pixels = config.pixels
        self.codomain = config.codomain
        self.distance = config.distance                
        self.path = config.path
        self.test = config.test
        self.NFW_profile = config.NFW_profile

        self.sym = False 
        self.uni = False
        self.imp = True 
    #--- From test --->
        if (self.test == True):
            self.testlensing_potential = config.testlensing_potential
            if (self.testlensing_potential.domain.fourier == False):
                self.lensing_potential = self.testlensing_potential.transform()
#            print(self.lensing_potential.domain)
            print("Test case activated!")
        else:
    #--- Default explicit --->
            if (self.NFW_profile == True):
                self.lensing_potential = config.lensing_potential.transform()
                if (self.lensing_potential.domain.fourier == False):
                    self.lensing_potential = self.lensing_potential.transform()
                print("R-Operator: Lensing Potential in rg-space was passed")
            else:
                if (lensing_potential == None):     # Takes a fixed and explicit potential
                    self.lensing_potential = config.C_Psi.get_random_field(domain = config.codomain, target = config.domain)
                    if (self.lensing_potential.domain.fourier == False):
                        self.lensing_potential = self.lensing_potential.transform()
#                print(self.lensing_potential.domain)
                print("R-Operator: Lensing Potential was created")

        
    
            

    #--- Gradient --->
        self.gradient_x = self.gradientfield(self.lensing_potential)[0]
        self.gradient_y = self.gradientfield(self.lensing_potential)[1]
#        print("A_x")
#        print(self.gradient_x)
#        print("B_y")
#        print(self.gradient_y)

    #--- Delta X --->
        self.xshift = self.pix2shift(self.gradient_x)[0]
#        print("DeltaX: ")
#        print(self.xshift)
    #--- Transform X --->
        self.xposition = self.pix2shift(self.gradient_x)[1]
#        print("PositionX: ")
#        print(self.xposition)

    #--- Delta Y --->
        self.yshift = self.pix2shift(self.gradient_y)[0]
#        print("DeltaY: ")
#        print(self.yshift)
    #--- Transform Y --->
        self.yposition = self.pix2shift(self.gradient_y)[2]
#        print("PositionY: ")
#        print(self.yposition)
        
        
    def pix2shift(self, field):
        """
        Parameters
        -----------
        field : array
            Takes a field or gradient field
        
        Function then rounds values to appropriate pixel, 
        returns the delta coordinates of the given field,
        then returns the positions of x and y
        """
    #--- Get Dimensions --->
        pixels = field.shape[0]
    #--- For Delta --->
        deltashift = np.round(field / self.distance).astype(float)
#        print("Delta: ")
#        print(field/self.distance)
    #--- Adjust Boundary Conditions for Delta --->
        pixelshift = deltashift / pixels # Number of times looped through grid.float becomes where pixel goes to
        pixelshiftint = pixelshift.astype(int)
        deltashift = (pixelshift - pixelshiftint) * pixels
#        deltashift[deltashift < 0] += pixels # adjust for negative values
        deltashift = deltashift.astype(float)
    #--- Define Empty Array --->
        zeroarray = np.zeros(shape = (deltashift.shape)).astype(float)
    #--- For X --->
        xarray = np.array(range(pixels)).astype(float)
        xpositions = zeroarray + xarray[None,:]
        newx = deltashift + xpositions
    #--- Boundary Conditions --->
        newx[newx < 0] += pixels
        newx[newx >= pixels] -= pixels
        #print(newx)
    #--- For Y --->
        yarray = np.array(range(pixels)).astype(float)
        ypositions = zeroarray + yarray[:,None]
        newy = deltashift + ypositions
#        print("Before")
#        print(newy)
    #--- Boundary Conditions --->
        newy[newy < 0] += pixels
        newy[newy >= pixels] -= pixels
#        print("After")
#        print(newy)
        return deltashift, newx, newy
        
    def gradientfield(self, field):
        """
        Parameters
        -----------
        field : array
            Takes any given field
        
        Function then calculates the discrete or continuous gradient field using Maksim's implementation
        see 'derivatives.py'
        """
    #--- Nabla Operator --->
#        p = position_field(self.domain)
#        print("P", p.val)
        ##test for Ripple effect
#        deri_cont = Nabla_continuous(self.codomain)   #The Fourier transform of the continuous derivative
        deri_disc = Nabla_discrete(self.codomain)     #The Fourier transform of the discrete derivative
    
    #--- Compute Gradient Field --->
    #--- SUPER IMPORTANT ---> !!!!!
        gradient = (deri_disc * field).transform() 
#        gradient = (deri_disc * field.transform()).transform()
        
    #--- Compute Divergence
#        divergence = (deri_cont.vec_dot(a.transform())).transform()
#        print("Divergence_a: ", div(a,deri_cont))        #divergence of a
    #--- Compute Curl --->
#        curl = (deri_cont.cross(a.transform())).transform()
#        print("Curl_a: ", curl(a,deri_cont))       #curl of a
        return gradient
        
    def _multiply(self, x, **kwargs):
        """
        Parameters
        -----------
        x : array
            takes signal as input
            
        this is the pixel shifting function, returns the LensedField
        
        d = R(s)
        """
        self.lensed = np.zeros(shape = (self.pixels, self.pixels)).astype(float)
        for i in np.arange(self.pixels):
            for j in np.arange(self.pixels):
                self.lensed[i][j] = x[self.yposition[i][j]][self.xposition[i][j]]
#        print("Lensed Field: ")
#        print(self.lensed.astype(float))
        LensedField = field(self.domain, val = self.lensed)
#        LensedField.plot(power = False)
        return LensedField
        
    def _adjoint_multiply(self, x, **kwargs): 
        """
        Parameters
        -----------
        x : array
            takes data as input
            
        this is an adjoint function that replaces the changes from the original field to zeros (loss of data)
        
        s_hat = R_dagger(d)
        
        the function also sums pixels that are shifted to the same location
        
        if s(alpha) = s_hat(beta): 
            then d(alpha) + d(beta)
        """
        adjointlensed = np.zeros(shape = (self.pixels, self.pixels)).astype(float)
        for i in np.arange(self.pixels):
            for j in np.arange(self.pixels):
            # Either if statement or "+=" in the else: statement
#                if (adjointlensed[self.yposition[i][j]][self.xposition[i][j]] == x[self.yposition[i][j]][self.xposition[i][j]]):
#                    adjointlensed[self.yposition[i][j]][self.xposition[i][j]] = self.lensed[i][j] + x[i][j]
#                else:
                adjointlensed[self.yposition[i][j]][self.xposition[i][j]] += x[i][j]
#        print("AdjointLensed: ")
#        print(adjointlensed.astype(float))
        AdjointLensedField = field(self.domain, val=adjointlensed)
#        RevLensedField.plot(power = False)
        return AdjointLensedField
        
        
        
        
        
        
        
        
        
        
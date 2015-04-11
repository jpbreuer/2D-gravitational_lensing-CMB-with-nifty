from vector_field import *

about.hermitianize.off()

def Nabla_discrete(k_space,s_space=None,kfield=None):
    """
        This function returns the difference operator represented in Fourier
        space.
    """
    
    if(about.hermitianize.status):
        print 'It is not recommended to calculate derivatives in'
        print 'Fourier space while hermitianize is on.'
        print 'Please type \'about.hermitianize.off()\' to switch off hermitianization. '
    
    if(kfield is None):
        kfield = position_field(k_space)
    
    if(s_space is None):
        s_space = k_space.get_codomain()
    
    Ndim = k_space.naxes()
    
    basis_vectors = [np.zeros(Ndim) for ii in range(Ndim) ]
    
    for ii in range(Ndim):
        basis_vectors[ii][ii] = s_space.vol[ii]
    
    Nabla = [exp((0.+1.j)*2*pi*kfield.vec_dot(basis_vectors[ii]))-1. for ii in range(Ndim)]
    val = np.array([Nabla[ii].val/s_space.vol[ii] for ii in range(len(s_space.vol))])
    Nabla = vector_field(k_space,Ndim=k_space.naxes(),target=s_space,val=val)
    
    return Nabla
    

def Nabla_continuous(k_space,s_space=None,kfield=None):
    """
        This function returns the differential operator represented in Fourier
        space.
        i 2 \pi \vec{k}
    """
    
    if(about.hermitianize.status):
        print 'It is not recommended to calculate derivatives in'
        print 'Fourier space while hermitianize is on.'
        print 'Please type \'about.hermitianize.off()\' to switch off hermitianization. '
    
    if(kfield is None):
        kfield = position_field(k_space)
    
    if(s_space is None):
        s_space = k_space.get_codomain()
        
    Nabla = (0.+1.j)*2*pi*kfield
    
    return Nabla
    

def curl(x,Nabla):
    """
        This function returns the curl of a field x.
        x needs to be a vector field.
        k_Nabla needs to be the Nabla operator in Fourier representation.
    """
    return (Nabla.cross(x.transform())).transform()


def div(x,Nabla):
    """
        This function returns the divergence of a field x.
        x needs to be a vector field.
        k_Nabla needs to be the Nabla operator in Fourier representation.
    """
    return (Nabla.vec_dot(x.transform())).transform()

def grad(x,Nabla):
    """
        This function returns the gradient of a field x.
        x needs to be a scalar field.
        k_Nabla needs to be the Nabla operator in Fourier representation.
    """
    return (Nabla*(x.transform())).transform()
    

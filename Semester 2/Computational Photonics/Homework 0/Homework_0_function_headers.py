''' Python module containing functions to use the transfer matrix method.
'''

import numpy as np
from matplotlib import pyplot as plt

def transfermatrix(thickness, epsilon, polarisation, wavelength, kz):
    '''Computes the transfer matrix for a given stratified medium.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : float
        The wavelength of the incident light in µm.
    kz : float
        Transverse wavevector in 1/µm.
        
    Returns
    -------
    M : 2d-array
        The transfer matrix of the medium.
    '''
    pass 


def spectrum(thickness, epsilon, polarisation, wavelength, angle_inc, n_in, n_out):

    '''Computes the reflection and transmission of a stratified medium.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : 1d-array
        The wavelength of the incident light in µm.
    angle_inc : float
        The angle of incidence in degree (not radian!).
    n_in, n_out : float
        The refractive indices of the input and output layers.
        
	Returns
    -------
    t : 1d-array
        Transmitted amplitude
    r : 1d-array
        Reflected amplitude
    T : 1d-array
        Transmitted energy
    R : 1d-array
        Reflected energy
    '''
    pass



def field(thickness, epsilon, polarisation, wavelength, kz, n_in, n_out, Nx, l_in, l_out):
    '''Computes the field inside a stratified medium.
    
    The medium starts at x = 0 on the entrance side. The transmitted field
    has a magnitude of unity.
    
    Parameters
    ----------
    thickness : 1d-array
        Thicknesses of the layers in µm.
    epsilon : 1d-array
        Relative dielectric permittivity of the layers.
    polarisation : str
        Polarisation of the computed field, either 'TE' or 'TM'.
    wavelength : float
        The wavelength of the incident light in µm.
    kz : float
        Transverse wavevector in 1/µm.            
	n_in, n_out : float
        The refractive indices of the input and output layers.
    Nx : int
        Number of points where the field will be computed.
    l_in, l_out : float
        Additional thickness of the input and output layers where the field will be computed.
        
    Returns
    -------
    f : 1d-array
        Field structure
    index : 1d-array
        Refractive index distribution
    x : 1d-array
        Spatial coordinates
    '''    
    pass

def bragg(n1, n2, d1, d2, N):
    '''Generates the stack parameters of a Bragg mirror
    The Bragg mirror starts at the incidence side with layer 1 and is
    terminated by layer 2

    Parameters
    ----------
    n1, n2 : float or complex
        Refractive indices of the layers of one period
    d1, d2 : float
        Thicknesses of layers of one period
    N  : in
        Number of periods

    Returns
    -------
    epsilon : 1d-array
        Vector containing the permittivities
    thickness : 1d-array
        Vector containing the thicknesses
    '''
    pass

def timeanimation(x, f, index, steps, periods):
    ''' Animation of a quasi-stationary field.
    
    Parameters
    ----------
    x : 1d-array
        Spatial coordinates
    f : 1d-array
        Field
    index : 1d-array
        Refractive index
    steps : int
        Total number of time points
    periods : int
        Number of the oscillation periods.
    
    '''
    pass

    
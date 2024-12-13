import numpy as np
import scipy.special as sci_sp
import scipy as sp
import math
import cmath # for complex math
import fractions

# function definition to compute magnitude of the vector
def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def to_cartesian(radius, theta, phi):
    """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(theta)
    return (x, y, z)

def calc_delta(n, mp, m):
    delta = np.sqrt((sci_sp.gamma(n+mp+1) * sci_sp.gamma(n-mp+1)) / (sci_sp.gamma(n+m+1) * sci_sp.gamma(n-m+1))) * 1 / (2**mp) * sci_sp.jacobi(n-mp, abs(mp-m), abs(mp+m))(0)
    return delta

def interpft(a, ny):

    # Operates on the last axis of a 2D array
    axis = -1

    # Get initial length and width of the 2D array
    n, m = np.shape(a)

    # Ensure that ny is an integer
    ny = np.floor(ny)

    # If necessary, increase ny by an integer multiple to make ny > size(a,axis)
    if ny <= 0:
        raise Exception("n must be an integer greater than 0.")
    elif np.size(a, axis) > m:
        incr = 1
    else:
        incr = np.floor(m/ny) + 1
        ny *= incr

    b = np.fft.fft(a, axis=axis)

    nyqst = np.ceil((m + 1.0)/2.0)

    c = np.zeros((n, ny), dtype='complex')
    c[:, 0:nyqst] = b[:, 0:nyqst]
    c[:, nyqst+(ny-m):] = b[:, nyqst:]

    if np.remainder(m, 2) == 0:
        c[:, nyqst-1] = c[:, nyqst-1]/2.0
        c[:, nyqst+ny-m-1] = c[:, nyqst-1]

    d = np.fft.ifft(c, axis=axis)

    d *= float(ny)/float(m)

    d = d[:, 0::incr]  # Skip over extra points when original ny <= m

    return d

def singlesphere2doublesphere(singlesphere):

    numthetas = np.size(singlesphere, axis=0)
    numphis = np.size(singlesphere, axis=1)

    if numphis % 2 == 1:
        ss = interpft(singlesphere, numphis-1)
        numphis -= 1
    else:
        ss = singlesphere.copy()

    doublesphere = np.zeros((2*(numthetas-1), numphis), dtype='complex')

    doublesphere[:numthetas, :] = ss[:, :]
    doublesphere[numthetas:, :] = -np.roll(ss[-2:0:-1, :], int(numphis/2), axis=1)

    return doublesphere

def delta_pyramid(n_max):
    # Function used to calculate the delta pyramid.
    # The delta pyramid is defined in [1], Section A2.4

    # Region Diagram
    #
    #      |<--(-m)---m---(+m)-->|
    #  -    ---------------------
    #  ˄   |          |          |
    #  |   |          |          |
    # -m'  |    IV    |   III    |
    #  |   |          |          |
    #  |   |          |          |
    #  m'  |---------------------|
    #  |   |          |          |
    #  |   |          |          |
    # +m'  |    II    |    I     |
    #  |   |          |          |
    #  ˅   |          |          |
    #  -    ---------------------

    # Set mp_max and m_max equal to n_max in order to simplify calculations
    mp_max = m_max = n_max

    # Initialize the matrix to store the entire delta pyramid
    deltas = np.zeros((n_max+1, 2*mp_max+1, 2*m_max+1))

    # Initialize two "helper" functions
    def get_mp_index(mp_): return mp_max + mp_

    def get_m_index(m_): return m_max + m_

    # Initialize the delta values for n==0 and n==1, [1],Section A2.6
    deltas[0, get_mp_index(0), get_m_index(0)] = 1.0
    deltas[1, get_mp_index(0), get_m_index(1)] = -np.sqrt(2.0)/2.0
    deltas[1, get_mp_index(1), get_m_index(0)] = np.sqrt(2.0)/2.0
    deltas[1, get_mp_index(1), get_m_index(1)] = 1.0/2.0

    # Initialize positive values of mp and m
    mp_vec = np.linspace(0, mp_max, mp_max+1)
    m_vec = np.linspace(0, m_max, m_max+1)

    # Promote mp_vec and m_vec to 2 dimensions
    # (this will facilitate broadcasting later)
    mp_array = np.reshape(mp_vec, (mp_max+1, 1))
    m_array = np.reshape(m_vec, (1, m_max+1))

    # Copy two common indices to their own variables
    ind_mp_0 = get_mp_index(0)
    ind_m_0 = get_m_index(0)

    # Traverse through each n level from n==2 to n==N
    for n in range(2, n_max+1):

        # Copy two more common indices to their own variables
        ind_mp_n = get_mp_index(n)
        ind_m_n = get_m_index(n)

        # Calculate this particular n level using the previous two n levels.
        # [1],(A2.35)
        term1 = -1.0/(np.sqrt((n+mp_array[0:n, :]) *
                                 (n-mp_array[0:n, :]) *
                                 (n+m_array[:, 0:n]) *
                                 (n-m_array[:, 0:n]))*(n-1))
        term2 = np.sqrt((n+mp_array[0:n, :]-1) *
                           (n-mp_array[0:n, :]-1) *
                           (n+m_array[:, 0:n]-1) *
                           (n-m_array[:, 0:n]-1))*n
        term3 = (2*n-1)*mp_array[0:n, :]*m_array[:, 0:n]

        deltas[n, ind_mp_0:ind_mp_n, ind_m_0:ind_mp_n] = term1*(
            term2*deltas[n-2, ind_mp_0:ind_mp_n, ind_m_0:ind_mp_n] +
            term3*deltas[n-1, ind_mp_0:ind_mp_n, ind_m_0:ind_mp_n])

        # Initialize the vector full of this iteration's n value
        n_vec = n*np.ones(n+1)
        # Calculate the bottom edge of Region I, [1],(A2.41)
        temp = 1/(2.0**n)*np.sqrt(sci_sp.binom(2*n_vec, n_vec - m_vec[0:n+1]))
        deltas[n, ind_mp_n, ind_m_0:ind_m_n+1] = temp
        # Copy the bottom edge of Region I to the right edge of Region I
        # using the symmetry relationship [1],(A2.26) with m set equal to n.
        deltas[n, ind_mp_0:ind_mp_n, ind_m_n] = (-1)**(mp_vec[0:n] + n)*temp[0:n]

    # Create new n_vec, mp_vec, and m_vec vectors
    n_vec = np.linspace(0, n_max, n_max+1)
    mp_vec = np.linspace(-mp_max, mp_max, 2*mp_max+1)
    m_vec = np.linspace(-m_max, m_max, 2*m_max+1)

    # Promote n_vec, mp_vec, and m_vec to 3 dimensions
    # (this will facilitate broadcasting later)
    n_array = np.reshape(n_vec, (n_max+1, 1, 1))
    mp_array = np.reshape(mp_vec, (1, 2*mp_max+1, 1))
    m_array = np.reshape(m_vec, (1, 1, 2*m_max+1))

    # Compute Region II using symmetry relation [1],(A2.32)
    mmin2max = ((-1)**(n_array + mp_array[:, get_mp_index(0):, :]) *
             deltas[:, get_mp_index(0):, get_m_index(1):])
    mmin2max = mmin2max[:, :, ::-1]
    deltas[:, get_mp_index(0):, 0:get_m_index(0)] = mmin2max

    # Compute Region III using symmetry relation [1],(A2.28)
    nmin2max = ((-1)**(n_array + m_array[:, :, get_m_index(0):]) *
             deltas[:, get_mp_index(1):, get_m_index(0):])
    nmin2max = nmin2max[:, ::-1, :]
    deltas[:, 0:get_mp_index(0), get_m_index(0):] = nmin2max

    # Compute Region IV using symmetry relation [1],(A2.30)
    temp3 = deltas[:, get_mp_index(1):, get_m_index(1):]
    temp3 = temp3[:, ::-1, ::-1].transpose((0, 2, 1))
    deltas[:, 0:get_mp_index(0), 0:get_m_index(0)] = temp3

    # Return the deltas from the function
    return deltas

def step12(n_max):

    jj = np.linspace(-2*n_max+1, 2*n_max, 4*n_max)

    even_indices = jj % 2 == 0

    pi_wig = np.zeros(4*n_max)

    pi_wig[even_indices] = 2.0/(1.0 - jj[even_indices]**2)

    pi_wig = np.roll(pi_wig, -2*n_max+1)

    pi_wig = np.reshape(pi_wig,(4*n_max,1,1))

    return pi_wig

def Phertzian(frequency_Hz, n_max, dist):
    # calculate relevant constants
    c = 3e8 # light speed [m/s]
    λ = c / frequency_Hz # wavelength [m]
    β = (2*np.pi) / λ # wavenumber [1/m]

    #Calculate Input Coefficients
    g_radial_function = np.zeros((n_max, 2), dtype=complex)
    PHertzian = np.zeros((n_max, 2, 2), dtype=complex)
    for n_idx, n in enumerate(range(1, n_max+1)):
        z = β * dist
        g_radial_function[n_idx, 0] = sci_sp.spherical_jn(n, z, derivative=False) + 1j*sci_sp.spherical_yn(n, z, derivative=False) # equaivalent to R^3 s = 1
        g_radial_function[n_idx, 1] = ((1/z) * g_radial_function[n_idx, 0]) + (sci_sp.spherical_jn(n, z, derivative=True) - 1j*sci_sp.spherical_yn(n, z, derivative=True)) # equvalent to R^3 s = 2
        #Calculating P for Hertzian dipole
        PHertzian[n_idx, 0, 0] = np.sqrt(6)/8 * 1j**-1 * np.sqrt(2*n+1) * g_radial_function[n_idx, 0]
        PHertzian[n_idx, 1, 0] = np.sqrt(6)/8 * 1j**-2 * np.sqrt(2*n+1) * g_radial_function[n_idx, 1]
        PHertzian[n_idx, 0, 1] = - np.sqrt(6)/8 * 1j**1 * np.sqrt(2*n+1) * g_radial_function[n_idx, 0]
        PHertzian[n_idx, 1, 1] = - np.sqrt(6)/8 * 1j**2 * np.sqrt(2*n+1) * g_radial_function[n_idx, 1]
    return PHertzian

def b_wiggle(b_l_m_mu):

    numrows, numcolumns, numpages = np.shape(b_l_m_mu)
    n_max = (numrows-1)/2
    m_max = (numcolumns-1)/2

    b_l_m_mu_wiggle = np.zeros((int(4*n_max), int(2*m_max+1), 2), dtype=complex)

    def get_l_index(l_): return int(l_ + 2*n_max - 1)

    b_l_m_mu_wiggle[get_l_index(-n_max):get_l_index(n_max)+1, :, :] = b_l_m_mu

    b_l_m_mu_wiggle = np.roll(b_l_m_mu_wiggle, shift=int(-2*n_max+1), axis=0)

    return b_l_m_mu_wiggle

def wavecoeffs2farfield_uniform(q_n_m_s, thetapoints, phipoints, frequency, dist):
    # Taken from pysnf by rcutshall
    # Determine n_max and m_max from q_n_m_s
    n_max, m_max, s_max = np.shape(q_n_m_s)
    m_max = (m_max - 1)/2

    # Determine the number of theta and phis points required based on dT and dP
    numthetas = thetapoints
    numphis = phipoints

    # Get the dipole probe response constants
    # (N x 2 x 2, where dim 0 = n, dim 1 = mu, dim 2 = s)
    p_n_mu_s = Phertzian(frequency_Hz=frequency, n_max=n_max, dist=dist)

    # Copy out the mu==1, s==1 probe response constants
    p_n = np.reshape(p_n_mu_s[:, 0, 0], (n_max,1,1))  # N x M x Theta

    # Calculate the rotation coefficients
    # (N x 3 x M x numThetas)
    d_n_mu_m = rotation_coefficients(n_max, m_max, 1, np.linspace(0, np.pi, numthetas))

    # Calculate the addition and subtraction of the rotation coefficients
    # (N x M x numThetas)
    dp1_plus_dm1 = d_n_mu_m[:, 2, :, :] + d_n_mu_m[:, 0, :, :]  # N x M x Theta
    dp1_minus_dm1 = d_n_mu_m[:, 2, :, :] - d_n_mu_m[:, 0, :, :]

    # Reshape the spherical wave coefficients to get ready for multiplication
    q_n_m_s = np.reshape(q_n_m_s, (n_max,int(2*m_max+1),s_max,1))  # N x M x S x Theta

    # Perform the N summation as detailed in [1], (4.135), AND A FUTURE BLOG POST
    temp = p_n*(q_n_m_s[:, :, 0, :]*dp1_plus_dm1 + q_n_m_s[:, :, 1, :]*dp1_minus_dm1)  # N x M x Theta
    n_sum_chi_0 = np.swapaxes(np.sum(temp, 0), 0, 1)  # Theta x M

    # Perform the M summations as detailed in [1], (4.135), using an FFT
    n_sum_chi_0 = np.fft.ifftshift(n_sum_chi_0, axes=1)
    temp = np.zeros((numthetas,numphis),dtype='complex')
    temp[:,0:int(m_max+1)] = n_sum_chi_0[:,0:int(m_max+1)]
    temp[:,int(numphis-m_max):] = n_sum_chi_0[:,int(m_max+1):]
    theta_pol = np.fft.fft(temp, axis=1)

    # Perform the N summation as detailed in [1], (4.135), AND A FUTURE BLOG POST
    temp = 1j*p_n*(q_n_m_s[:, :, 0, :]*dp1_minus_dm1 + q_n_m_s[:, :, 1, :]*dp1_plus_dm1)  # N x M x Theta
    n_sum_chi_90 = np.swapaxes(np.sum(temp, 0), 0, 1)  # Theta x M

    # Perform the M summations as detailed in [1], (4.135), using an FFT
    n_sum_chi_90 = np.fft.ifftshift(n_sum_chi_90, axes=1)
    temp = np.zeros((numthetas,numphis),dtype='complex')
    temp[:,0:int(m_max+1)] = n_sum_chi_90[:,0:int(m_max+1)]
    temp[:,int(numphis-m_max):] = n_sum_chi_90[:,int(m_max+1):]
    phi_pol = np.fft.fft(temp, axis=1)

    return theta_pol, phi_pol

def rotation_coefficients(n_max, m_max, mu_max, thetas):
    # Taken from pysnf by rcutshall
    # Function used to calculate the rotation coefficients.
    # The rotation coefficients are defined in [1],Section A2.3
    m_max = int(m_max)
    n_max = int(n_max)
    # Make sure that the thetas variable is a numpy array
    if isinstance(thetas, float):
        thetas = np.array([thetas])
    else:
        thetas = np.array(thetas)

    # Make sure that MU <= M <= N
    if not(mu_max <= m_max <= n_max):
        raise Exception('MU must be less than or equal to M, '
                        'which in turn must be less than or '
                        'equal to N.')

    # Also make sure that MU >= 1
    if mu_max < 1:
        raise Exception('MU must be greater than or equal to 1.')

    # Make sure that all theta values are between 0 and pi
    if np.any(thetas < 0) or np.any(thetas > np.pi):
        raise Exception('theta values must be between 0 and pi, inclusive.')

    # For mu == -1, 0, or 1, calculate the rotation coefficients using [1],(A2.17),
    # (A2.18), and (A2.19). This is done to improve computation speed.
    ## print 'Calculating rotation coefficients for mu = -1, 0, and 1'

    # Calculate the normalized associated Legendre function values,
    # and the values of the derivative with respect to cos(theta)
    legendre_norm, dlegendre_norm = lpmn_norm(n_max, m_max, thetas)

    # Extend the legendre_norm array to negative values
    # of m (to ease future calculations), but set the arrays
    # such that legendre_norm of -m is equal to legendre_norm of m.
    # Also remove the n == 0 part of the array.
    mmin2max = int(2*m_max+1)
    nmin2max = int(2*n_max+1)
    lpmn_norm_extended = np.zeros((n_max, mmin2max, len(thetas)))
    lpmn_norm_extended[:, 0:m_max, :] = np.fliplr(legendre_norm[1:, 1:, :])
    lpmn_norm_extended[:, m_max:, :] = legendre_norm[1:, :, :]

    # Extend the dlegendre_norm array to negative values
    # of m (to ease future calculations), but set the arrays
    # such that legendre_norm of -m is equal to legendre_norm of m.
    # Also remove the n == 0 part of the array.
    dlpmn_norm_extended = np.zeros((n_max, mmin2max, len(thetas)))
    dlpmn_norm_extended[:, 0:m_max, :] = np.fliplr(dlegendre_norm[1:, 1:, :])
    dlpmn_norm_extended[:, m_max:, :] = dlegendre_norm[1:, :, :]

    # Initialize the matrix that will hold the rotation coefficients
    d = np.zeros((n_max, 3, mmin2max, len(thetas)))

    # Create arrays that hold the n, m, and mu indices. Also promote
    # the thetas to a 4-dimensional array. This will come in handy
    # when later multiplying the matrices. No need to perform
    # a matrix replication because of the way that Numpy "broadcasts"
    # the arrays during multiplication.
    n_vec = np.linspace(1, n_max, n_max)
    n_array = np.reshape(n_vec, (n_max, 1, 1, 1))
    mu_vec = np.linspace(-1, 1, 3)
    mu_array = np.reshape(mu_vec, (1, 3, 1, 1))
    m_vec = np.linspace(-m_max, m_max, 2*m_max+1)
    m_array = np.reshape(m_vec, (1, 1, 2*m_max+1, 1))

    # Calculate the (-m/m)**m factor, using [1],(2.19) to handle
    # the case when m==0
    nonzeroinds = m_array != 0
    mm = np.ones(np.shape(m_array))
    mm[nonzeroinds] = (-m_array[nonzeroinds] /
                       abs(m_array[nonzeroinds]))**m_array[nonzeroinds]

    # Calculate leading terms that are used in [1],(A2.17),(A2.18),
    # and (A2.19)
    c1 = mm*np.sqrt(2.0/(2.0*n_array+1))
    c2 = -2.0/np.sqrt(n_array*(n_array+1.0))

    # Calculate the mu==0 rotation coefficients according to (A2.17)
    d_0 = c1[:, 0, :, :]*lpmn_norm_extended

    # Initialize the d_plus1 and d_minus1 arrays
    d_plus1 = np.zeros([n_max, mmin2max, len(thetas)])
    d_minus1 = np.zeros([n_max, mmin2max, len(thetas)])

    # Calculate the mu==-1 and mu==1 rotation coefficients by
    # solving for d_-1 and d_+1 using equations (A2.18) and (A2.19).
    # Only calculate for values of theta where 1e-6 <= theta <= pi-1e-6.
    inds = np.logical_and((thetas >= 1e-6), (thetas <= np.pi-1e-6))
    d_plus1__plus__d_minus1 = (c2[:, 0, :, :]*m_array[:, 0, :, :])*d_0[:, :, inds]/np.sin(thetas[inds])
    d_plus1__minus__d_minus1 = (c1[:, 0, :, :]*c2[:, 0, :, :])*dlpmn_norm_extended[:, :, inds]
    d_minus1[:, :, inds] = (d_plus1__plus__d_minus1 - d_plus1__minus__d_minus1)/2.0
    d_plus1[:, :, inds] = (d_plus1__plus__d_minus1 + d_plus1__minus__d_minus1)/2.0

    # Assign d_minus1, d_0, and d_plus1 back into the d array
    d[:, 0, :, :] = d_minus1
    d[:, 1, :, :] = d_0
    d[:, 2, :, :] = d_plus1

    # Calculate the special cases when theta < 1e-6 or theta > pi-1e-6
    inds = thetas < 1e-6
    d[:, :, :, inds] = mu_array == m_array
    inds = thetas > np.pi-1e-6
    d[:, :, :, inds] = (-1)**(n_array+m_array)*(mu_array == -m_array)

    # If MU > 1, calculate the mu != -1, 0, or 1 rotation coefficients using [1],(A2.11)
    if mu_max > 1:

        # Store d for mu == -1,0,1 to a temp variable
        d_temp = d

        # Output shaped like this: (n,mu,m,theta)
        d = np.zeros((n_max, int(2*mu_max+1), mmin2max, len(thetas)))

        # Define a helper function
        def get_mu_index(mu_): return mu_max + mu_

        # Place d_temp back inside of d
        d[:, get_mu_index(-1):get_mu_index(1)+1, :, :] = d_temp

        # Calculate the delta pyramid
        deltas = delta_pyramid(n_max)

        # Create the mp and m vectors
        mp_vec = np.linspace(-n_max, n_max, nmin2max)
        m_vec = np.linspace(-m_max, m_max, mmin2max)

        # Remove the extra m values from the delta pyramid, if necessary
        extra = (len(mp_vec)-len(m_vec))/2
        if extra > 0:
            deltas = deltas[:, :, extra:-extra]

        # promote mp_vec to 4 dimensions
        mp_array = np.reshape(mp_vec, (1, nmin2max, 1, 1))

        # Assign len(thetas) to a variable before we promote it to 4 dimensions
        numthetas = len(thetas)

        # promote thetas to 4 dimensions
        thetas = np.reshape(thetas, (1, 1, 1, len(thetas)))

        # Delete n==0 in deltas
        deltas = np.delete(deltas, 0, 0)

        # promote the deltas to 4 dimensions
        deltas4d = np.reshape(deltas, (n_max, nmin2max, mmin2max, 1))

        # Calculate the exponent matrix
        expon = np.exp(-1j*mp_array*thetas)

        # promote the m_vec to 2 dimensions
        m_array = np.reshape(m_vec, (1, mmin2max))

        # define a helper function
        def get_m_index(m_): return m_max + m_

        # Calculate the summation for mu != -1,0,1
        mu_vec = np.linspace(-mu_max, mu_max, int(2*mu_max+1))
        mu_vec = mu_vec[np.logical_or(mu_vec < -1, mu_vec > 1)]
        for mu in mu_vec:
            ## print 'Calculating rotation coefficients for mu =', mu
            # Calculate the kernel
            temp = np.reshape(deltas4d[:, :, get_m_index(mu), :], (n_max, nmin2max, 1, 1))
            kernel = temp*deltas4d
            for nt in range(numthetas):
                d[:, get_mu_index(mu), :, nt] = ((1j**(mu-m_array))*np.sum(kernel[:, :, :, 0] *
                                                                           expon[:, :, :, nt], 1)).real
    return d

def lpmn_norm(n_max, m_max, thetas):
    # Returns the normalized associated Legendre function values of
    # cos(theta), as defined in [1],(A1.25). Also returns the derivative
    # of the normalized associated Legendre function values, where the
    # derivative is taken with respect to cos(theta).

    # Make sure that the thetas variable is a numpy array
    if isinstance(thetas, float):
        thetas = np.array([thetas])
    else:
        thetas = np.array(thetas)

    # Make sure that M <= N
    if m_max > n_max:
        raise Exception('M must be less than or equal to N.')

    # Make sure that all theta values are between 0 and pi
    if np.any(thetas < 0) or np.any(thetas > np.pi):
        raise Exception('theta values must be between 0 and pi, inclusive.')

    # Initialize the matrices which will store the associated Legendre
    # function values. legendre_m_n_thetas will contain the associated Legendre
    # function values, whereas dlegendre_m_n_thetas will contain the derivative
    # of the associated Legendre function ( derivative taken with
    # respect to cos(theta) )
    tempm = int(m_max+1)
    tempn = int(n_max+1)
    legendre_m_n_thetas = np.zeros((tempm, tempn, len(thetas)))
    dlegendre_m_n_thetas = np.zeros((tempm, tempn, len(thetas)))

    # Loop through all theta values, using the scipy.special.lpmn
    # function to calculate the associated Legendre function values.
    for tt in range(1, len(thetas)): #Small change here, so that there is no legendre of 0.
        legendre_m_n_thetas[:, :, tt], dlegendre_m_n_thetas[:, :, tt] = sci_sp.lpmn(m_max, n_max, np.cos(thetas[tt]))
        dlegendre_m_n_thetas[:, :, tt] = dlegendre_m_n_thetas[:, :, tt]*-np.sin(thetas[tt])
    # Next, we wish to obtain the normalized associated Legendre functions
    # (and the derivatives) from the associated Legendre functions.
    # Therefore, we must calculate the normalization factors,
    # as given in [1],(A1.25). The math.factorial and fractions.Fraction
    # functions are used to prevent numerical overflow from occurring.
    # Also, it should be noted that [1] defines the associated Legendre
    # function differently than the scipy.special package. Therefore,
    # we must multiply the scipy.special values by (-1)**m in order
    # to have our normalized associated Legendre function values agree
    # with those given in the table in [1],pg.322
    normfactor = np.zeros((tempm, tempn, 1))
    for m in range(tempm):
        for n in range(tempn):
            if m > n:
                continue
            mmin2max = math.factorial(n-m)
            nmin2max = math.factorial(n+m)
            temp3 = fractions.Fraction(mmin2max, nmin2max)
            normfactor[m, n, 0] = (-1)**m*math.sqrt((2*n+1)/2.0*float(temp3))

    # Multiply the associate Legendre function values by the
    # normalization factors to obtain the normalized associated
    # Legendre function values. Also, take this opportunity to copy
    # the values such that legendre_m_n_thetas has dimensions N x (2*M+1) x len(thetas)
    legendre_m_n_thetas = legendre_m_n_thetas*normfactor
    legendre_m_n_thetas = legendre_m_n_thetas.transpose((1, 0, 2))  # now arranged as (N, M, THETA)

    # Multiply the associate Legendre function derivative values by the
    # normalization factors to obtain the derivative of the normalized
    # associated Legendre function values ( derivative taken with respect
    # to cos(theta) ). Also, take this opportunity to copy
    # the values such that dlegendre_m_n_thetas has dimensions N x (2*M+1) x len(thetas)
    dlegendre_m_n_thetas = dlegendre_m_n_thetas*normfactor
    dlegendre_m_n_thetas = dlegendre_m_n_thetas.transpose((1, 0, 2))  # now arranged as (N, M, THETA)

    return legendre_m_n_thetas, dlegendre_m_n_thetas


def normalize_near_field_data(nf_data):
    """
    Normalize the near-field data by scaling E_theta and E_phi to have a maximum value of 1.
    """

    # Extract components
    E_theta = nf_data[:, :, 0]
    E_phi = nf_data[:, :, 1]
    
    # Normalize E_theta and E_phi magnitudes
    E_theta /= np.max(np.abs(E_theta))
    E_phi /= np.max(np.abs(E_phi))
    
    # Update the nf_data array
    nf_data[:, :, 0] = E_theta
    nf_data[:, :, 1] = E_phi

    return nf_data

def spherical_far_field_transform(nf_data, theta_f, phi_f, max_l):
    """
    Compute and return the normalized far-field pattern from the near-field data.
    The sizes and ranges of theta and phi are determined from the nf_data array.
    """

    # Normalize the near-field data first
    #nf_data = normalize_near_field_data(nf_data)    
    #nf_data = np.abs(np.real(nf_data))

    # Create a meshgrid for the far-field angles
    #phi_f_grid, theta_f_grid = np.meshgrid(phi_f, theta_f, indexing='ij')
    phi_f_grid, theta_f_grid = np.meshgrid(theta_f, phi_f, indexing='ij')

    E_tot = nf_data.flatten()

    # Compute spherical harmonic coefficients a_lm
    a_lm = np.zeros((max_l + 1, 2 * max_l + 1), dtype=complex)
    
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = sci_sp.sph_harm(m, l, theta_f_grid.flatten(), phi_f_grid.flatten())
            Y_lm_conj = np.conjugate(Y_lm)
            a_lm[l, m + l] = np.sum(E_tot * Y_lm_conj)  # Use E_theta only for simplicity

    # Calculate the far-field pattern from coefficients
    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm_f = sci_sp.sph_harm(m, l, phi_f_grid, theta_f_grid)
            E_far += a_lm[l, m + l] * Y_lm_f

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude


def spherical_far_field_transform_cook(nf_data, theta_f, phi_f, Δθ, Δφ, frequency_Hz, nf_meas_dist = 0.2, N = 3, M = 3):
    # refactor to either calculate frauhofer distance or as a parameter
    transform_dist = 3.2 # the distance in meters to transform NF to FF at ex. FF at 5 meters

    # validate parameters
    if N < 1:
        raise ValueError(f"N: {N} can't be less than 1")
    
    if M > N:
        raise ValueError(f"M: {M} can't be greater than N: {N}")


    # calculate relevant constants
    c = 3e8 # light speed [m/s]
    λ = c / frequency_Hz # wavelength [m]
    β = (2*np.pi) / λ # wavenumber [1/m]

    n_range = range(1, N+1) # N+1 as end is exclusive
    m_range = range(-N, N+1) # M+1 as end is exclusive

    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    def calc_legendre(n, m, cos_θ):
        legendre_inner = math.sqrt((((2 * n) + 1) / (4 * np.pi)) * (math.factorial(abs(n - m)) / math.factorial(abs(n + m))))
        return legendre_inner * sci_sp.lpmv(abs(m), n, cos_θ)

    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.20{a,b}  
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: F_{1}; index 1: F_{2}
    f_angular_spherical_harmonics = np.zeros((theta_size, phi_size, len(n_range), len(m_range), 2, 3), dtype=complex)
    F_angular_spherical_harmonics = np.zeros((f_angular_spherical_harmonics.shape), dtype=complex)
    for θ_idx, θ in enumerate(theta_f):
        cos_θ = math.cos(θ)
        sin_θ = math.sin(θ)

        for φ_idx, φ in enumerate(phi_f):
            cos_φ = math.cos(φ)
            sin_φ = math.sin(φ)
            φ_hat = np.asarray([-sin_φ, cos_φ, 0])
            θ_hat = np.asarray([cos_φ, sin_φ, -sin_θ]) * (1.0 / math.sqrt(1 + sin_θ**2))

            for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
                for m in range(-n, n+1):
                    m_idx = m_range.index(m)    
                    m_term = 1
                    if m != 0:
                        m_term = (-m / abs(m))**m

                    legendre = calc_legendre(n, m, cos_θ)
                    legendre_next_order = calc_legendre(n+1, m, cos_θ)
                    if sin_θ == 0:
                        sin_θ = 0.00001

                    n_term = 1/(np.sqrt(2*np.pi*n*(n+1)))

                    frac_legrende = ((1j * m) / sin_θ) * legendre
                    deriv_legrende = (1.0 / sin_θ) * (((n - m + 1) * legendre_next_order) - ((n + 1) * cos_θ * legendre))

                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * n_term *((frac_legrende * θ_hat) - (deriv_legrende * φ_hat))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * n_term * ((deriv_legrende * θ_hat) + (frac_legrende * φ_hat))
                    
                    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.19
                    ejmφ = cmath.exp(1j * m * φ)
                    F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] * ejmφ
                    F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] * ejmφ

    # calculate g1n and g2n, implementation of report eq. 2.18{a,b}
    # the implementation of hankel functions are calculated using eq. A.16
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: g_{1}; index 1: g_{2}
    g_radial_function = np.zeros((N, 2), dtype=complex)
    for n_idx, n in enumerate(n_range):
        z = β * nf_meas_dist
        g_radial_function[n_idx, 0] = sci_sp.spherical_jn(n, z, derivative=False) - 1j*sci_sp.spherical_yn(n, z, derivative=False) # equvalent to g_{1n}
        g_radial_function[n_idx, 1] = ((1/z) * g_radial_function[n_idx, 0]) + (sci_sp.spherical_jn(n, z, derivative=True) - 1j*sci_sp.spherical_yn(n, z, derivative=True)) # equvalent to g_{2n}

    # calculate spherical expansion coefficients a1nm and a2nm, implementation of report eq. 2.22
    a_spherical_wave_expansion_coefficient = np.zeros((len(n_range), len(m_range), 2, 3), dtype=complex)
    for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
        frac_g1 = 1.0 / (β * g_radial_function[n_idx, 0])
        frac_g2 = 1.0 / (β * g_radial_function[n_idx, 1])
        for m in range(-n, n+1):
            m_idx = m_range.index(m)
            double_sum_f1 = complex(0.0, 0.0)
            double_sum_f2 = complex(0.0, 0.0)
            for φ_idx, φ in enumerate(phi_f): # equivalent to integral from zero -> 2*pi
                e = cmath.exp(1j * m * φ)
                for θ_idx, θ in enumerate(theta_f): # equivalent to integral from zero -> pi
                    E_tot = nf_data[θ_idx, φ_idx]
                    sin_θ = math.sin(θ)
                    f1 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0]
                    f2 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1]

                    double_sum_f1 += E_tot * np.conjugate(f1) * e * sin_θ * Δθ
                    double_sum_f2 += E_tot * np.conjugate(f2) * e * sin_θ * Δφ
            
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] = frac_g1 * double_sum_f1
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] = frac_g2 * double_sum_f2


    # implementation of report eq. 2.24
    constant_term = cmath.exp(-1j * β * transform_dist) / transform_dist
    for θ_idx, θ in enumerate(theta_f):
        for φ_idx, φ in enumerate(phi_f):
            double_sum = 0 # eq to the double sum
            #for n_idx, n in enumerate(n_range):
                #for m_idx, m in range(-n, n):
                #for m_idx, m in enumerate(m_range): # this range results in -M is index 0
            for n_idx, n in enumerate(n_range):
                for m in range(-n, n+1):
                    m_idx = m_range.index(m)

                    a1nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] # in report denoted: a_{1nm}
                    a2nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] # in report denoted: a_{2nm}
                    f1nm = F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] # in report denoted: f_{1nm}
                    f2nm = F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] # in report denoted: f_{2nm}

                    double_sum += (((1j**(n + 1)) * a1nm * f1nm) + ((1j**n) * a2nm * f2nm))

            vector_res_3d = constant_term * double_sum
            vector_res_3d = np.abs(vector_res_3d)
            vector_sphere_3d = to_cartesian(transform_dist, θ, φ)
            vector_efield_3d = ((vector_res_3d * vector_sphere_3d) / magnitude(vector_sphere_3d)**2) * vector_sphere_3d
            E_far[θ_idx, φ_idx] = magnitude(vector_efield_3d)

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude

def spherical_far_field_transform_cook_fft(nf_data, theta_f, phi_f, Δθ, Δφ, frequency_Hz, nf_meas_dist = 0.2, N = 3, M = 3):
    # refactor to either calculate frauhofer distance or as a parameter
    transform_dist = 150 # the distance in meters to transform NF to FF at ex. FF at 150 meters

    # validate parameters
    if N < 1:
        raise ValueError(f"N: {N} can't be less than 1")
    
    if M > N:
        raise ValueError(f"M: {M} can't be greater than N: {N}")


    # calculate relevant constants
    c = 3e8 # light speed [m/s]
    λ = c / frequency_Hz # wavelength [m]
    β = (2*np.pi) / λ # wavenumber [1/m]

    n_range = range(1, N+1) # N+1 as end is exclusive
    m_range = range(-N, N+1) # M+1 as end is exclusive

    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    def calc_legendre(n, m, cos_θ):
        legendre_inner = math.sqrt((((2 * n) + 1) / (4 * np.pi)) * (math.factorial(abs(n - m)) / math.factorial(abs(n + m))))
        return legendre_inner * sci_sp.lpmv(abs(m), n, cos_θ)

    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.20{a,b}  
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: F_{1}; index 1: F_{2}
    f_angular_spherical_harmonics = np.zeros((theta_size, phi_size, N, 2*M+1, 2, 3), dtype=complex)
    #F_angular_spherical_harmonics = np.zeros((f_angular_spherical_harmonics.shape), dtype=complex)
    for θ_idx, θ in enumerate(theta_f):
        cos_θ = math.cos(θ)
        sin_θ = math.sin(θ)

        for φ_idx, φ in enumerate(phi_f):
            cos_φ = math.cos(φ)
            sin_φ = math.sin(φ)
            φ_hat = np.asarray([-sin_φ, cos_φ, 0])
            θ_hat = np.asarray([cos_φ, sin_φ, -sin_θ]) * (1.0 / math.sqrt(1 + sin_θ**2))

            for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
                for m_idx, m in enumerate(m_range): # this range results in -M is index 0
                    m_term = 1
                    if m != 0:
                        m_term = (-m / abs(m))**m

                    legendre = calc_legendre(n, m, cos_θ)
                    legendre_next_order = calc_legendre(n+1, m, cos_θ)
                    if sin_θ == 0:
                        sin_θ = 1

                    frac_legrende = ((1j * m) / sin_θ) * legendre
                    deriv_legrende = (1.0 / sin_θ) * (((n - m + 1) * legendre_next_order) - ((n + 1) * cos_θ * legendre))

                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * ((frac_legrende * θ_hat) - (deriv_legrende * φ_hat))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * ((deriv_legrende * θ_hat) + (frac_legrende * φ_hat))
                    
                    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.19
                    #ejmφ = math.exp(math.e, 1j * m * φ)
                    #F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] * ejmφ
                    #F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] * ejmφ


    # calculate g1n and g2n, implementation of report eq. 2.18{a,b}
    # the implementation of hankel functions are calculated using eq. A.16
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: g_{1}; index 1: g_{2}
    g_radial_function = np.zeros((N, 2), dtype=complex)
    for n_idx, n in enumerate(n_range):
        z = β * nf_meas_dist
        g_radial_function[n_idx, 0] = sci_sp.spherical_jn(n, z, derivative=False) - 1j*sci_sp.spherical_yn(n, z, derivative=False) # equvalent to g_{1n}
        g_radial_function[n_idx, 1] = ((1/z) * g_radial_function[n_idx, 0]) + (sci_sp.spherical_jn(n, z, derivative=True) - 1j*sci_sp.spherical_yn(n, z, derivative=True)) # equvalent to g_{2n}

    # calculate spherical expansion coefficients a1nm and a2nm, implementation of report eq. 2.22
    m_step_size = int((nf_data.shape[1] / len(m_range))-1)
    E_2d_ifft = np.fft.fft2(nf_data)
    a_spherical_wave_expansion_coefficient = np.zeros((len(n_range), len(m_range), 2, 3), dtype=complex)
    for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
        frac_g1 = 1.0 / (β * g_radial_function[n_idx, 0])
        frac_g2 = 1.0 / (β * g_radial_function[n_idx, 1])
        for m_idx, m in enumerate(m_range): # this range results in -M is index 0
            double_sum_f1 = complex(0.0, 0.0)
            double_sum_f2 = complex(0.0, 0.0)
            f1 = f_angular_spherical_harmonics[:, 0, n_idx, m_idx, 0]
            f2 = f_angular_spherical_harmonics[:, 0, n_idx, m_idx, 1]
            f1_fft = np.fft.fft(f1)
            f2_fft = np.fft.fft(f2)

            for l_idx in range(0, E_2d_ifft.shape[0]):
                for i_idx in range(0, E_2d_ifft.shape[0]):
                    frac = 1j * (np.pi / 2)
                    if abs(l_idx - i_idx) != 1:
                        frac = -((1 + cmath.exp(1j * np.pi * (l_idx - i_idx))) / ((l_idx - i_idx)**2 - 1))

                    double_sum_f1 += E_2d_ifft[l_idx, m_idx * m_step_size] * np.conjugate(f1_fft[i_idx]) * frac
                    double_sum_f2 += E_2d_ifft[l_idx, m_idx * m_step_size] * np.conjugate(f2_fft[i_idx]) * frac

            # for φ_idx, φ in enumerate(phi_f): # equivalent to integral from zero -> 2*pi
            #     e = cmath.exp(1j * m * φ)
            #     for θ_idx, θ in enumerate(theta_f): # equivalent to integral from zero -> pi
            #         E_tot = nf_data[θ_idx, φ_idx]
            #         sin_θ = math.sin(θ)
            #         f1 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0]
            #         f2 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1]

            #         double_sum_f1 += E_tot * np.conjugate(f1) * e * sin_θ * Δθ
            #         double_sum_f2 += E_tot * np.conjugate(f2) * e * sin_θ * Δφ
            
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] = frac_g1 * double_sum_f1
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] = frac_g2 * double_sum_f2


    # implementation of report eq. 2.24
    constant_term = cmath.exp(-1j * β * transform_dist) / transform_dist
    for θ_idx, θ in enumerate(theta_f):
        for φ_idx, φ in enumerate(phi_f):
            double_sum = 0 # eq to the double sum
            #for n_idx, n in enumerate(n_range):
                #for m_idx, m in range(-n, n):
                #for m_idx, m in enumerate(m_range): # this range results in -M is index 0
            for m_idx, m in enumerate(m_range):
                for n in range(abs(m), N+1):
                    if n == 0:
                        continue
                    n_idx = n_range.index(n) # get the corresponding index of m

                    a1nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] # in report denoted: a_{1nm}
                    a2nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] # in report denoted: a_{2nm}
                    f1nm = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] # in report denoted: f_{1nm}
                    f2nm = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] # in report denoted: f_{2nm}

                    double_sum += (((1j**(n + 1)) * a1nm * f1nm) + ((1j**n) * a2nm * f2nm))

            vector_res_3d = constant_term * double_sum
            vector_res_3d = np.abs(vector_res_3d)
            vector_sphere_3d = to_cartesian(transform_dist, φ, θ)
            vector_efield_3d = ((vector_res_3d * vector_sphere_3d) / magnitude(vector_sphere_3d)**2) * vector_sphere_3d
            E_far[θ_idx, φ_idx] = magnitude(vector_efield_3d)

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude

def spherical_far_field_transform_cookV2(nf_data, theta_f, phi_f, Δθ, Δφ, frequency_Hz, nf_meas_dist = 0.2, N = 3, M = 3):
    # refactor to either calculate frauhofer distance or as a parameter
    transform_dist = 150 # the distance in meters to transform NF to FF at ex. FF at 150 meters

    # validate parameters
    if N < 1:
        raise ValueError(f"N: {N} can't be less than 1")
    
    if M > N:
        raise ValueError(f"M: {M} can't be greater than N: {N}")


    # calculate relevant constants
    c = 3e8 # light speed [m/s]
    λ = c / frequency_Hz # wavelength [m]
    β = (2*np.pi) / λ # wavenumber [1/m]

    n_range = range(1, N+1) # N+1 as end is exclusive
    m_max_range = range(-N, N+1) # M+1 as end is exclusive

    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    def calc_legendre(n, m, cos_θ):
        legendre_inner = math.sqrt((((2 * n) + 1) / (4 * np.pi)) * (math.factorial(abs(n - m)) / math.factorial(abs(n + m))))
        return legendre_inner * sci_sp.lpmv(abs(m), n, cos_θ)

    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.20{a,b}  
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: F_{1}; index 1: F_{2}
    f_angular_spherical_harmonics = np.zeros((theta_size, phi_size, len(n_range), len(m_max_range), 2), dtype=complex)
    #F_angular_spherical_harmonics = np.zeros((f_angular_spherical_harmonics.shape), dtype=complex)
    for θ_idx, θ in enumerate(theta_f):
        cos_θ = math.cos(θ)
        sin_θ = math.sin(θ)

        for φ_idx, φ in enumerate(phi_f):
            cos_φ = math.cos(φ)
            sin_φ = math.sin(φ)
            φ_hat = np.asarray([-sin_φ, cos_φ, 0])
            θ_hat = np.asarray([cos_φ, sin_φ, -sin_θ]) * (1.0 / math.sqrt(1 + sin_θ**2))

            for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
                for m_idx, m in enumerate(m_max_range): # this range results in -M is index 0
                    m_term = 1
                    if m != 0:
                        m_term = (-m / abs(m))**m

                    legendre = calc_legendre(n, m, cos_θ)
                    legendre_next_order = calc_legendre(n+1, m, cos_θ)
                    if sin_θ == 0:
                        sin_θ = 1

                    frac_legrende = ((1j * m) / sin_θ) * legendre
                    deriv_legrende = np.gradient(legendre)
                    print
                    print(deriv_legrende)

                    #f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * ((frac_legrende * θ_hat) - (deriv_legrende * φ_hat))
                    #f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * ((deriv_legrende * θ_hat) + (frac_legrende * φ_hat))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * ((frac_legrende) - (deriv_legrende))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * ((deriv_legrende) + (frac_legrende))
                    
                    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.19
                    #ejmφ = math.exp(math.e, 1j * m * φ)
                    #F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] * ejmφ
                    #F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] * ejmφ


    # calculate g1n and g2n, implementation of report eq. 2.18{a,b}
    # the implementation of hankel functions are calculated using eq. A.16
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: g_{1}; index 1: g_{2}
    g_radial_function = np.zeros((N, 2), dtype=complex)
    for n_idx, n in enumerate(n_range):
        z = β * nf_meas_dist
        g_radial_function[n_idx, 0] = sci_sp.spherical_jn(n, z, derivative=False) - 1j*sci_sp.spherical_yn(n, z, derivative=False) # equvalent to g_{1n}
        g_radial_function[n_idx, 1] = ((1/z) * g_radial_function[n_idx, 0]) + (sci_sp.spherical_jn(n, z, derivative=True) - 1j*sci_sp.spherical_yn(n, z, derivative=True)) # equvalent to g_{2n}

    # calculate spherical expansion coefficients a1nm and a2nm, implementation of report eq. 2.22
    a_spherical_wave_expansion_coefficient = np.zeros((len(n_range), len(m_max_range), 2), dtype=complex)
    for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
        frac_g1 = 1.0 / (β * g_radial_function[n_idx, 0])
        frac_g2 = 1.0 / (β * g_radial_function[n_idx, 1])
        for m_idx, m in enumerate(m_max_range): # this range results in -M is index 0
            double_sum_f1 = complex(0.0, 0.0)
            double_sum_f2 = complex(0.0, 0.0)
            for φ_idx, φ in enumerate(phi_f): # equivalent to integral from zero -> 2*pi
                e = cmath.exp(1j * m * φ)
                for θ_idx, θ in enumerate(theta_f): # equivalent to integral from zero -> pi
                    E_tot = nf_data[θ_idx, φ_idx]
                    sin_θ = math.sin(θ)
                    f1 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0]
                    f2 = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1]

                    double_sum_f1 += E_tot * np.conjugate(f1) * e * sin_θ * Δθ
                    double_sum_f2 += E_tot * np.conjugate(f2) * e * sin_θ * Δφ
            
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] = frac_g1 * double_sum_f1
            a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] = frac_g2 * double_sum_f2


    # implementation of report eq. 2.24
    constant_term = cmath.exp(-1j * β * transform_dist) / transform_dist
    for θ_idx, θ in enumerate(theta_f):
        for φ_idx, φ in enumerate(phi_f):
            double_sum = 0 # eq to the double sum
            for n_idx, n in enumerate(n_range):
                for m in range(-n, n+1):
                    m_idx = m_max_range.index(m) # get the corresponding index of m

                    a1nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] # in report denoted: a_{1nm}
                    a2nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] # in report denoted: a_{2nm}
                    f1nm = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] # in report denoted: f_{1nm}
                    f2nm = f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] # in report denoted: f_{2nm}

                    double_sum += (((1j**(n + 1)) * a1nm * f1nm) + ((1j**n) * a2nm * f2nm))

            vector_res_3d = constant_term * double_sum
            E_far[θ_idx, φ_idx] = vector_res_3d #math.sqrt(abs(vector_res_3d[0])**2 + abs(vector_res_3d[1])**2 + abs(vector_res_3d[2])**2)

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    #E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude

def spherical_far_field_transform_megacook(nf_data, theta_f, phi_f, Δθ, Δφ, frequency_Hz, nf_meas_dist = 3.2, N = 3, M = 3):
    
    # validate parameters
    if N > (nf_data.shape[0]-1)/2 or 1 > N:
        raise ValueError("N must uphold Table 4.4 in JE Hansen")
    
    n_range = range(1,N+1)
    j_range = range(0, 4*N) # Eq 4.86
    m_range = range(-N,N+1)
    zero_array1 = np.zeros((N-2, nf_data.shape[1]), dtype=complex)
    zero_array2 = np.zeros((N-1, nf_data.shape[1]), dtype=complex)

    PHertzian = Phertzian(frequency_Hz=frequency_Hz, n_max=N, nf_meas_dist=nf_meas_dist)

    NF_IFFT_phi_datamu1 = np.fft.ifft(nf_data[:, :, 0], axis = 1) #Equation 4.127
    NF_IFFT_phi_datamu2 = np.fft.ifft(nf_data[:, :, 1], axis = 1) #Equation 4.127
    T1 = np.zeros((2*N+1, N+1), dtype= complex)
    T2 = np.zeros((2*N+1, N+1), dtype= complex)
    for m_idx, m in enumerate(m_range):
        bl_mmu1 = np.fft.ifft(NF_IFFT_phi_datamu1, axis = 0) #Equation 4.128
        bl_mmu2 = np.fft.ifft(NF_IFFT_phi_datamu2, axis = 0) #Equation 4.128
        bl_tilte_mmu1 = np.concatenate((zero_array1, bl_mmu1, zero_array2)) #Equation 4.87
        bl_tilte_mmu2 = np.concatenate((zero_array1, bl_mmu2, zero_array2)) #Equation 4.87
        K_mmu1 = 0
        K_mmu2 = 0
        for l in j_range:
            PI_lm = 0
            if (l - m) % 2 == 0:
                PI_lm = 2 / (1 - ((l - m)**2))
            K_mmu1 += PI_lm * bl_tilte_mmu1 #Equation 4.130
            K_mmu2 += PI_lm * bl_tilte_mmu2 #Equation 4.130
        for n_idx, n in enumerate(n_range):
            w_n_uA_const = ((2*n+1)/2) * 1j**(-m)
            for m_mark in range(-n, n+1):
                w_n_uA1 = calc_delta(n, -1, m) * calc_delta(n, m_mark, m) * K_mmu1 #Equation 4.132
                w_n_uA2 = calc_delta(n, 1, m) * calc_delta(n, m_mark, m) * K_mmu2 #Equation 4.132
            w_n_uA1 *= w_n_uA_const
            w_n_uA2 *= w_n_uA_const
            T2[m_idx, n_idx] = ((w_n_uA2[n_idx, m_idx] * PHertzian[n_idx,0,0]) - (w_n_uA1[n_idx, m_idx] * PHertzian[n_idx, 0, 1])) / ((PHertzian[n_idx, 0, 0] * PHertzian[n_idx, 1, 1]) - (PHertzian[n_idx, 1, 0] * PHertzian[n_idx, 0, 1]))
            T1[m_idx, n_idx] = ((-T2[m_idx, n_idx]) * PHertzian[n_idx, 1, 0] + w_n_uA1[n_idx, m_idx]) / PHertzian[n_idx, 0, 0]

def step2 (nf_data_double, m_max, mmin2max, num_ph, num_th):
    # Perform (4.127)
    temp = np.fft.ifft(nf_data_double, axis=1)

    # Reorganize such that m runs from -m_max to m_max

    w_th_m_mu = np.zeros((num_th, mmin2max, 2), dtype=complex)
    temp = np.fft.fftshift(temp, axes=(1,))
    if num_ph/2.0 == m_max:
        w_th_m_mu[:, :-1, :] = temp
        w_th_m_mu[:, -1, :] = temp[:, 0, :]
    elif num_ph/2.0 > m_max:
        w_th_m_mu[:, :, :] = temp[:, int(num_ph/2-m_max):int(num_ph/2+m_max+1), :]
    
    return w_th_m_mu

def step3 (w_th_m_mu, n_max, nmin2max, mmin2max, num_th):
        # Perform (4.128)
    temp = np.fft.ifft(w_th_m_mu, axis=0)

    # Reorganize such that n runs from -n_max to n_max
    b_l_m_mu = np.zeros((nmin2max, mmin2max, 2), dtype=complex)
    temp = np.fft.fftshift(temp, axes=(0,))
    if num_th/2.0 == n_max:
        b_l_m_mu[:-1, :, :] = temp
        b_l_m_mu[-1, :, :] = temp[0, :, :]
    elif num_th/2.0 > n_max:
        b_l_m_mu[:, :, :] = temp[num_th/2-n_max:num_th/nmin2max, :, :]
    
    b_l_m_mu_wiggle = b_wiggle(b_l_m_mu)
    
    return b_l_m_mu_wiggle

def step4 (pi_wig, b_l_m_mu_wiggle, nmin2max, mmin2max, n_max):
    # Calculate k_mp with fast convolution via FFT methods as explained in [1],(4.89).
    # However, note that [1],(4.89) has a typo. If correct, [1],(4.89) should read:
    #
    #   K(m') = IDFT{ DFT{ PI_wiggle(i) | i = 0,1,...,4N-1 } *
    #                 DFT{ b_j_m_mu_wiggle | j = 0,1,...,4N-1 } }
    #
    k_mp = np.fft.ifft(np.fft.fft(pi_wig, axis=0) * np.fft.fft(b_l_m_mu_wiggle, axis=0), axis=0)

    # Keep only the values of k_mp where -n_max <= m' <= n_max.
    # This is required prior to the evaluation of [1],(4.92)
    temp = np.zeros((nmin2max, mmin2max, 2), dtype=complex)
    temp[0:n_max, :, :] = k_mp[3*n_max:, :, :]
    temp[n_max:, :, :] = k_mp[0:n_max+1, :, :]
    k_mp = temp

    # Pull out k_mp for mu == -1 and mu == +1
    k_mp_m1 = np.reshape(k_mp[:, :, 0], (1, nmin2max, mmin2max))
    k_mp_p1 = np.reshape(k_mp[:, :, 1], (1, nmin2max, mmin2max))

    return (k_mp_m1, k_mp_p1)

def step5 (n_max, m_max, mmin2max, deltas, deltas_mu_m1, deltas_mu_p1, k_mp_m1, k_mp_p1):
    # Initialize the n and m arrays
    n_array = np.reshape(np.linspace(1, n_max, n_max), (n_max, 1))
    m_array = np.reshape(np.linspace(-m_max, m_max, mmin2max), (1, mmin2max))
    # Calculate w_n_m_mu as shown in [1],(4.92). Same as 4.132, if with both polarizations as this is.
    w_n_m_mu = np.zeros((n_max, mmin2max, 2), dtype=complex)
    w_n_m_mu[:, :, 0] = (
        (2.0*n_array+1.0)/2.0 *
        1j**(-1-m_array) *
        np.sum(deltas_mu_m1*deltas*k_mp_m1, axis=1)
    )
    w_n_m_mu[:, :, 1] = (
        (2.0*n_array+1.0)/2.0 *
        1j**(+1-m_array) *
        np.sum(deltas_mu_p1*deltas*k_mp_p1, axis=1)
    )

    return w_n_m_mu

def step6 (n_max, m_max, nmin2max):
        # Initialize a delta pyramid helper function
    def get_mi(m_): return m_ + n_max  # This returns the m index of the deltas

    # Get the deltas for 1 <= n <= n_max and -m_max <= m <= m_max
    deltas = delta_pyramid(n_max)
    deltas = deltas[1:, :, get_mi(-m_max):get_mi(m_max)+1]

    # Get the deltas for only mu == -1 or 1
    deltas_mu_m1 = np.reshape(deltas[:, :, get_mi(-1)], (n_max, nmin2max, 1))
    deltas_mu_p1 = np.reshape(deltas[:, :, get_mi(+1)], (n_max, nmin2max, 1))

    return deltas, deltas_mu_m1, deltas_mu_p1

def step7 (n_max, mmin2max, PHertzian, w_n_m_mu):
        # Initialize the wave coefficient matrix
    q_n_m_s = np.zeros((n_max, mmin2max, 2), dtype='complex')

    # Pull out and reshape the necessary values of the probe response constants
    p_n_neg1_1 = np.reshape(PHertzian[:, 0, 0], (n_max,1))  # mu = -1 , s = 1
    p_n_pos1_1 = np.reshape(PHertzian[:, 1, 0], (n_max,1))  # mu = +1 , s = 1
    p_n_neg1_2 = np.reshape(PHertzian[:, 0, 1], (n_max,1))  # mu = -1 , s = 2
    p_n_pos1_2 = np.reshape(PHertzian[:, 1, 1], (n_max,1))  # mu = +1 , s = 2

    # Solve for q_n_m_s as described in [1],(4.133) and [1],(4.134)
    determinant = p_n_pos1_1*p_n_neg1_2 - p_n_neg1_1*p_n_pos1_2
    q_n_m_s[:, :, 0] = (p_n_neg1_2*w_n_m_mu[:, :, 1] - p_n_pos1_2*w_n_m_mu[:, :, 0])/determinant
    q_n_m_s[:, :, 1] = (p_n_pos1_1*w_n_m_mu[:, :, 0] - p_n_neg1_1*w_n_m_mu[:, :, 1])/determinant

    return q_n_m_s

def spherical_far_field_transform_SNIFT(nf_data, frequency_Hz, meas_dist, transpose_dist):

    if (transpose_dist < meas_dist):
        raise ValueError(f"Inwards/Reverse transform is not implemented")
    
    #This part is based on pysnf by rcutshall
    nf_data_double = np.zeros((2*nf_data.shape[0]-2, nf_data.shape[1], 2), dtype=complex)
    nf_data_double[:,:,0] = singlesphere2doublesphere(nf_data[:,:,0])
    nf_data_double[:,:,1] = singlesphere2doublesphere(nf_data[:,:,1])

    num_th = nf_data_double.shape[0]
    num_ph = nf_data_double.shape[1]

    n_max = int(num_th/2)   # This works because num_th should always be even after
                            # the singlesphere2doublesphere function
    if (n_max > int(num_ph/2)):
        m_max = int(num_ph/2)   # This works because num_ph should always be even after
    else:                       # the singlesphere2doublesphere function
        m_max = n_max           # Since m is not allowed to be higher than n.
    
    mmin2max = int((m_max * 2) + 1)
    nmin2max = int((n_max * 2) + 1)

    w_th_m_mu = step2 (nf_data_double, m_max, mmin2max, num_ph, num_th)

    b_l_m_mu_wiggle = step3 (w_th_m_mu, n_max, nmin2max, mmin2max, num_th)

    # Calculate the pi_wiggle array with [1],(4.84) and [1],(4.86)
    pi_wig = step12 (n_max)
    
    k_mp_m1, k_mp_p1 = step4 (pi_wig, b_l_m_mu_wiggle, nmin2max, mmin2max, n_max)
    
    deltas, deltas_mu_m1, deltas_mu_p1 = step6 (n_max, m_max, nmin2max)

    w_n_m_mu = step5 (n_max, m_max, mmin2max, deltas, deltas_mu_m1, deltas_mu_p1, k_mp_m1, k_mp_p1)

    #Compute the probe constants for input probe step 13
    PHertzian = Phertzian (frequency_Hz=frequency_Hz, n_max=n_max, dist = meas_dist)

    q_n_m_s = step7 (n_max, mmin2max, PHertzian, w_n_m_mu)

    #This function represents steps 8-11
    theta, phi = wavecoeffs2farfield_uniform(q_n_m_s, nf_data.shape[0], nf_data.shape[1], frequency_Hz, transpose_dist)

    ffData = np.zeros(nf_data.shape, dtype=complex)
    ffData[:, :, 0] = theta
    ffData[:, :, 1] = phi

    #Flip the array:
    ffData = np.flip(ffData, 0)

    return ffData

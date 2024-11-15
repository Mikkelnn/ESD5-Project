import numpy as np
import scipy.special as sci_sp
import math

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

    # Extract theta, phi, and electric field components
    E_H_plane = nf_data[:, :, 0].flatten()
    E_E_plane = nf_data[:, :, 1].flatten()

    # abs() calculates the magnitude of a complex number see python ref: https://www.geeksforgeeks.org/finding-magnitude-of-a-complex-number-in-python/
    # calculate the length between the two polarities
    E_tot = (abs(E_H_plane)**2 + abs(E_E_plane)**2)**0.5

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


def spherical_far_field_transform_cook(nf_data, theta_f, phi_f, frequency, nf_meas_dist = 0.2, N = 3, M = 3):

    # validate parameters
    if N < 1:
        raise ValueError(f"N: {N} can't be less than 1")
    
    if M > N:
        raise ValueError(f"M: {M} can't be greater than N: {N}")


    # calculate relevant constants
    c = 3e8 # light speed [m/s]
    λ = c / frequency # wavelength [m]
    β = (2*np.pi) / λ # wavenumber [1/m]

    n_range = range(1, N+1) # N+1 as end is exclusive
    m_range = range(-M, M+1) # M+1 as end is exclusive

    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    def calc_legendre(n, m, cos_θ):
        legendre_inner = math.sqrt((((2 * n) + 1) / (4 * np.pi)) * (math.factorial(n - m) / math.factorial(n + m)))
        return legendre_inner * sci_sp.lpmv(abs(m), n, cos_θ)

    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.20{a,b}  
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: F_{1}; index 1: F_{2}
    f_angular_spherical_harmonics = np.zeros((theta_size, phi_size, N, 2*M+1, 2), dtype=complex)
    F_angular_spherical_harmonics = np.zeros((f_angular_spherical_harmonics.shape), dtype=complex)
    for θ_idx, θ in enumerate(theta_f):
        cos_θ = math.cos(θ)
        sin_θ = math.sin(θ)

        for φ_idx, φ in enumerate(phi_f):
            cos_φ = math.cos(φ)
            sin_φ = math.sin(φ)
            φ_hat = [-sin_φ, cos_φ, 0]
            θ_hat = [cos_φ, sin_φ, -sin_θ] * (1 / math.sqrt(1 + sin_θ**2))

            for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
                for m_idx, m in enumerate(m_range): # this range results in -M is index 0
                    m_term = (-m / abs(m))**m
                    legendre = calc_legendre(n, m, cos_θ)
                    legendre_next_order = calc_legendre(n+1, m, cos_θ)

                    frac_legrende = ((1j * m) / sin_θ) * legendre
                    deriv_legrende = (1 / sin_θ) * (((n - m + 1) * legendre_next_order) - ((n + 1) * cos_θ * legendre))

                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * ((frac_legrende * θ_hat) - (deriv_legrende * φ_hat))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * ((deriv_legrende * θ_hat) + (frac_legrende * φ_hat))
                    
                    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.19
                    ejmφ = math.exp(math.e, 1j * m * φ)
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

    #
    a_spherical_wave_expansion_coefficient = np.zeros((len(n_range), len(m_range), 2), dtype=complex)
    

    # implementation of report eq. 2.17
    for θ_idx, θ in enumerate(theta_f):
        for φ_idx, φ in enumerate(phi_f):
            sum = 0 # eq to the double sum
            for n_idx, n in enumerate(n_range): # this range results in N=1 is index 0
                for m_idx, m in range(m_range): # this range results in -M is index 0
                    a1nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 0] # in report denoted: a_{1nm}
                    a2nm = a_spherical_wave_expansion_coefficient[n_idx, m_idx, 1] # in report denoted: a_{2nm}
                    g1n = g_radial_function[n_idx, 0] # in report denoted: g_{1n}
                    g2n = g_radial_function[n_idx, 1] # in report denoted: g_{2n}
                    F1nm = F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] # in report denoted: F_{1nm}
                    F2nm = F_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] # in report denoted: F_{2nm}
                    sum += (a1nm * g1n * F1nm) + (a2nm * g2n * F2nm)

            E_far[θ_idx, φ_idx] = β * sum
                    

    # Normalize the far-field electric field magnitudes
    E_far_magnitude = np.abs(E_far)
    E_far_magnitude /= np.max(E_far_magnitude)  # Normalize to the maximum value

    return E_far_magnitude


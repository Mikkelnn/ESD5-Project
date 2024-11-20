import numpy as np
import scipy.special as sci_sp
import scipy as sp
import math
import cmath # for complex math

# function definition to compute magnitude of the vector
def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def to_cartesian(radius, theta, phi):
    """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(theta)
    return (x, y, z)

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
    m_range = range(-M, M+1) # M+1 as end is exclusive

    theta_size = len(theta_f)
    phi_size = len(phi_f)
    E_far = np.zeros((theta_size, phi_size), dtype=complex)

    def calc_legendre(n, m, cos_θ):
        legendre_inner = math.sqrt((((2 * n) + 1) / (4 * np.pi)) * (math.factorial(abs(n - m)) / math.factorial(abs(n + m))))
        return legendre_inner * sci_sp.lpmv(abs(m), n, cos_θ)

    # calculate the angular spherical harmonics F1 and F2, using report eq. 2.20{a,b}  
    # for each N value in 1 to N, where N=1 is index: 0; The second axis is for index 0: F_{1}; index 1: F_{2}
    f_angular_spherical_harmonics = np.zeros((theta_size, phi_size, N, 2*M+1, 2, 3), dtype=complex)
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
                for m_idx, m in enumerate(m_range): # this range results in -M is index 0
                    m_term = 1
                    if m != 0:
                        m_term = (-m / abs(m))**m

                    legendre = calc_legendre(n, m, cos_θ)
                    legendre_next_order = calc_legendre(n+1, m, cos_θ)
                    if sin_θ == 0:
                        sin_θ = 0.00001

                    frac_legrende = ((1j * m) / sin_θ) * legendre
                    deriv_legrende = (1.0 / sin_θ) * (((n - m + 1) * legendre_next_order) - ((n + 1) * cos_θ * legendre))

                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 0] = m_term * ((frac_legrende * θ_hat) - (deriv_legrende * φ_hat))
                    f_angular_spherical_harmonics[θ_idx, φ_idx, n_idx, m_idx, 1] = m_term * ((deriv_legrende * θ_hat) + (frac_legrende * φ_hat))
                    
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
        for m_idx, m in enumerate(m_range): # this range results in -M is index 0
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
            for m_idx, m in enumerate(m_range):
                for n in range(abs(m), N+1):
                    if n == 0:
                        continue
                    n_idx = n_range.index(n) # get the corresponding index of m

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
    m_range = range(-M, M+1) # M+1 as end is exclusive

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
                    deriv_legrende = (1.0 / sin_θ) * (((n - m + 1) * legendre_next_order) - ((n + 1) * cos_θ * legendre))

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
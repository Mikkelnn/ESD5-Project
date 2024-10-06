import math

def calculate_field_regions(antenna_size, wavelength):
    """
    Calculate the boundaries for the reactive near-field, Fresnel (radiating near-field), and far-field regions.
    
    Parameters:
    - antenna_size: The largest dimension of the antenna (D) in meters.
    - wavelength: The wavelength of the signal in meters.
    
    Returns:
    - A dictionary with the boundaries of the different regions.
    """
    # Reactive near-field boundary
    reactive_near_field = 0.62 * math.sqrt((antenna_size ** 3) / wavelength)
    
    # Fresnel (radiating near-field) boundary
    fresnel_region_upper = (2 * antenna_size**2) / wavelength
    
    # Far-field boundary (Fraunhofer region)
    far_field_start = fresnel_region_upper
    
    return {
        'reactive_near_field': reactive_near_field,
        'fresnel_region_upper': fresnel_region_upper,
        'far_field_start': far_field_start
    }

# Example usage:
antenna_size = 5  # Largest dimension of the antenna (D) in meters
wavelength = 10    # Wavelength in meters

regions = calculate_field_regions(antenna_size, wavelength)
print(f"Reactive Near Field boundary: {regions['reactive_near_field']:.2f} meters")
print(f"Fresnel Region upper boundary: {regions['fresnel_region_upper']:.2f} meters")
print(f"Far Field starts at: {regions['far_field_start']:.2f} meters")

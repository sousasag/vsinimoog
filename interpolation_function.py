import numpy as np

def interpolate_spectrum(wavelength, flux, delta_wave):
    """
    Interpolate wavelength and flux arrays to create continuous data.

    Parameters:
    wavelength (array-like): Input wavelength array (may have gaps)
    flux (array-like): Input flux array (may have gaps)
    delta_wave (float): Desired delta wavelength for output arrays

    Returns:
    tuple: (new_wavelength, new_flux) - continuous arrays with specified delta wavelength
    """
    # Convert inputs to numpy arrays
    wavelength = np.array(wavelength)
    flux = np.array(flux)

    # Sort arrays by wavelength (in case they're not sorted)
    sorted_indices = np.argsort(wavelength)
    wavelength = wavelength[sorted_indices]
    flux = flux[sorted_indices]

    # Find the range of wavelengths
    min_wave = wavelength[0]
    max_wave = wavelength[-1]

    # Create new wavelength array with specified delta
    new_wavelength = np.arange(min_wave-10, max_wave+10 + delta_wave, delta_wave)

    # Interpolate flux values
    # For points that fall between original data points, we use linear interpolation
    # For points that don't have original data, we set flux = 1
    new_flux = np.ones_like(new_wavelength)

    # Use numpy's interp function for linear interpolation
    # Only interpolate where we have original data points
    valid_indices = (new_wavelength >= wavelength.min()) & (new_wavelength <= wavelength.max())
    new_flux[valid_indices] = np.interp(new_wavelength[valid_indices], wavelength, flux)

    return new_wavelength, new_flux

# Example usage function
def example_usage():
    """Example showing how to use the interpolation function"""
    # Sample data with gaps
    wavelength, flux = np.loadtxt('running_dir/TIC231637303_synth_normalized_spectra.rdb', unpack=True, skiprows=1)  # Replace with actual file path
    delta_wave = 0.01

    new_wavelength, new_flux = interpolate_spectrum(wavelength, flux, delta_wave)

    np.savetxt("test.dat", np.transpose([new_wavelength, new_flux]), newline='\n')

    return new_wavelength, new_flux

if __name__ == "__main__":
    example_usage()
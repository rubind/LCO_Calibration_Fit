import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from FileRead import readcol, writecol
import matplotlib.pyplot as plt


[wavelength_um, n, k] = readcol("Green-2008.txt", 'fff')

wavelength_cm = wavelength_um * 1e-4

alpha_cm_inv = 4.0 * np.pi * k / wavelength_cm

silicon_wavelength_nm = wavelength_um*1000.

# silicon_table should contain wavelength in nm and alpha in cm^-1.
alpha_interp = interp1d(
    silicon_wavelength_nm,
    alpha_cm_inv,
    kind="linear",
    bounds_error=False,
    fill_value=0.0,
)


def qe_model(params, wavelength_nm):
    log_d_um, logit_A, slope_per_nm = params

    d_cm = np.exp(log_d_um) * 1e-4
    A = 1.0 / (1.0 + np.exp(-logit_A))

    alpha = np.clip(alpha_interp(wavelength_nm), 0.0, None)

    absorption_probability = -np.expm1(-alpha * d_cm)
    smooth_factor = np.exp(slope_per_nm * (wavelength_nm - 1000.0))

    return A * smooth_factor * absorption_probability


def residuals(params, wavelength_nm, qe):
    #plt.plot(qe_model(params, wavelength_nm) - qe)
    #plt.show()
    #plt.close()
    
    return (qe_model(params, wavelength_nm) - qe)/10.


[wavelength_A, qe] = readcol("QE_UV.txt", 'ff')
wavelength_nm = wavelength_A/10.

qe /= 100.


fit_region = (
    (wavelength_nm >= 900.0)
    & (wavelength_nm <= 1050.0)
    & np.isfinite(qe)
)

initial_params = [
    np.log(40.0),        # effective thickness, microns
    2.0,                 # A ~ 0.88
    0.0,                 # weak smooth slope
]

result = least_squares(
    residuals,
    initial_params,
    args=(
        wavelength_nm[fit_region],
        qe[fit_region]
    ),
)

wavelength_extrap = np.arange(1070.0, 1401.0, 10.)
qe_extrap = qe_model(result.x, wavelength_extrap)


wavelength_A = np.concatenate((wavelength_A, wavelength_extrap*10.))
qe = np.concatenate((qe, qe_extrap))

writecol("QE.txt", [wavelength_A, qe*100.])

plt.plot(wavelength_A, qe)
plt.savefig("QE_extrap.pdf")
plt.close()


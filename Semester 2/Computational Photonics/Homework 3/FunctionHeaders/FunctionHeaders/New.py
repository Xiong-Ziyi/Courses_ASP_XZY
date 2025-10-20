import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm
#import bluered_dark
import time

'''
plt.rcParams.update({
    'figure.figsize': (12 / 2.54, 9 / 2.54),
    'figure.subplot.bottom': 0.15,
    'figure.subplot.left': 0.165,
    'figure.subplot.right': 0.90,
    'figure.subplot.top': 0.9,
    'axes.grid': False,
    'image.cmap': 'bluered_dark',
})
'''
# save figures to disk
save_figures = False
# save movie to disk (requires Ffmpeg)
save_movie = False

plt.close('all')

# constants
c = 2.99792458e8  # speed of light [m/s]
mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
epsilon_0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0 / epsilon_0)  # vacuum impedance [Ohm]

# geometry parameters
x_width = 18e-6  # X_x_width of computatinal domain [m]
n1 = 1  # refractive index in front of interface
n2 = 2  # refractive index behind interface
interf_pos = x_width / 4  # postion of dielectric interface

# simulation parameters
dx = 15e-9 # grid spacing [m]
dt = 60e-15  # duration of simulation [s]

Nx = int(round(x_width / dx)) + 1  # number of grid points

# source parameters
source_frequency = 500e12  # [Hz]
source_position = 0  # [m]
source_pulse_length = 1e-15  # [s]

# timer to measure the execution time


def fdtd_1d(eps_rel, dx, dt, source_frequency, source_position,
            source_pulse_length):
    """Computes the temporal evolution of a pulsed excitation using the
    1D FDTD method. The temporal center of the pulse is placed at a
    simulation time of 3*source_pulse_length. The origin x=0 is in the
    center of the computational domain. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel : 1d-array
            Rel. permittivity distribution within the computational domain.
        dx : float
            Spacing of the simulation grid (please ensure dx <= lambda/20).
        time_span : float
            Time span of simulation.
        source_frequency : float
            Frequency of current source.
        source_position : float
            Spatial position of current source.
        source_pulse_length :
            Temporal width of Gaussian envelope of the source.

    Returns
    -------
        Ez : 2d-array
            Z-component of E(x,t) (each row corresponds to one time step)
        Hy : 2d-array
            Y-component of H(x,t) (each row corresponds to one time step)
        x  : 1d-array
            Spatial coordinates of the field output
        t  : 1d-array
            Time of the field output
    """


    # calculate time step and prefactors
    del_t = dx / (2 * c)
    e_factor =del_t / epsilon_0
    h_factor =del_t / mu0

    # create position and time vectors
    Nx = eps_rel.size
    x = np.arange(Nx) * dx - (Nx - 1) / 2.0 * dx
    Numb_iter = int(round(dt /del_t))
    t = np.arange(Numb_iter + 1) * del_t

    # allocate field arrays
    Ez = np.zeros((Numb_iter + 1, Nx), dtype=complex)
    Hy = np.zeros((Numb_iter + 1, Nx - 1), dtype=complex)

    # source properties
    # angular frequency (avoids multiplication by 2*pi every iteration)
    source_angular_frequency = 2 * np.pi * source_frequency
    # time offset of pulse center
    t0 = 3 * source_pulse_length
    # x-grid index of delta-source (rounded to nearest grid point)
    source_ind = int(round((source_position - x[0]) / dx))
    if (source_ind < 1) or (source_ind > Nx - 2):
        raise ValueError('Source position out of range')

    for n in range(0, Numb_iter):
        # calculate E at time n + 1, the values at the spatial indices
        # 0 and Nx -1 are determined by the PEC boundary conditions
        # and do not have to be updated
        Ez[n + 1, 1:-1] = (Ez[n, 1:-1]
                + e_factor / dx * (Hy[n, 1:] - Hy[n, :-1]) / eps_rel[1:-1])

        # add source term to Ez
        # source current has to  be taken at n + 1/2
        t_source = (n + 0.5) * del_t - t0
        j_source = (np.exp(-1j * source_angular_frequency * t_source)  # carrier
                 * np.exp(-(t_source / source_pulse_length) ** 2))  # envelope
        Ez[n + 1, source_ind] -= e_factor / eps_rel[source_ind] * j_source

        # calculate H at time n + 3/2
        Hy[n + 1, :] = Hy[n, :] + h_factor / dx * (Ez[n + 1, 1:] - Ez[n + 1, :-1])

    # The fields are returned on the same x-grid as eps_rel whereby
    # both Ez and Hy are returned at the same points in space and time
    # (the user should not need to care about the peculiarities of
    # the Yee grid and the leap frog algorithm).

    # interpolate Hy to same t and x grid as Ez
    Hy[1:, :] = 0.5 * (Hy[:-1, :] + Hy[1:, :])
    Hy = avg_axes(repl_bound_val(Hy, [1]), [1])
    return Ez, Hy, x, t

def avg_axes(field, axes):
    """Averages neighboring values of the given field along the specified axes

    Arguments
    ---------
        field : nd-array
            Field array to be averaged.
        axes : sequence
            Sequence of axes that shall be averaged.

    Returns
    -------
        res: nd-array
            Field averaged along the specified axes.
    """

    # prefactor for calculating avergaes with the same numerical preciosion
    # as the input field.
    f = np.array(0.5, dtype=field.dtype)

    c = slice(0, None)  # full
    l = slice(0, -1)  # left part of average
    r = slice(1, None)  # right part of average
    res = field
    for ax in axes:
        a = tuple((l if i == ax else c for i in range(res.ndim)))
        b = tuple((r if i == ax else c for i in range(res.ndim)))
        res = f * (res[a] + res[b])
    return res




def repl_bound_val(field, axes):
    """Replicates the roundary values of the given field along the
    specified axes

    Arguments
    ---------
        field : nd-array
            Field array to be padded.
        axes : sequence
            Sequence of axes that shall be padded.

    Returns
    -------
        res: nd-array
            Field padded along the specified axes.
    """
    res = field
    for ax in axes:
        a = tuple((0 if i == ax else slice(None)
                   for i in range(field.ndim)))
        b = tuple((-1 if i == ax else slice(None)
                   for i in range(field.ndim)))
        na = tuple((np.newaxis if i == ax else slice(None)
                    for i in range(field.ndim)))
        res = np.concatenate((res[a][na], res, res[b][na]), axis=ax)
    return res

class Timer(object):
    """Tic-toc timer.
    """

    def __init__(self):
        """Initializer.
        Stores the current time.
        """
        self._tic = time.time()

    def tic(self):
        """Stores the current time.
        """
        self._tic = time.time()

    def toc(self):
        """Returns the time in seconds that has elapsed since the last call
        to tic().
        """
        return time.time() - self._tic

# timer to measure the execution time
timer = Timer()
# simulate homogeneous medium
eps_rel = np.ones((Nx,)) * n1 ** 2
Ez_ref, Hy_ref, x, t = fdtd_1d(eps_rel, dx, dt,
                               source_frequency, source_position,
                               source_pulse_length)

# simulate dielectric interface
eps_rel[x >= interf_pos] = n2 ** 2
timer.tic()
Ez, Hy, x, _ = fdtd_1d(eps_rel, dx, dt,
                       source_frequency, source_position,
                       source_pulse_length)
print('time: {0:g}s'.format(timer.toc()))

#  plot time traces of electric field and Poynting vector
extent = [t[0] * c * 1e6, t[-1] * c * 1e6, x[-1] * 1e6, x[0] * 1e6]

# plot electric field
vmax = max(np.max(np.abs(Ez_ref)), np.max(np.abs(Ez))) * 1e6
fig = plt.figure()
plt.imshow(Ez_ref.real.T * 1e6, extent=extent, vmin=-vmax, vmax=vmax)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('real part of $E_z$ [µV/m]')
if save_figures:
    fig.savefig('Ez_ref_1D.pdf', dpi=300)

fig = plt.figure()
plt.imshow(Ez.real.T * 1e6, extent=extent, vmin=-vmax, vmax=vmax)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('real part of $E_z$ [µV/m]')
if save_figures:
    fig.savefig('Ez_1D.pdf', dpi=300)

# calculate Poynting vector
Sx_ref = -0.5 * np.real(Ez_ref * np.conj(Hy_ref))
Sx = -0.5 * np.real(Ez * np.conj(Hy))

# plot power flow
vmax = np.abs(Sx_ref).max()
fig = plt.figure()
plt.imshow(Sx_ref.T / vmax, extent=extent, vmin=-1, vmax=1)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('normalize Poynting vector')
cb.set_ticks(np.linspace(-1, 1, 11))
if save_figures:
    fig.savefig('Sx_ref_1D.pdf', dpi=300)

fig = plt.figure()
plt.imshow(Sx.T / vmax, extent=extent, vmin=-1, vmax=1)
cb = plt.colorbar()
plt.xlabel('$ct$ [µm]')
plt.ylabel('$x$ [µm]')
cb.set_label('normalize Poynting vector')
cb.set_ticks(np.linspace(-1, 1, 11))
plt.show()

if save_figures:
    fig.savefig('Sx_1D.pdf', dpi=300)



from scipy.interpolate import interp1d

def convergence_test(fdtd_1d_func, initial_dx, initial_dt, n1, n2, source_frequency, source_position, source_pulse_length, factor=0.5, num_tests=5):
    # Store initial dx and dt
    dx = initial_dx
    dt = initial_dt

    # Initialize list to hold errors
    dx_list = []
    dt_list = []
    errors = []

    # Initial simulation with initial parameters
    Nx = int(round(x_width / dx)) + 1
    eps_rel = np.ones((Nx,)) * n1 ** 2
    eps_rel[int(round((interf_pos / dx))):] = n2 ** 2
    _, Hy_prev, x_prev, _ = fdtd_1d_func(eps_rel, dx, dt, source_frequency, source_position, source_pulse_length)

    for _ in range(num_tests):
        # Update dx and dt
        dx *= factor
        dt *= factor
        Nx = int(round(x_width / dx)) + 1
        eps_rel = np.ones((Nx,)) * n1 ** 2
        eps_rel[int(round((interf_pos / dx))):] = n2 ** 2

        # Run simulation
        _, Hy, x, _ = fdtd_1d_func(eps_rel, dx, dt, source_frequency, source_position, source_pulse_length)

        # Interpolate Hy_prev to the new x grid
        interpolate = interp1d(x_prev, Hy_prev[-1], kind='cubic', fill_value="extrapolate")
        Hy_prev_interpolated = interpolate(x)

        # Calculate error (L2 norm of the difference)
        error = np.linalg.norm(Hy[-1] - Hy_prev_interpolated) / np.linalg.norm(Hy_prev_interpolated)

        # Store results
        dx_list.append(dx)
        dt_list.append(dt)
        errors.append(error)

        # Update previous solution
        Hy_prev = Hy
        x_prev = x

    return dx_list, dt_list, errors


# Define initial parameters for the convergence test
initial_dx = 15e-9  # initial grid spacing [m]
initial_dt = 60e-15  # initial duration of simulation [s]

# Run the convergence test
dx_list, dt_list, errors = convergence_test(fdtd_1d, initial_dx, initial_dt, n1, n2, source_frequency, source_position, source_pulse_length)

# Plot results
plt.figure()
plt.loglog(dx_list, errors)
plt.xlabel('dx (m)')
plt.ylabel('Error')
plt.title('Convergence with respect to dx')
plt.grid(True)
plt.show()


plt.figure()
plt.loglog(dt_list, errors)
plt.xlabel('dt (s)')
plt.ylabel('Error')
plt.title('Convergence with respect to dt')
plt.grid(True)
plt.show()
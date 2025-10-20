import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.interpolate import interpn
def average_axes(field, axes):


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
def fdtd_3d_interpolate_field(field, z_ind, component):
    component = component.lower()
    if component == 'ex':
        rep_axes = [0]
        pad_axes = [1, 2]
    elif component == 'ey':
        rep_axes = [1]
        pad_axes = [0, 2]
    elif component == 'ez':
        rep_axes = [2]
        pad_axes = [0, 1]
    elif component == 'hx':
        rep_axes = [1, 2]
        pad_axes = [0]
    elif component == 'hy':
        rep_axes = [0, 2]
        pad_axes = [1]
    elif component == 'hz':
        rep_axes = [0, 1]
        pad_axes = [2]
    else:
        raise ValueError('Invalid field component')
    res = average_axes(
        replicate_boundary_values(
            boundary_values(field, pad_axes), rep_axes),
        rep_axes)
    return res[:, :, z_ind]


def replicate_boundary_values(field, axes):

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


def boundary_values(field, axes):

    res = field
    for ax in axes:
        shape = tuple((1 if i == ax else n for i, n in enumerate(res.shape)))
        pad = np.zeros(shape, dtype=res.dtype)
        res = np.concatenate((pad, res, pad), axis=ax)
    return res

def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz,field_component, z_ind, output_step):
    c = 2.99792458e8
    mu0 = 4 * np.pi * 1e-7
    eps0 = 1 / (mu0 * c ** 2)
    #the eps_rel should be the grid size for 3d fdtd implementation
    Nx, Ny, Nz = eps_rel.shape
    #preset the factor used in defintion
    #time space  dr is the gird spacing here
    dt=np.float32(dr/(2*c))
    dr=np.float32(dr)
    #factor eps and mu
    E=np.float32(dt/eps0)
    M=np.float32(dt/mu0)
    #get the iteration number for the for the time
    Niter = int(round(time_span /dt/ output_step) * output_step)
    #arrange the actual time for each iteration time
    t = np.arange(0, Niter + 1, output_step) * dt
        #use the inversion of permittivity in fdtd method
    eps_rel=np.float32(1)/eps_rel
    #interpolating method
    #we need to use the interpolated as mention in the seminar in each direction
    iepsx = average_axes(eps_rel, [0])
    iepsy = average_axes(eps_rel, [1])
    iepsz = average_axes(eps_rel, [2])
    #after getting the interpolated value,deleting the original eps_rel
    del eps_rel
    #use the same method to get the interpolated value of sources
    jx = average_axes(jx, [0])
    jy = average_axes(jy, [1])
    jz = average_axes(jz, [2])

    # preset the E component in each direction according to the array sizes in the seminar
    Ex = np.zeros((Nx - 1, Ny, Nz), dtype=np.complex64)
    Ey = np.zeros((Nx, Ny - 1, Nz), dtype=np.complex64)
    Ez = np.zeros((Nx, Ny, Nz - 1), dtype=np.complex64)
    # preset the E component in each direction according to the array sizes in the seminar
    Hx = np.zeros((Nx, Ny - 1, Nz - 1), dtype=np.complex64)
    Hy = np.zeros((Nx - 1, Ny, Nz - 1), dtype=np.complex64)
    Hz = np.zeros((Nx - 1, Ny - 1, Nz), dtype=np.complex64)
    #get the slice from 1 to N-2
    Ax = slice(1, Nx - 1)
    Ay = slice(1, Ny - 1)
    Az = slice(1, Nz - 1)
    #get the slice from 0 to N-3
    Bx = slice(0, Nx - 2)
    By = slice(0, Ny - 2)
    Bz = slice(0, Nz - 2)
    #get the slice from 0 to N-2
    Cx = slice(0, Nx - 1)
    Cy = slice(0, Ny - 1)
    Cz = slice(0, Nz - 1)
    #get the slice from 1 to N-1
    Dx = slice(1, Nx)
    Dy = slice(1, Ny)
    Dz = slice(1, Nz)
    #get the slice for specific time spep in the x and y direction
    St=np.zeros((t.size,Nx,Ny),dtype=np.complex64)
    #prefactor for calculation of averages
    f = np.float32(0.5)
    F = np.zeros((t.size, Nx, Ny), dtype=np.complex64)
    next_out = 1
    #next slice index in output St
    Nextis=1
 #for time iteration ,use the for loop to do the iteration
    for n in range(Niter):
#firstly,the source need to be define
        #the time dependent part of the source,3 is initial time position
        jt=(n+0.5)*dt-3*tau
        #source term according to the formula in the seminar
        js=np.complex64(E*np.exp(-2j*np.pi*freq*jt)*np.exp(-((n+0.5)*dt-3*tau/tau)**2))
#update the Ex field here
        U =E/ dr * (Hz[Cx, Ay,Az] - Hz[Cx,By,Az])
        U -=E/ dr * (Hy[Cx, Ay,Az] - Hy[Cx, Ay, Bz])
        U -= js * jx[Cx, Ay,Az]
         # divide by eps_rel (interpolated to Ex grid)
        U *= iepsx[Cx, Ay,Az]
         # + Ex(t=n*dt)
        U += Ex[Cx, Ay,Az]
        if ((n + 1) % output_step == 0) and (field_component == 'ex'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                           field_component)
             next_out += 1
        Ex[Cx,Ay,Az] = U
 #update the Ey field here
        U =E/ dr * (Hx[Ax, Cy,Az] - Hx[Ax,Cy,Bz])
        U -=E/ dr * (Hz[Ax,Cy,Az] - Hz[Bx,Cy,Az])
        U -= js * jy[Ax,Cy,Az]
        U *= iepsy[Ax,Cy,Az]
        U += Ey[Ax, Cy,Az]
        if ((n + 1) % output_step == 0) and (field_component == 'ey'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                           field_component)
             next_out += 1
        Ey[Ax,Cy,Az] = U
#update the Ez field here
        U =E/ dr * (Hy[Ax, Ay, Cz] - Hy[Bx, Ay,Cz])
        U -=E/ dr * (Hx[Ax, Ay,Cz] - Hx[Ax, By,Cz])
        U -=js * jz[Ax,Ay,Cz]
        U *= iepsz[Ax, Ay,Cz]
        U += Ez[Ax, Ay,Cz]
        if ((n + 1) % output_step == 0) and (field_component == 'ez'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(U, z_ind,
                                                           field_component)
             next_out += 1
        Ez[Ax, Ay,Cz] = U
#update the Hx field here
        U = Hx[Ax, Cy,Cz]
        U -=M/ dr * (Ez[Ax, Dy,Cz] - Ez[Ax,Cy,Cz])
        U += M / dr * (Ey[Ax,Cy, Dz] - Ey[Ax,Cy,Cz])
        if ((n + 1) % output_step == 0) and (field_component == 'hx'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(
                 f * (U + Hx[Ax, Cy,Cz]), z_ind,
                 field_component)
             next_out += 1
        Hx[Ax,Cy,Cz] = U
#update the Hy field here
        U = Hy[Cx,Ay,Cz]
        U -= M/ dr * (Ex[Cx, Ay,Dz] - Ex[Cx,Ay,Cz])
        U += M/ dr * (Ez[Dx,Ay,Cz] - Ez[Cx, Ay,Cz])
        if ((n + 1) % output_step == 0) and (field_component == 'hy'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(
                 f * (U + Hy[Cx,Ay,Cz]), z_ind,
                 field_component)
             next_out += 1
        Hy[Cx,Ay,Cz] = U
#update the Hz field here
        U = Hz[Cx,Cy,Az]
        U -=M / dr * (Ey[Dx,Cy,Az] - Ey[Cx, Cy,Az])
        U +=M/ dr * (Ex[Cy,Dy,Az] - Ex[Cx,Cy,Az])
        if ((n + 1) % output_step == 0) and (field_component == 'hz'):
             F[next_out, :, :] = fdtd_3d_interpolate_field(
                 f * (U + Hz[Cx, Cy,Az]), z_ind,
                 field_component)
             next_out += 1
        Hz[Cx, Cy,Az] = U

        progress = (n + 1) / Niter


    return F, t


#We use the test parameter here,firstly construct the plot tools
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
save_figures = False
# save movie to disk (requires Ffmpeg)
save_movie = False
plt.close('all')
'''
#Setting basic parameter
c = 2.99792458e8  # speed of light [m/s]
mu0 = 4 * np.pi * 1e-7  # vacuum permeability [Vs/(Am)]
eps0 = 1 / (mu0 * c ** 2)  # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0 / eps0)  # vacuum impedance [Ohm]

# grid parameter,time span and dr
Nx = 199  # nuber of grid points in x-direction
Ny = 201  # nuber of grid points in y-direction
Nz = 5  # nuber of grid points in z-direction
dr = 30e-9  # grid spacing in [m]
time_span = 10e-15  # duration of simulation [s]

# source parameters from  seminar
freq = 500e12  # pulse [Hz]
tau = 1e-15  # pulse width [s]
source_width = 2  # width of Gaussian current dist. [grid points]

#get the x,y and z number for different index,x and y from -N-1/2 to N-1/2
x = np.arange(-int(np.ceil((Nx - 1) / 2)), int(np.floor((Nx - 1) / 2)) + 1) * dr
y = np.arange(-int(np.ceil((Ny - 1) / 2)), int(np.floor((Ny - 1) / 2)) + 1) * dr
#assign the epsilon of 1 to the space
eps_rel = np.ones((Nx, Ny, Nz), dtype=np.float32)

midx = int(np.ceil((Nx - 1) / 2))
midy = int(np.ceil((Ny - 1) / 2))
midz = int(np.ceil((Nz - 1) / 2))
#source term we have in different direction
jx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
jy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
jz = np.tile(np.exp(-((np.arange(Nx)[:, np.newaxis, np.newaxis]- midx) / source_width) ** 2)* np.exp(-((np.arange(Ny)[np.newaxis, :, np.newaxis]- midy) / source_width) ** 2),(1, 1, Nz))
# output parameters
z_ind = midz
output_step = 4

#use 3dfdtd function to simulate it
#get the hx distribution for starting hx component
field_component = 'hx'
Hx, t = fdtd_3d(eps_rel, dr, time_span, freq, tau,jx, jy, jz, field_component, z_ind, output_step)
#get the hy distribution for starting hy component
field_component = 'hy'
Hy, _ = fdtd_3d(eps_rel, dr, time_span, freq, tau,jx, jy, jz, field_component, z_ind, output_step)
#get the ez distribution for starting ez component
field_component = 'ez'
Ez, _ = fdtd_3d(eps_rel, dr, time_span, freq, tau,jx, jy, jz, field_component, z_ind, output_step)


#plot the field at the end of the simulation
fig = plt.figure()
# plot electric field
#Here, Ez is likely a 2D array where Ez[i, j] represents the Ez component of the electric field at position (x[j], y[i]). Ez[-1, :] selects the last row of Ez, and .T transposes it to make it suitable for plotting.
res = Ez[-1, :].T
#color_range calculates a range for coloring the plot based on the maximum absolute value of res, scaled by 1e6.
#pix calculates half of the pixel width based on the spacing of x values.
#extent defines the spatial extent of the plot in micrometers (xmin, xmax, ymin, ymax).
color_range = np.max(np.abs(res)) * 1e6
pix = 0.5 * (x[1] - x[0])
extent = ((x[0] - pix) * 1e6, (x[-1] + pix) * 1e6,(y[-1] + pix) * 1e6, (y[0] - pix) * 1e6)
#plt.imshow() plots the 2D array res.real * 1e6, where res.real extracts the real part of res and * 1e6 scales it to micrometers.
#vmin and vmax set the minimum and maximum values for the color scale.
plt.imshow(res.real * 1e6, vmin=-color_range, vmax=color_range,extent=extent)
cb = plt.colorbar()
plt.gca().invert_yaxis()
#R: Defines the radial positions for the polar grid. It is calculated based on the grid spacing dr and spans from a fraction of the maximum x value to the maximum x value, stepping by 2.
#p: Defines the angular positions for the polar grid, covering 360 degrees (or 2π radians) in increments of 5 degrees.
R = dr * np.arange(int(round(max(x) * 3 / 8 / dr)) - 1,int(round(max(x) / dr)) + 1, 2)
P = np.deg2rad(np.arange(72) * 5)
#xr and yr: Convert polar coordinates (R, phi) into Cartesian coordinates (xr, yr). These arrays represent the x and y coordinates on the polar grid.
xr = R[:, np.newaxis] * np.cos(P[np.newaxis, :])
yr = R[:, np.newaxis] * np.sin(P[np.newaxis, :])
#X and Y: Meshgrid of original Cartesian coordinates x and y. These grids represent the spatial positions where the magnetic field components
interp_points = np.hstack((xr.reshape((-1, 1)), yr.reshape((-1, 1))))
#U and V are reshaped to match the shape of xr using reshape(xr.shape). This step organizes the interpolated values back into the shape that matches the polar grid defined earlier
U = interpn((x, y), Hx[-1, :, :].real, interp_points)
U = U.reshape(xr.shape)
V = interpn((x, y), Hy[-1, :, :].real, interp_points)
V = V.reshape(xr.shape)
#Normalization is crucial for ensuring that vector fields are represented consistently in plots or computations.
norm = np.sqrt(U ** 2 + V ** 2).max()
U /= norm
V /= norm
# plot magnetic field as vectors
plt.quiver(xr * 1e6, yr * 1e6, U, V, scale=1e-6 / dr, angles='xy', label='magnetic field')
l = plt.legend(frameon=False, loc='lower left')
for text in l.get_texts():
    text.set_color('w')
plt.title('$ct$ = {0:1.2f}µm'.format(t[-1] * c * 1e6))
plt.xlabel('x coordinate [µm]')
plt.ylabel('y coordinate [µm]')
cb.set_label('$\\Re\\{E_z\\}$ [µV/m]')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def calculate_error(reference, result):
    return np.linalg.norm(reference - result) / np.linalg.norm(reference)

def fdtd_convergence_test(eps_rel, dr_list, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step):
    c = 2.99792458e8  # speed of light [m/s]
    results = {}
    errors = []

    for dr in dr_list:
        dt = dr / (2 * c)  # Time step corresponding to the spatial step
        Nx, Ny, Nz = eps_rel.shape
        time_span = 10e-15  # duration of simulation [s]

        field, _ = fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)
        results[dr] = field[-1, :, :]  # Store the last field distribution

    # Use the finest grid as the reference solution
    reference = results[dr_list[-1]]

    for dr in dr_list:
        errors.append(calculate_error(reference, results[dr]))

    return errors, dr_list

def plot_error(errors, dr_list):
    plt.figure(figsize=(8, 6))
    plt.loglog(dr_list, errors, marker='o')
    plt.xlabel('Spatial step size (dr) [m]')
    plt.ylabel('Relative error')
    plt.title('Convergence of FDTD Solution')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Define the range of spatial step sizes (in meters)
dr_list = [40e-9, 35e-9, 30e-9, 25e-9, 20e-9, 15e-9, 10e-9, 5e-9]

# Run the convergence test
errors, dr_list = fdtd_convergence_test(eps_rel, dr_list, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)

# Plot the error versus dr
plot_error(errors, dr_list)


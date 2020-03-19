import numpy as np
from devito import *
from examples.seismic import RickerSource, Receiver, TimeAxis
from sympy import sqrt

# Define a physical size
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
vp = np.empty(shape, dtype=np.float32)
density= np.empty(shape, dtype=np.float32)
vp[:, :51] = 1.5
vp[:, 51:] = 2.5

# Define density with Gardner equation
alpha=0.31
beta=0.25
density[:, :51] = 1.
density[:, 51:] =alpha*(vp[:, 51:]**beta)

g = Grid (shape =shape, origin =origin, extent =((shape[0]-1)*spacing[0], (shape[1]-1)*spacing[1]))
u = TimeFunction (name =" u", grid =g, space_order =2, time_order =2) # Wavefield
m = Function (name =" m", grid =g) # Physical parameter
rho = Function (name =" rho", grid =g, space_order =2) # Physical parameter
#inv_rho=Function ( name =" inv_rho", grid =g, space_order =2) # Physical parameter

m.data[:]=1/(vp*vp)
#inv_rho.data[:]=1/np.sqrt(density)
rho.data[:]=density
#rho.data[:]=np.sqrt(density)

#pde = rho*inv_rho.laplace
pde=sqrt(rho)*(1/sqrt(rho)).laplace
stencil = Eq(rho, pde)
Operator([stencil])()

pde=m*u.dt2 - u.laplace + rho*u
stencil = Eq(u.forward, solve(pde, u.forward))

t0 = 0.  # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
dt = 1.68  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=g, f0=f0, npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(g.extent) * .5
src.coordinates.data[0, -1] = 20.  # Depth is 20m

rec = Receiver(name='rec', grid=g, npoint=101, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0, g.extent[0], num=101)
rec.coordinates.data[:, 1] = 20.  # Depth is 20m

src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
rec_term = rec.interpolate(expr=u.forward)

op = Operator([stencil]+ src_term + rec_term)
op(time=time_range.num-1,dt=dt)

from devito import norm, clear_cache
import numpy as np
from examples.seismic import Model, TimeAxis, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver

# Define a physical size
shape = (101, 101)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,space_order=4, nbpml=10)

t0 = 0.     # Simulation starts a t=0
tn = 1000.  # Simulation last 1 second (1000 ms)
nreceivers=shape[0]

# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.  # Depth is 20m

# Define acquisition geometry: receivers
# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] = 20.
# Geometry
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=.010, src_type='Ricker')

# Execute foward and adjoint runs
clear_cache()
solver = AcousticWaveSolver(model, geometry, space_order=4)
rec , _, _ = solver.forward(vp=model.vp)
srca_d , _, _ = solver.adjoint(rec=rec,vp=model.vp)

# Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
term1 = np.dot(srca_d.data.reshape(-1), solver.geometry.src.data)
#term2= np.linalg.norm(rec.data)**2
term2 = norm(rec) ** 2

print('<Ax,y>: %f, <x, A^Ty>: %f, difference: %4.4e, ratio: %f'% (term1, term2, (term1 - term2)/term1, term1 / term2))

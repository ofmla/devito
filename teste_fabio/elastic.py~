from devito import *
from devito import Function, clear_cache, TimeFunction
from examples.seismic import Model,ModelElastic, plot_velocity, AcquisitionGeometry, plot_shotrecord, Receiver
from examples.seismic.elastic import ElasticWaveSolver
from examples.seismic.acoustic import AcousticWaveSolver
import numpy as np
#import osfile = '../../data/velocity'# define the first offset
offmin = 200
# the number of receivers
nreceivers = 250
# the distance between two receivers
d = 25
# the maximun of offset
offmax = offmin + (nreceivers - 1) * d# using devito to do forward modeling
from devito import clear_cache
clear_cache()
shape = (1295, 611)  # Number of grid point (nx, nz)
spacing = (6.25, 7.5)  # Grid spacing in m. 
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
t0 = 0.
tn = 4000.
f0 = 0.010# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] =  0
src_coordinates[0, -1] = 20.  # Depth is 20m# Define acquisition geometry: receivers# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(offmin, offmax, num=nreceivers)
rec_coordinates[:, 1] = 20.


#vp = np.load(file)
#vs = (vp-1.36)/1.16
#rho = 1.74*vp**0.25

vp_top = 1.5
vp_bottom = 3.5

# Define a velocity profile in km/s
vp = np.empty(shape, dtype=np.float32)
vp[:] = vp_top  # Top velocity (background)
vp_i = np.linspace(vp_top, vp_bottom, 3)
for i in range(1, 3):
    vp[..., i*int(shape[-1] / 3):] = vp_i[i]  # Bottom velocity

vs = 0.5 * vp[:]
rho = 0.31 * (1e3*vp)**0.25
rho[vp < 1.51] = 1.0
vs[vp < 1.51] = 0.0

model = ModelElastic(vp=vp, vs=vs, rho=rho, origin=origin, shape=shape, spacing=spacing,
		  space_order=4, nbpml=100)
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
# trace
solver = ElasticWaveSolver(model, geometry, space_order=4)
true_d, rec2, _,_,_,_,_,_ = solver.forward(vp=model.vp, vs = model.vs, rho=model.rho)
trace = rec2.resample(dt=4)

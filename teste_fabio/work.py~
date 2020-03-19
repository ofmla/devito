import numpy as np
from examples.seismic import Receiver
from examples.seismic import RickerSource
from examples.seismic import Model, plot_velocity, TimeAxis
from devito import TimeFunction
from devito import Eq, solve
from devito import Operator


# Set velocity model
nx = 201
nz = 201
nb = 20
shape = (nx, nz)
spacing = (5., 5.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :int(nx/2)] = 2.0
v[:, int(nx/2):] = 2.5

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbpml=nb)

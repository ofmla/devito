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

# Set time range, source, source coordinates and receiver coordinates
t0 = 0.  # Simulation starts a t=0
tn = 1000.  # Simulation lasts tn milliseconds
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt = time_range.num  # number of time steps

f0 = 0.020  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(
    name='src',
    grid=model.grid,
    f0=f0,
    time_range=time_range)  

src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 200.  # Depth is 20m

rec = Receiver(
    name='rec',
    grid=model.grid,
    npoint=101,
    time_range=time_range)  # new
rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.  # Depth is 20m
depth = rec.coordinates.data[:, 1]  # Depth is 20m


plot_velocity(model, source=src.coordinates.data,
              receiver=rec.coordinates.data[::4, :])

#Used for reshaping
vnx = nx+20 
vnz = nz+20

# Set symbolics for the wavefield object `u`, setting save on all time steps 
# (which can occupy a lot of memory), to later collect snapshots (naive method):

#####

def mirror(field, model, dim_fs):
    """
    Free surface expression. Mirrors the negative wavefield above the sea level
    :return: Symbolic equation of the free surface
    """
    dim = field.dimensions[-1]
    next = field.forward
    return [Eq(next.subs({dim: dim_fs}), - next.subs({dim: 2*model.nbpml - dim_fs}))]

# Free surface
from devito import SubDimension
x, z = model.grid.dimensions
dim_fs = SubDimension.left(name='abc_'+ z.name + '_left', parent=z, thickness=model.nbpml)





from devito import ConditionalDimension

nsnaps = 100               # desired number of equally spaced snaps
factor = round(nt / nsnaps)  # subsequent calculated factor

print(f"factor is {factor}")

#Part 1 #############
time_subsampled = ConditionalDimension(
    't_sub', parent=model.grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=2,
                     save=(nt + factor - 1) // factor, time_dim=time_subsampled)
print(time_subsampled)
#####################

u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
# Mirror
fs = mirror(u, model, dim_fs)
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
stencil = Eq(u.forward, solve(pde, u.forward))
src_term = src.inject(
    field=u.forward,
    expr=src * dt**2 / model.m)
    #offset=model.nbpml)
rec_term = rec.interpolate(expr=u)#, offset=model.nbpml)

#Part 2 #############
#op1 = Operator([stencil] + src_term + rec_term,
#               subs=model.spacing_map)  # usual operator
#op2 = Operator([stencil] + src_term + [Eq(usave, u)] + rec_term,
#               subs=model.spacing_map)  # operator with snapshots
op2 = Operator([stencil] + src_term + fs + [Eq(usave, u)] +rec_term, subs=model.spacing_map)
print(op2)
#op1(time=nt - 2, dt=model.critical_dt)  # run only for comparison
op2(time=nt - 2, dt=model.critical_dt)
g= open('shot.file', 'wb')
#first shot
print(rec.data.shape)
np.transpose(rec.data).astype('float32').tofile(g)
#####################

#Part 3 #############
print("Saving snaps file")
print("Dimensions: nz = {:d}, nx = {:d}".format(nz + 2 * nb, nx + 2 * nb))
filename = "snaps2.bin"
usave.data.tofile(filename)

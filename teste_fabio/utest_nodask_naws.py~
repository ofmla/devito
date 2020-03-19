import numpy
import sys, json
import finetocoarse
import gc

from examples.seismic import (Model, plot_velocity, AcquisitionGeometry,
                              plot_shotrecord, Receiver, RickerSource,
                              TimeAxis)
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import PointSource, Receiver
from examples.seismic.acoustic.operators import laplacian, iso_stencil

from devito import Eq, solve, Operator, TimeFunction
from devito import clear_cache
import time
import pprint
from memory_profiler import profile
import logging
import tracemalloc
#f=open('hi.txt','w+')
#@profile(stream=f)

def trace_leak(start,prev):
 current = tracemalloc.take_snapshot()
 stats=current.compare_to(start,'filename')
 prev_stats=current.compare_to(prev,'lineno')

 print("[ Top Diffs since Start ]")
 for i,stat in enumerate(stats[:10],1):
       print('top_diffs','i:',i,stat) 
       #logging.info('top_diffs',i=i,stat=str(stat))
 print("[ Top Incremental ]")
 for i,stat in enumerate(prev_stats[:10],1):
       print('top_incremental','i:',i,stat) 
      #logging.info('top_incremental',i=i,stat=str(stat))
 print("[ Top Current]")
 for i,stat in enumerate(current.statistics('filename')[:10],1):
       print('top_current','i:',i,stat) 
      #logging.info('top_current',i=i,stat=str(stat))
 traces=current.statistics('traceback')
 for stat in traces[:1]:
      print('traceback','memory_blocks:',stat.count,'size_kB:',stat.size/1024) 
      #logging.info('traceback',memory_blocks=stat.count, size_kB=stat.size/1024)
      for line in stat.traceback.format():
           print(line)
           #logging.info(line)
 prev=current

@profile
def my_task(part,op,param,dobs,src,src_coor,rec):

    error=0.
    model=get_true_model(part*(1/1000.),param)
    op.arguments()
    tstart = time.time()
    # Geometry 
    # Create symbols for forward wavefield, source and receivers


    #source_locations= numpy.empty((1, 2))
    #geometry = AcquisitionGeometry(model_part, rec_coor, source_locations,
    #                               param['t0'], param['tn'], src_type='Ricker',
    #                               f0=param['f0'])
    #geometry.resample(param['dt'])
    #print(geometry.nt) 

    # Set up solver.
    #solver = AcousticWaveSolver(model_part, geometry, space_order=param['space_order'])
    #print("{:>30}: {:>8}".format('solver',sizeof_fmt(sys.getsizeof(solver))))


    #tracemalloc.start()
    #start=tracemalloc.take_snapshot()
    #prev=start
    for i in  range(len(dobs[0][0])):
    #for i in  range(5):
      #dcalc=generate_shotdata_i(numpy.array([src_coor[i]]), geometry, solver, model_part)
      src.coordinates.data[0, :]=numpy.array([src_coor[i]])
      #op(time=time_range.num-1, dt=param['dt'])
      op(damp=model.damp, vp=model.vp, time=time_range.num-1, dt=model.critical_dt, rec=rec, src=src)
      dcalc=numpy.array(rec.data)
      res=get_value(dcalc,dobs[:,:,i])
      u.data.fill(0)
      error += res
      clear_cache()
      #trace_leak(start,prev)
    #print("{:>30}: {:>8}".format('dcalc',sizeof_fmt(sys.getsizeof(dcalc))))
    #print("{:>30}: {:>8}".format('res',sizeof_fmt(sys.getsizeof(res))))
    #print("{:>30}: {:>8}".format('error',sizeof_fmt(sys.getsizeof(error))))


    #dcalc=None
    #model=None
    #del dcalc,model

    print("Forward modeling took {}".format(time.time() - tstart))
    print(error)
    return (error,)

def get_value(dcalc, dobs):
    return  0.5 * numpy.sum((dcalc - dobs)**2.) 

def get_true_model(v,param):
    return Model(vp=v,origin=param['origin'], shape=param['shape'],
                 spacing=param['spacing'], space_order=param['space_order'], nbpml=param['nbpml'])

def generate_shotdata_i(src_coordinates,geometry, solver, true_model):

    clear_cache()
    geometry.src_positions[0, :] = src_coordinates[:]
    # Generate synthetic receiver data from true model.
    true_d, _, _ = solver.forward(m=true_model.m)
    return numpy.array(true_d.data)

def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# Get parameters. 
js=open('parameters.json') 
par=json.load(js)
numpy.random.seed(0)
# Load the data
shots = numpy.fromfile('shots.file', dtype=numpy.float32)
shots = numpy.reshape(shots, (6559, 369, 50))

# Set up source/receiver data and geometry.
src_coordinates = numpy.empty((par['nshots'], len(par['shape'])))
max_offset= (par['nshots']-1)*par['int_btw_shots']

for i in  range(par['nshots']):
  if par['nshots'] > 1: 
    src_coordinates[i,:] = [(i*max_offset/(par['nshots']-1))+par['first_src_xcoor'],par['src_depth']]
  else:
    src_coordinates[i,:] = [par['first_src_xcoor'],par['src_depth']]

#rec_coordinates = numpy.empty((par['nreceivers'], len(par['shape'])))


# Compute the x, z coordinates of coarse grid points and lower and upper bounds of search space
x,z,lb,ub=finetocoarse.create_coarse_grid_coord(par['shape'][0],par['shape'][1],par['spacing'][0],
         par['spacing'][1],par['cgrid']['vstart'],par['cgrid']['vend'],par['f0']*2000.,par['cgrid']['scale'], par['cgrid']['water_samples'])

pop=[]
pop.append(((ub-lb) * numpy.random.random_sample(len(x))) + lb)

interpolated_model=finetocoarse.coarse2fine(par['shape'][0],par['shape'][1],
                                                par['spacing'][0],par['spacing'][1],pop[0],x,z)
interpolated_model[:,0:par['cgrid']['water_samples']]=1500.

######
time_range = TimeAxis(start=par['t0'], stop=par['tn'], step=par['dt'])
model_part=get_true_model(interpolated_model*(1/1000.),par)
src = RickerSource(name='src', grid=model_part.grid, f0=par['f0'],
                   npoint=1, time_range=time_range)
rec = Receiver(name='rec', grid=model_part.grid, npoint=par['nreceivers'], time_range=time_range)
rec.coordinates.data[:, 0] = numpy.linspace(0., (par['shape'][0]-1)*(par['spacing'][0]), num=par['nreceivers'])
rec.coordinates.data[:, 1] = par['rec_depth'] # at surface
u = TimeFunction(name="u", grid=model_part.grid, time_order=2, space_order=par['space_order'])
m, damp = model_part.m, model_part.damp
s = model_part.grid.stepping_dim.spacing
eqn = iso_stencil(u, m, s, damp, kernel='OT2')
src_term = src.inject(field=u.forward, expr=src * s**2 / m)
rec_term = rec.interpolate(expr=u)
op=Operator(eqn + src_term + rec_term, subs=model_part.spacing_map,name='Forward')
#######

tstart = time.time()
tracemalloc.start()
start=tracemalloc.take_snapshot()
prev=start
# Using for loop 
for i in range(5):
 #print(my_task(pop[i],par,shots,x,z,src_coordinates,rec_coordinates))
 print(my_task(interpolated_model,op,par,shots,src,src_coordinates,rec))
 trace_leak(start,prev)
 #print("[ tracemalloc stats ]")
 #for stat in snapshot.statistics('lineno')[:20]:
 # print(stat) 
 #mem_usage =memory_usage((my_task, (interpolated_model,par,shots,src_coordinates,rec_coordinates)))
 #print(mem_usage)
print("Fitness computation took {}".format(time.time() - tstart))

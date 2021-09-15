"""
Dedalus script for full sphere anelastic convection,
applied to the Marti Boussinesq benchmark.

Usage:
    boussinesq_hydro.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 3e-4]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 2.85e-2]
    --Prandtl=<Prandtl>                  Prandtl number  [default: 1]

    --Ntheta=<Ntheta>                    Latitudinal modes [default: 32]
    --Nr=<Nr>                            Radial modes [default: 32]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.1]
    --safety=<safety>                    CFL safety factor [default: 0.4]
    --fixed_dt                           Fix dt

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_rotation=<run_time_rot>   How long to run, in rotation timescale; overrides run_time_diffusion if set
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --dt_output=<dt_output>              Time between outputs, in rotation times (P_rot = 4pi) [default: 2]
    --scalar_dt_output=<dt_scalar_out>   Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 2]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --debug                              Produce debugging output for NCCs
"""
import numpy as np
from mpi4py import MPI
import time

import pathlib
import os
import sys
import h5py

from dedalus.tools.parallel import Sync
from docopt import docopt
args = docopt(__doc__)

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += '_Ek{}_Co{}_Pr{}'.format(args['--Ekman'],args['--ConvectiveRossbySq'],args['--Prandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--benchmark']:
    data_dir += '_benchmark'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
logger.info("saving data in {}".format(data_dir))
from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
print(os.path.join(data_dir,'logs/dedalus_log'))


import dedalus.public as de
from dedalus.core import timesteppers, operators
from dedalus.extras import flow_tools

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

Nθ = int(args['--Ntheta'])
Nr = int(args['--Nr'])
Nφ = Nθ*2

ncc_cutoff = float(args['--ncc_cutoff'])

radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
logger.info("Ek = {}, Co2 = {}, Pr = {}".format(Ek,Co2,Pr))

L_dealias = 3/2
N_dealias = 3/2

start_time = time.time()
c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor((c,), mesh=mesh)
b = de.BallBasis(c, (Nφ,Nθ,Nr), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=np.float64)
b_S2 = b.S2_basis()
phi1, theta1, r1 = b.local_grids((1,1,1))
phi, theta, r = b.local_grids((L_dealias,L_dealias,N_dealias))
phig,thetag,rg= b.global_grids((L_dealias,L_dealias,N_dealias))

u = de.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
p = de.Field(dist=d, bases=(b,), dtype=np.float64)
s = de.Field(dist=d, bases=(b,), dtype=np.float64)
τ_u = de.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.float64)
τ_s = de.Field(dist=d, bases=(b_S2,), dtype=np.float64)

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
curl = lambda A: de.Curl(A)
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
ddt = lambda A: de.TimeDerivative(A)
trans = lambda A: de.TransposeComponents(A)
radial = lambda A: de.RadialComponent(A)
angular = lambda A: de.AngularComponent(A, index=1)
trace = lambda A: de.Trace(A)
power = lambda A, B: de.Power(A, B)
LiftTau = lambda A, n: de.LiftTau(A,b,n)
integ = lambda A: de.Integrate(A, c)
sqrt = lambda A: operators.UnaryGridFunction(np.sqrt, A)

# NCCs and variables of the problem
ez = de.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
ez.set_scales(b.dealias)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.Grid(ez).evaluate()

T = de.Field(dist=d, bases=(b.radial_basis,), dtype=np.float64)

T['g'] = 0.5*(1-r1**2)

r_vec = de.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=np.float64)
r_vec.set_scales(b.dealias)
r_vec['g'][2] = r
#r_vec_g = de.operators.Grid(r_vec).evaluate()

# Entropy source function, inspired from MESA model
source = de.Field(dist=d, bases=(b,), dtype=np.float64)
source['g'] = 3

e = grad(u) + trans(grad(u))
e.store_last = True

problem = de.IVP([p, u,  τ_u, s,  τ_s])
problem.add_equation((div(u), 0))
problem.add_equation((ddt(u) + grad(p)  - Ek*lap(u) - Co2*r_vec*s + LiftTau(τ_u,-1),
                      - dot(u, e) - cross(ez_g, u)))
problem.add_equation((ddt(s) - Ek/Pr*(lap(s)) + LiftTau(τ_s,-1),
                     - dot(u, grad(s)) + Ek/Pr*source ))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((p(r=radius), 0), condition = "ntheta == 0")
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((s(r=radius), 0))
logger.info("Problem built")

s.require_scales(L_dealias)
s['g'] = 0.5*(1-r**2) # static solution

if args['--benchmark']:
    amp = 1e-1
    𝓁 = int(args['--ell_benchmark'])
    norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
    s.require_scales(L_dealias)
    s['g'] += amp*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))
else:
    amp = 1e-5
    rng = np.random.default_rng(seed=42+rank)
    noise = de.Field(name='noise', dist=d, bases=(b,), dtype=np.float64)
    noise['g'] = 2*rng.random(noise['g'].shape)-1 # -1--1 uniform distribution
    noise.require_scales(0.25)
    noise['g']
    noise.require_scales(1)
    s.require_scales(1)
    s['g'] += amp*noise['g']

# Solver
solver = problem.build_solver(timesteppers.SBDF2, ncc_cutoff=ncc_cutoff)

def vol_avg(q):
    Q = integ(q/(4/3*np.pi)).evaluate()['g']
    if rank == 0:
        return Q[0][0][0]
    else:
        return 0

int_test = de.Field(dist=d, bases=(b,), dtype=np.float64)
int_test['g']=1
int_test.require_scales(L_dealias)
logger.info("vol_avg(1)={}".format(vol_avg(int_test)))

KE = 0.5*dot(u,u)
ens = dot(curl(u),curl(u))
Ts = T*s
Lz = dot(cross(r_vec,u), ez)

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=10, max_writes=np.inf)
traces.add_task(integ(KE)/(4/3*np.pi), name='KE')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(sqrt(integ(ens))/(4/3*np.pi), name='Ro')
traces.add_task(sqrt(integ(2*KE))/Ek/(4/3*np.pi), name='Re')
traces.add_task(Co2*integ(Ts)/(4/3*np.pi), name='PE')
traces.add_task(integ(Lz)/(4/3*np.pi), name='Lz')


report_cadence = 100
energy_report_cadence = 100
max_dt = float(args['--max_dt'])
if args['--fixed_dt']:
    dt = max_dt
else:
    dt = max_dt/10
cfl_safety_factor = float(args['--safety'])
timestepper_history = [0,1]
hermitian_cadence = 100
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)

main_start = time.time()
good_solution = True
while solver.ok and good_solution:
    if not args['--fixed_dt']:
        dt = CFL.compute_timestep()

    if solver.iteration % energy_report_cadence == 0:
        KE_avg = vol_avg(KE.evaluate())
        E0 = KE_avg/Ek**2*(4/3*np.pi) # volume integral, not average

        Ro = np.sqrt(vol_avg(ens.evaluate()))

        Re = np.sqrt(2*vol_avg(KE.evaluate()))/Ek

        PE = Co2*vol_avg(Ts.evaluate())

        Lz_avg = vol_avg(Lz.evaluate())

        log_string = "iter: {:d}, dt={:.2e}, t={:.3e} ({:.3e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", KE={:.2e} ({:.4e}), PE={:.2e}, Lz={:.2e}".format(KE_avg, E0, PE, Lz_avg)
        logger.info(log_string)
        good_solution = np.isfinite(E0)
    if solver.iteration % hermitian_cadence in timestepper_history:
        for field in solver.state:
            field['g']
    solver.step(dt)

end_time = time.time()

startup_time = main_start - start_time
main_loop_time = end_time - main_start
DOF = nm*(Lmax+1)*(Nmax+1)
niter = solver.iteration
if rank==0:
    print('performance metrics:')
    print('    startup time   : {:}'.format(startup_time))
    print('    main loop time : {:}'.format(main_loop_time))
    print('    main loop iter : {:d}'.format(niter))
    print('    wall time/iter : {:f}'.format(main_loop_time/niter))
    print('          iter/sec : {:f}'.format(niter/main_loop_time))
    print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))

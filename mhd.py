"""
Dedalus script for full sphere boussinesq mhd convection.

Usage:
    mhd.py [options]

Options:
    --Ekman=<Ekman>                      Ekman number    [default: 2.5e-5]
    --ConvectiveRossbySq=<Co2>           Squared Convective Rossby = Ra*Ek**2/Pr [default: 0.168]
    --Prandtl=<Pr>                       Prandtl number  [default: 1]
    --MagneticPrandtl=<Pm>               Magnetic Prandtl number [default: 1]

    --Ntheta=<Ntheta>                    Latitudinal modes [default: 128]
    --Nr=<Nr>                            Radial modes [default: 128]
    --mesh=<mesh>                        Processor mesh for 3-D runs; if not set a sensible guess will be made

    --benchmark                          Use benchmark initial conditions
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --thermal_eq                         Start with thermally equilibrated (unstable) ICs
    --scale_eq=<scale_eq>                Scalie unstable profile by fixed amount [default: 0.1]

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.1]
    --safety=<safety>                    CFL safety factor [default: 0.4]
    --fixed_dt                           Fix timestep size

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
from dedalus.tools.parallel import Sync
from mpi4py import MPI
import time

import pathlib
import os
import sys
import h5py

from docopt import docopt
args = docopt(__doc__)

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
# suppress azimuthal mode warnings for float64 optimal distribution
dlog = logging.getLogger('basis')
dlog.setLevel(logging.ERROR)

data_dir = sys.argv[0].split('.py')[0]
data_dir += '_Ek{}_Co{}_Pr{}_Pm{}'.format(args['--Ekman'],args['--ConvectiveRossbySq'],args['--Prandtl'],args['--MagneticPrandtl'])
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

import dedalus.public as de
from dedalus.extras import flow_tools
from dedalus.core.basis import SphericalAzimuthalAverage, SphericalAverage

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
Pm = MagneticPrandtl = float(args['--MagneticPrandtl'])

logger.info("Ek = {}, Co2 = {}, Pr = {}, Pm = {}".format(Ek,Co2,Pr,Pm))

dealias = L_dealias = N_dealias = 3/2

start_time = time.time()
c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor((c,), mesh=mesh, dtype=np.float64)
b = de.BallBasis(c, (Nφ,Nθ,Nr), radius=radius, dealias=(L_dealias,L_dealias,N_dealias), dtype=np.float64)
b_S2 = b.S2_basis()
phi1, theta1, r1 = b.local_grids((1,1,1))
phi, theta, r = b.local_grids((L_dealias,L_dealias,N_dealias))
phig,thetag,rg= b.global_grids((L_dealias,L_dealias,N_dealias))

u = de.Field(name="u", dist=d, bases=(b,), tensorsig=(c,))
p = de.Field(name="p", dist=d, bases=(b,))
s = de.Field(name="s", dist=d, bases=(b,))
A = de.Field(name="A", dist=d, bases=(b,), tensorsig=(c,))
φ = de.Field(name="φ", dist=d, bases=(b,))
τ_u = de.Field(name="τ_u", dist=d, bases=(b_S2,), tensorsig=(c,))
τ_s = de.Field(name="τ_s", dist=d, bases=(b_S2,))
τ_A = de.Field(name="τ_A", dist=d, bases=(b_S2,), tensorsig=(c,))

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
Lift = lambda A, n: de.LiftTau(A,b,n)
integ = lambda A: de.Integrate(A, c)
azavg = lambda A: de.Average(A, c.coords[0])
shellavg = lambda A: de.Average(A, c.S2coordsys)
avg = lambda A: de.Integrate(A, c)/(4/3*np.pi*radius**3)

ell_func = lambda ell: ell+1
ellp1 = lambda A: de.SphericalEllProduct(A, c, ell_func)

# NCCs and variables of the problem
ez = de.Field(dist=d, bases=(b,), tensorsig=(c,), name='ez')
ez.set_scales(b.dealias)

ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
ez_g = de.Grid(ez).evaluate()

r_vec = de.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), name='r_vec')
r_vec.set_scales(b.dealias)
r_vec['g'][2] = r
#r_vec_g = de.Grid(r_vec).evaluate()

# Entropy source function, inspired from MESA model
source = de.Field(dist=d, bases=(b,), name='source')
source['g'] = 3

e = grad(u) + trans(grad(u))
e.store_last = True

B = curl(A)
J = curl(B)

problem = de.IVP([p, u,  τ_u, s,  τ_s, φ, A, τ_A])
problem.add_equation((div(u), 0))
problem.add_equation((ddt(u) + grad(p) - Ek*lap(u) - Co2*r_vec*s + Lift(τ_u,-1),
                      - cross(curl(u) + ez, u) + cross(J,B) ))
problem.add_equation((ddt(s) - Ek/Pr*lap(s) + Lift(τ_s,-1),
                      - dot(u, grad(s)) + Ek/Pr*source ))
problem.add_equation((div(A), 0)) # coulomb gauge
problem.add_equation((ddt(A) + grad(φ) - Ek/Pm*lap(A) + Lift(τ_A,-1),
                        cross(u, B) ))
# Boundary conditions
problem.add_equation((radial(u(r=radius)), 0), condition = "ntheta != 0")
problem.add_equation((p(r=radius), 0), condition = "ntheta == 0")
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((s(r=radius), 0))
problem.add_equation((radial(grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0), condition = "ntheta != 0")
problem.add_equation((φ(r=radius), 0), condition = "ntheta == 0")

logger.info("Problem built")

# Solver
solver = problem.build_solver(de.SBDF2, ncc_cutoff=ncc_cutoff)

# ICs
s.require_scales(L_dealias)
if args['--thermal_eq']:
    s['g'] = float(args['--scale_eq'])*0.5*(1-r**2) # static solution
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
    noise = de.Field(name='noise', dist=d, bases=(b,))
    noise['g'] = 2*rng.random(noise['g'].shape)-1 # -1--1 uniform distribution
    noise.require_scales(0.25)
    noise['g']
    noise.require_scales(1)
    s.require_scales(1)
    s['g'] += amp*noise['g']


mag_amp = 1e-4
invert_B_to_A = False
if invert_B_to_A:
    B_IC = de.Field(name="B_IC", dist=d, bases=(b,), tensorsig=(c,))
    B_IC.require_scales(dealias)
    B_IC['g'][2] = 0 # radial
    B_IC['g'][1] = -mag_amp*3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
    B_IC['g'][0] = -mag_amp*3./4.*r*(-1+r**2)*np.cos(theta)* \
                                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                                 +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))
    logger.info("set initial conditions for B")
    IC_problem = de.LBVP([φ, A, τ_A])
    IC_problem.add_equation((div(A), 0))
    IC_problem.add_equation((curl(A) + grad(φ) + Lift(τ_A, -1), B_IC))
    IC_problem.add_equation((radial(grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0), condition = "ntheta != 0")
    IC_problem.add_equation((φ(r=radius), 0), condition = "ntheta == 0")
    IC_solver = IC_problem.build_solver()
    IC_solver.solve()
    logger.info("solved for initial conditions for A")
else:
    # Marti convective dynamo benchmark values
    A_analytic_2 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi))
                   +3/8*r**3*(2-7*r**2+9*r**4-4*r**6)
                       *(3*np.cos(theta)**2-1)
                   +9/160*r**2*(-200/21*r+980/27*r**3-540/11*r**5+880/39*r**7)
                         *(3*np.cos(theta)**2-1)
                   +9/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                        *(3*np.cos(theta)**2-1)
                   +1/8*r*(-48/5*r+288/7*r**3-64*r**5+360/11*r**7)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi))
                   +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                       *np.sin(theta)*(np.sin(phi)-np.cos(phi)))
    A_analytic_1 = (-27/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                            *np.cos(theta)*np.sin(theta)
                    +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                        *np.cos(theta)*(np.sin(phi)-np.cos(phi)))
    A_analytic_0 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                       *(np.cos(phi)+np.sin(phi)))

    A.require_scales(dealias)
    A['g'][0] = mag_amp*A_analytic_0
    A['g'][1] = mag_amp*A_analytic_1
    A['g'][2] = mag_amp*A_analytic_2


# Outputs
max_dt = float(args['--max_dt'])
if args['--fixed_dt']:
    dt = max_dt
else:
    dt = max_dt/10
if not args['--restart']:
    mode = 'overwrite'
else:
    write, dt = solver.load_state(args['--restart'])
    mode = 'append'

KE = 0.5*dot(u,u)
PE = Co2*s
Lz = dot(cross(r_vec,u), ez)
ME = 0.5*dot(B,B)
ME.store_last = True
enstrophy = dot(curl(u),curl(u))
enstrophy.store_last = True

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=10, max_writes=np.inf, virtual_file=True, mode=mode)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(ME), name='ME')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(avg(KE)*2)/Ek, name='Re')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(Lz), name='Lz')

# Analysis
eφ = de.Field(dist=d, bases=(b,), tensorsig=(c,))
eφ['g'][0] = 1
er = de.Field(dist=d, bases=(b,), tensorsig=(c,))
er['g'][2] = 1
Bφ = dot(B, eφ)
Aφ = dot(A, eφ)
ρ_cyl = de.Field(dist=d, bases=(b,))
ρ_cyl.require_scales(dealias)
ρ_cyl['g'] = r*np.sin(theta)
Ωz = dot(u, eφ)/ρ_cyl # this is not ω_z; misses gradient terms; this is angular differential rotation.
slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt = 10, max_writes = 10, virtual_file=True, mode=mode)
slices.add_task(s(theta=np.pi/2), name='s')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
slices.add_task(azavg(Bφ), name='<Bφ>')
slices.add_task(azavg(Aφ), name='<Aφ>')
slices.add_task(azavg(Ωz), name='<Ωz>')
slices.add_task(azavg(s), name='<s>')
slices.add_task(shellavg(s), name='s(r)')
slices.add_task(dot(B,er)(r=radius), name='Br') # is this sufficient?  Should we be using radial(B) instead?

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(KE*2)/Ek, name='Re')
flow.add_property(np.sqrt(enstrophy), name='Ro')
flow.add_property(KE, name='KE')
flow.add_property(ME, name='ME')
flow.add_property(PE, name='PE')
flow.add_property(Lz, name='Lz')
flow.add_property(np.sqrt(dot(τ_u,τ_u)), name='|τ_u|')
flow.add_property(np.abs(τ_s), name='|τ_s|')
flow.add_property(np.sqrt(dot(τ_A,τ_A)), name='|τ_A|')

cfl_safety_factor = float(args['--safety'])
timestepper_history = [0,1]
hermitian_cadence = 100
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)
CFL.add_velocity(B)

if args['--run_time_rotation']:
    solver.stop_sim_time = float(args['--run_time_rotation'])
else:
    solver.stop_sim_time = float(args['--run_time_diffusion'])/Ek

logger.info("avg div A: {}".format(avg(div(A)).evaluate()['g']))

main_start = time.time()
good_solution = True
while solver.proceed and good_solution:
    if not args['--fixed_dt']:
        dt = CFL.compute_timestep()
    if solver.iteration % report_cadence == 0 and solver.iteration > 0:
        KE_avg = flow.grid_average('KE') # volume average needs a defined volume
        E0 = KE_avg/Ek**2
        Re_avg = flow.grid_average('Re')
        Ro_avg = flow.grid_average('Ro')
        ME_avg = flow.grid_average('ME')
        PE_avg = flow.grid_average('PE')
        Lz_avg = flow.grid_average('Lz')
        τ_u_m = flow.max('|τ_u|')
        τ_s_m = flow.max('|τ_s|')
        τ_A_m = flow.max('|τ_A|')
        log_string = "iter: {:d}, dt={:.1e}, t={:.3e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", KE={:.2e}, ME={:.2e}, PE={:.2e}".format(KE_avg, ME_avg, PE_avg)
        log_string += ", Re={:.1e}, Ro={:.1e}".format(Re_avg, Ro_avg)
        log_string += ", Lz={:.1e}, τ=({:.1e},{:.1e},{:.1e})".format(Lz_avg, τ_u_m, τ_s_m, τ_A_m)
        logger.info(log_string)

        good_solution = np.isfinite(E0)

    if solver.iteration % hermitian_cadence in timestepper_history:
        for field in solver.state:
            field.require_grid_space()
    solver.step(dt)

end_time = time.time()

startup_time = main_start - start_time
main_loop_time = end_time - main_start
DOF = Nφ*Nθ*Nr
niter = solver.iteration
if rank==0:
    print('performance metrics:')
    print('    startup time   : {:}'.format(startup_time))
    print('    main loop time : {:}'.format(main_loop_time))
    print('    main loop iter : {:d}'.format(niter))
    print('    wall time/iter : {:f}'.format(main_loop_time/niter))
    print('          iter/sec : {:f}'.format(niter/main_loop_time))
    print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))

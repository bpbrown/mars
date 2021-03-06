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

    --thermal_equilibrium                Start with thermally equilibrated (unstable) ICs
    --scale_eq=<scale_eq>                Scalie unstable profile by fixed amount [default: 1]

    --max_dt=<max_dt>                    Largest possible timestep [default: 0.1]
    --safety=<safety>                    CFL safety factor [default: 0.4]
    --fixed_dt                           Fix timestep size

    --run_time_diffusion=<run_time_d>    How long to run, in diffusion times [default: 20]
    --run_time_rotation=<run_time_rot>   How long to run, in rotation timescale; overrides run_time_diffusion if set
    --run_time_iter=<run_time_i>         How long to run, in iterations

    --slice_dt=<slice_dt>                Time between outputs, in rotation times (P_rot = 4pi) [default: 10]
    --scalar_dt=<scalar_dt>              Time between scalar outputs, in rotation times (P_rot = 4pi) [default: 10]

    --restart=<restart>                  Merged chechpoint file to restart from.
                                         Make sure "--label" is set to avoid overwriting the previous run.

    --label=<label>                      Additional label for run output directory

    --ncc_cutoff=<ncc_cutoff>            Amplitude to truncate NCC terms [default: 1e-10]
    --debug                              Produce debugging output for NCCs
"""
import numpy as np
from dedalus.tools.parallel import Sync
from mpi4py import MPI

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
data_dir += '_Co{}_Ek{}_Pr{}_Pm{}'.format(args['--ConvectiveRossbySq'],args['--Ekman'],args['--Prandtl'],args['--MagneticPrandtl'])
data_dir += '_Th{}_R{}'.format(args['--Ntheta'], args['--Nr'])
if args['--thermal_equilibrium']:
    data_dir += '_therm'
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

N?? = int(args['--Ntheta'])
Nr = int(args['--Nr'])
N?? = N??*2

ncc_cutoff = float(args['--ncc_cutoff'])

radius = 1

Ek = Ekman = float(args['--Ekman'])
Co2 = ConvectiveRossbySq = float(args['--ConvectiveRossbySq'])
Pr = Prandtl = float(args['--Prandtl'])
Pm = MagneticPrandtl = float(args['--MagneticPrandtl'])

logger.debug(sys.argv)
logger.debug('-'*40)
logger.info("saving data in {}".format(data_dir))
logger.info("Run parameters")
logger.info("Ek = {}, Co2 = {}, Pr = {}, Pm = {}".format(Ek,Co2,Pr,Pm))

dealias = 3/2

c = de.SphericalCoordinates('phi', 'theta', 'r')
d = de.Distributor(c, mesh=mesh, dtype=np.float64)
b = de.BallBasis(c, (N??,N??,Nr), radius=radius, dealias=dealias, dtype=np.float64)
phi, theta, r = b.local_grids()

u = d.VectorField(c, name="u", bases=b)
p = d.Field(name="p", bases=b)
s = d.Field(name="s", bases=b)
A = d.VectorField(c, name="A", bases=b)
?? = d.Field(name="??", bases=b)
??_p = d.Field(name="??_p")
??_?? = d.Field(name="??_??")
??_s = d.Field(name="??_s", bases=b.S2_basis())
??_u = d.VectorField(c, name="??_u", bases=b.S2_basis())
??_A = d.VectorField(c, name="??_A", bases=b.S2_basis())

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
bk2 = b.clone_with(k=2)
lift = lambda A, n: de.LiftTau(A,bk2,n)
integ = lambda A: de.Integrate(A, c)
azavg = lambda A: de.Average(A, c.coords[0])
shellavg = lambda A: de.Average(A, c.S2coordsys)
avg = lambda A: de.Integrate(A, c)/(4/3*np.pi*radius**3)

ell_func = lambda ell: ell+1
ellp1 = lambda A: de.SphericalEllProduct(A, c, ell_func)

# NCCs and variables of the problem
bk = b.clone_with(k=1) # ez on k+1 level to match curl(u)
ez = d.VectorField(c, name='ez', bases=bk)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec = d.VectorField(c, name='r_vec', bases=bk2.radial_basis)
r_vec['g'][2] = r

r_S2 = d.VectorField(c, name='r_hat_S2')
r_S2['g'][2] = 1


# Entropy source function, inspired from MESA model
source_func = d.Field(name='S', bases=b)
source_func['g'] =  Ek/Pr*3
source = de.Grid(source_func).evaluate()
source.name = 'source'

e = grad(u) + trans(grad(u))
e.store_last = True

B = curl(A)
J = -lap(A) #curl(B)

problem = de.IVP([p, u, s, ??, A, ??_p, ??_u, ??_s, ??_??, ??_A])
problem.add_equation((div(u) + ??_p, 0))
problem.add_equation((ddt(u) + grad(p) - Ek*lap(u) - Co2*r_vec*s + lift(??_u,-1),
                      - cross(curl(u) + ez, u) + cross(J,B) ))
problem.add_equation((ddt(s) - Ek/Pr*lap(s) + lift(??_s,-1),
                      - dot(u, grad(s)) + source ))
problem.add_equation((div(A) + ??_??, 0)) # coulomb gauge
problem.add_equation((ddt(A) + grad(??) - Ek/Pm*lap(A) + lift(??_A,-1),
                        cross(u, B) ))
# Boundary conditions
problem.add_equation((integ(p), 0))
problem.add_equation((radial(u(r=radius)), 0))
problem.add_equation((radial(angular(e(r=radius))), 0))
problem.add_equation((s(r=radius), 0))
problem.add_equation((integ(??), 0))
problem.add_equation((dot(r_S2, grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))

logger.info("Problem built")

# Solver
solver = problem.build_solver(de.SBDF2, ncc_cutoff=ncc_cutoff)

# ICs
if args['--thermal_equilibrium']:
    s['g'] = float(args['--scale_eq'])*0.5*(1-r**2) # static solution
if args['--benchmark']:
    amp = 1e-1
    ???? = int(args['--ell_benchmark'])
    norm = 1/(2**????*np.math.factorial(????))*np.sqrt(np.math.factorial(2*????+1)/(4*np.pi))
    s['g'] += amp*norm*r**????*(1-r**2)*(np.cos(????*phi)+np.sin(????*phi))*np.sin(theta)**????
    logger.info("benchmark run with perturbations at ell={} with norm={}".format(????, norm))
else:
    amp = 1e-5
    noise = d.Field(name='noise', bases=b)
    noise.fill_random('g', seed=42, distribution='standard_normal')
    noise.low_pass_filter(scales=0.25)
    s['g'] += amp*noise['g']


mag_amp = 1e-4
invert_B_to_A = False
if invert_B_to_A:
    B_IC = d.VectorField(c, name="B_IC", bases=b)
    B_IC['g'][2] = 0 # radial
    B_IC['g'][1] = -mag_amp*3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
    B_IC['g'][0] = -mag_amp*3./4.*r*(-1+r**2)*np.cos(theta)* \
                                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                                 +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))
    logger.info("set initial conditions for B")
    IC_problem = de.LBVP([??, A, ??_??, ??_A])
    IC_problem.add_equation((div(A) + ??_??, 0))
    IC_problem.add_equation((curl(A) + grad(??) + lift(??_A, -1), B_IC))
    IC_problem.add_equation((integ(??), 0))
    IC_problem.add_equation((dot(r_S2, grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0))
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

scalar_dt = float(args['--scalar_dt'])
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=scalar_dt, max_writes=np.inf, virtual_file=True, mode=mode)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(ME), name='ME')
traces.add_task(integ(KE)/Ek**2, name='E0')
traces.add_task(np.sqrt(avg(enstrophy)), name='Ro')
traces.add_task(np.sqrt(avg(KE)*2)/Ek, name='Re')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(Lz), name='Lz')
traces.add_task(np.abs(??_p), name='??_p')
traces.add_task(np.abs(??_??), name='??_??')
traces.add_task(shellavg(np.abs(??_s)), name='??_s')
traces.add_task(shellavg(np.sqrt(dot(??_u,??_u))), name='??_u')
traces.add_task(shellavg(np.sqrt(dot(??_A,??_A))), name='??_A')

# Analysis
e?? = d.VectorField(c, bases=b)
e??['g'][0] = 1
er = d.VectorField(c, bases=b)
er['g'][2] = 1
B?? = dot(B, e??)
A?? = dot(A, e??)
??_cyl = d.Field(bases=b)
??_cyl['g'] = r*np.sin(theta)
??z = dot(u, e??)/??_cyl # this is not ??_z; misses gradient terms; this is angular differential rotation.

slice_dt = float(args['--slice_dt'])
slices = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt = slice_dt, max_writes = 10, virtual_file=True, mode=mode)
slices.add_task(s(theta=np.pi/2), name='s')
slices.add_task(enstrophy(theta=np.pi/2), name='enstrophy')
slices.add_task(azavg(B??), name='<B??>')
slices.add_task(azavg(A??), name='<A??>')
slices.add_task(azavg(??z), name='<??z>')
slices.add_task(azavg(s), name='<s>')
slices.add_task(shellavg(s), name='s(r)')
slices.add_task(shellavg(Co2*dot(er, u)*s), name='F_h(r)')
slices.add_task(shellavg(dot(er, u)*dot(u,u)), name='F_KE(r)')
slices.add_task(shellavg(-Co2*Ek/Pr*dot(er, grad(s))), name='F_??(r)')
slices.add_task(shellavg(Co2*source), name='Q_source(r)')
slices.add_task(dot(B,er)(r=radius), name='Br') # is this sufficient?  Should we be using radial(B) instead?

checkpoint = solver.evaluator.add_file_handler(data_dir+'/checkpoints', wall_dt = 3600, max_writes = 1, virtual_file=True, mode=mode)
checkpoint.add_tasks(solver.state)

report_cadence = 100
flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(np.sqrt(KE*2)/Ek, name='Re')
flow.add_property(np.sqrt(enstrophy), name='Ro')
flow.add_property(KE, name='KE')
flow.add_property(ME, name='ME')
flow.add_property(PE, name='PE')
flow.add_property(Lz, name='Lz')
flow.add_property(np.sqrt(dot(??_u,??_u)), name='|??_u|')
flow.add_property(np.abs(??_s), name='|??_s|')
flow.add_property(np.sqrt(dot(??_A,??_A)), name='|??_A|')

cfl_safety_factor = float(args['--safety'])
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety_factor, max_dt=max_dt, threshold=0.1)
CFL.add_velocity(u)
CFL.add_velocity(B)

if args['--run_time_rotation']:
    solver.stop_sim_time = float(args['--run_time_rotation'])
else:
    solver.stop_sim_time = float(args['--run_time_diffusion'])/Ek

if args['--run_time_iter']:
    solver.stop_iteration = int(float(args['--run_time_iter']))

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
        ??_u_m = flow.max('|??_u|')
        ??_s_m = flow.max('|??_s|')
        ??_A_m = flow.max('|??_A|')
        log_string = "iter: {:d}, dt={:.1e}, t={:.3e} ({:.2e})".format(solver.iteration, dt, solver.sim_time, solver.sim_time*Ek)
        log_string += ", KE={:.2e}, ME={:.2e}, PE={:.2e}".format(KE_avg, ME_avg, PE_avg)
        log_string += ", Re={:.1e}, Ro={:.1e}".format(Re_avg, Ro_avg)
        log_string += ", Lz={:.1e}, ??=({:.1e},{:.1e},{:.1e})".format(Lz_avg, ??_u_m, ??_s_m, ??_A_m)
        logger.info(log_string)

        good_solution = np.isfinite(E0)
    solver.step(dt)

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(N??*N??*Nr)))

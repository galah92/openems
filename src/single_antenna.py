from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

### General parameter setup
filename = Path(__file__).stem
sim_path = Path(__file__).parent / "sim" / filename
sim_path.mkdir(parents=True, exist_ok=True)

### Antenna array parameters
patch_width = 32  # patch width (resonant length) in x-direction
patch_length = 40  # patch length in y-direction

# substrate setup
substrate_epsR = 3.38
substrate_kappa = 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * substrate_epsR
substrate_width = 60
substrate_length = 60
substrate_thickness = 1.524
substrate_cells = 4

# setup feeding
feed_pos = -6  # feeding position in x-direction
feed_R = 50  # feed resistance

# size of the simulation box
SimBox = np.array([200, 200, 150])

# setup FDTD parameter & excitation function
f0 = 2e9  # center frequency
fc = 1e9  # 20 dB corner frequency

### FDTD setup
## * Limit the simulation to 30k timesteps
## * Define a reduced end criteria of -40dB
FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

### Setup CSXCAD geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

# setup the mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0 / (f0 + fc) / 1e-3 / 20

### Generate properties, primitives and mesh-grid
# initialize the mesh with the "air-box" dimensions
mesh.AddLine("x", [-SimBox[0] / 2, SimBox[0] / 2])
mesh.AddLine("y", [-SimBox[1] / 2, SimBox[1] / 2])
mesh.AddLine("z", [-SimBox[2] / 3, SimBox[2] * 2 / 3])

# create patch
patch = CSX.AddMetal("patch")  # create a perfect electric conductor (PEC)
start = [-patch_width / 2, -patch_length / 2, substrate_thickness]
stop = [patch_width / 2, patch_length / 2, substrate_thickness]
patch.AddBox(
    priority=10, start=start, stop=stop
)  # add a box-primitive to the metal property 'patch'
FDTD.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=mesh_res / 2)

# create substrate
substrate = CSX.AddMaterial("substrate", epsilon=substrate_epsR, kappa=substrate_kappa)
start = [-substrate_width / 2, -substrate_length / 2, 0]
stop = [substrate_width / 2, substrate_length / 2, substrate_thickness]
substrate.AddBox(priority=0, start=start, stop=stop)

# add extra cells to discretize the substrate thickness
mesh.AddLine("z", np.linspace(0, substrate_thickness, substrate_cells + 1))

# create ground (same size as substrate)
gnd = CSX.AddMetal("gnd")  # create a perfect electric conductor (PEC)
start[2] = 0
stop[2] = 0
gnd.AddBox(start=start, stop=stop, priority=10)

FDTD.AddEdges2Grid(dirs="xy", properties=gnd)

# apply the excitation & resist as a current source
start = [feed_pos, 0, 0]
stop = [feed_pos, 0, substrate_thickness]
port = FDTD.AddLumpedPort(1, feed_R, start, stop, "z", 1.0, priority=5, edges2grid="xy")

mesh.SmoothMeshLines("all", mesh_res, 1.4)

# Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox()

### Run the simulation
save_csx_xml = True
if save_csx_xml:
    res = CSX.Write2XML(sim_path / "csx.xml")

post_proc_only = False
if not post_proc_only:
    FDTD.Run(sim_path)


### Post-processing and plotting
f = np.linspace(max(1e9, f0 - fc), f0 + fc, 401)
port.CalcPort(sim_path, f)
s11 = port.uf_ref / port.uf_inc
s11_dB = 20.0 * np.log10(np.abs(s11))
plt.figure()
plt.plot(f / 1e9, s11_dB, "k-", linewidth=2, label="$S_{11}$")
plt.grid()
plt.legend()
plt.ylabel("S-Parameter (dB)")
plt.xlabel("Frequency (GHz)")
plt.savefig("S11.png")

idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
if not len(idx) == 1:
    raise Exception("No resonance frequency found for far-field calulation")

f_res = f[idx[0]]
theta = np.arange(-180.0, 180.0, 2.0)
phi = np.arange(-180.0, 180.0, 2.0)
nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0, 0, 1e-3])

plt.figure()
E_norm, Dmax = nf2ff_res.E_norm, nf2ff_res.Dmax
E_norm = 20.0 * np.log10(E_norm[0] / np.max(E_norm[0])) + 10.0 * np.log10(Dmax[0])
plt.plot(theta, np.squeeze(E_norm[:, 0]), "k-", linewidth=2, label="xz-plane")
plt.plot(theta, np.squeeze(E_norm[:, 1]), "r--", linewidth=2, label="yz-plane")
plt.grid()
plt.ylabel("Directivity (dBi)")
plt.xlabel("Theta (deg)")
plt.title("Frequency: {} GHz".format(f_res / 1e9))
plt.legend()
plt.savefig("Directivity.png")

Zin = port.uf_tot / port.if_tot
plt.figure()
plt.plot(f / 1e9, np.real(Zin), "k-", linewidth=2, label="$\Re\{Z_{in}\}$")
plt.plot(f / 1e9, np.imag(Zin), "r--", linewidth=2, label="$\Im\{Z_{in}\}$")
plt.grid()
plt.legend()
plt.ylabel("Zin (Ohm)")
plt.xlabel("Frequency (GHz)")
plt.savefig("Zin.png")

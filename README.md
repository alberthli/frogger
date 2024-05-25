# FRoGGeR: A Fast Robust Grasp Generator

This repository houses the code for the paper "[FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric](https://arxiv.org/abs/2302.13687)."

[Nov. 14, 2023] The codebase has been refactored for better general usability and a cleaner API. To facilitate this, a lot of minutiae has been stripped away that was present in the original code release. If you are looking to investigate the original paper results, those can be found on the `iros-release` branch. If you simply want to try to use FRoGGeR for grasp synthesis, use the `main` branch.

## Installation
We recommend using a `conda` environment to run the code in this repo. To get the latest changes, install as an editable package in the conda environment using
```
pip install -e .
```
If you have a MOSEK license and would like to activate the MOSEK solver in `Drake`, then place the license file `mosek.lic` in the repository root and run `setup_mosek.sh`.

## Usage
There are three major components to using FRoGGeR to sample grasps:
1. a FRoGGeR-compliant robot model,
2. a description of the target object, and
3. an initial condition sampler for the nonlinear optimization.

We provide usage examples for a few different robots:
* the floating Allegro hand,
* the floating underactuated Barrett Hand,
* the Allegro hand attached to the Franka Research 3 on a tabletop, and
* the Allegro hand attached to the Franka Research 3 on a tabletop with a Zed2i camera.
All examples synthesize grasps on a subset of the YCB dataset. The usage script times the execution of grasp generation for these objects on your system, and is located in `scripts/timing.py`. This script also gives an example of how to adjust an existing robot with a baseline non-FRoGGeR solver.

For example, synthesizing a grasp with the Allegro hand can be done in the following ~50 lines of code. Many of the options have reasonable default values already and are just explicitly assigned in this example to demonstrate what is available. 
```
import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig
from frogger.sampling import HeuristicAlgrICSampler
from frogger.solvers import FroggerConfig
from frogger.utils import timeout

# loading object
obj_name = "001_chips_can"
mesh = trimesh.load(ROOT + f"/data/{obj_name}/{obj_name}_clean.obj")
bounds = mesh.bounds
lb_O = bounds[0, :]
X_WO = RigidTransform(RotationMatrix(), np.array([0.0, 0.0, -lb_O[-1]]))
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

# loading model
model = AlgrModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
).create()

# creating solver and generating grasp
frogger = FroggerConfig(
    model=model,
    sampler=HeuristicAlgrICSampler(model),
    tol_surf=1e-3,
    tol_joint=1e-2,
    tol_col=1e-3,
    tol_fclosure=1e-5,
    xtol_rel=1e-6,
    xtol_abs=1e-6,
    maxeval=1000,
    maxtime=60.0,
).create()

print("Model compiled! Generating grasp...")
q_star = timeout(60.0)(frogger.generate_grasp)()  # our timeout util allows arbitrary timeouts for any func!
print("Grasp generated!")
model.viz_config(q_star)
```

Note that the first time you run the code, there will be `numba` compilation happening, so it will be slower. Afterwards, the compiled code should be cached, so everything will run quickly.

### Robot Model
We allow robot descriptions using URDFs or SDFs. We enforce that the _entire_ robot must be described in a single file. Important details that are required in the description:
* You must specify which collision geoms are allowed to touch the object. If a body has any geoms of this type, then those bodies MUST touch it (but only the specified geoms must touch). Such collision geometries are automatically parsed if the substring `FROGGERCOL` is in the name. 
* If you use the provided heuristic sampler, you must specify a canonical "dummy palm" frame in the robot description. The sampler will look for this frame to help place the hand. For an example, see the bottom of `allegro_rh.sdf`.
* If you define custom collision geometries from convex meshes, add the `drake:convex` tag to the URDF so `Drake` knows that the geometry is convex at parsing time.

### Object
Objects can be defined either by a supplied (watertight) mesh or by an analytical SDF. If you supply an analytical SDF, it must be written in `jax` to be compatible with the provided API, though you can easily add your own functionality.

### Sampler
FRoGGeR is sensitive to the choice of initial guess, because it's just solving a complicated nonlinear optimization program under the hood. We supply a coarse heuristic that performs reasonably well on a decent range of objects, but it can certainly be improved upon. You can implement your own custom samplers using our API.

## Citation
If you found our work useful (either the paper or the code), please use the following citation:

```
@article{
    li2023_frogger,
    title={FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric},
    author={Albert H. Li and Preston Culbertson and Joel W. Burdick and Aaron D. Ames},
    journal={arxiv:2302.13687},
    month={February},
    year={2023},
}
```

# FRoGGeR: A Fast Robust Grasp Generator

This repository houses the code for the paper "[FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric](https://arxiv.org/abs/2302.13687)."

[Nov. 14, 2023] The codebase has been refactored for better general usability and a cleaner API. To facilitate this, a lot of minutiae has been stripped away that was present in the original code release. If you are looking to investigate the original paper results, those can be found on the `iros-release` branch. If you simply want to try to use FRoGGeR for grasp synthesis, use the `main` branch.

## Installation
We recommend using a `conda` environment to run the code in this repo. To get the latest changes, install as an editable package in the conda environment using
```
pip install -e .
```
If you have a MOSEK license and would like to activate the MOSEK solver in `Drake`, then place the license file `mosek.lic` in the repository root and 

## Usage
There are three major components to using FRoGGeR to sample grasps:
1. a compliant robot model,
2. a description of the target object, and
3. an initial condition sampler for the nonlinear optimization.

We provide usage examples for 3 different robots (The floating Allegro hand, the floating underactuated Barrett Hand, and the Allegro hand attached to the Franka Research 3 on a tabletop) and on a subset of the YCB dataset. We also provide a script that times the execution of grasp generation for these objects on your system.

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

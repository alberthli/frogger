# FRoGGeR: A Fast Robust Grasp Generator

This repository houses the code for the paper "[FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric](https://arxiv.org/abs/2302.13687)."

## Installation
We recommend using a `conda` environment to run the code in this repo. To get the latest changes, install as an editable package in the conda environment using
```
pip install -e .
```
If you have a MOSEK license and would like to activate the MOSEK solver in `Drake`, then place the license file `mosek.lic` in the repository root and 

## Usage
TODO: add examples

To sample grasps on a custom manipulator, you need the following:
* a description of the manipulator in URDF or SDF form,
    * in the description, you must specify which collision geoms are allowed to touch the object. If a body has any geoms of this type, then those bodies MUST touch it
    * additionally, if using the provided heuristic IK sampler, a canonical dummy "palm" frame must be specified, which the sampler will look for
* a description of the object, either as a mesh or as an analytical SDF,
* an implementation of an IK sampler to produce initial guesses for the solver (NOTE: this is probably the most important part of the robot-specific implementations, as the nonlinear solver is very sensitive to the choice of initial guess)

## Upcoming Changes
This branch is the development branch for the upcoming FRoGGeR refactor.

Post-IROS, there are some planned changes to the codebase. If you have suggestions, feel free to open an issue!
- [x] Update Drake to 1.22.0, removing Dockerization requirement since `pip` issues were fixed (see [#19515](https://github.com/RobotLocomotion/drake/pull/19515)).
- [x] Fix internal paths, which are hardcoded based on the Docker container internals, to be relative package paths
- [x] Refactor dependency management onto `pyproject.toml`
- [x] Significantly loosen requirements due to moving away from Docker
- [ ] Add ability to scale joint/torque limits from URDF default
- [ ] Before pypi package release, update instructions for using the MOSEK license.
- [X] Handle multiple collision geoms on robot AND on object for surface constraint
- [x] Force the user to supply a single URDF or SDF for the entire system to greatly simplify the pipeline (requires a relatively large refactor of the sim pipeline)
- [x] Abstract the initial condition sampler so that people can implement their own.
- [x] Simplify `RobotModel` abstract API:
    - [x] Automatically read joint limit/torque bounds from file
    - [x] Remove requirement for specifying prescribed contact locations on fingertips, simply require that the user must specify the collision geometries in the URDF/SDF
    - [x] Move burden of custom implementation as much as possible to heuristic sampler.
    - [x] Simplify the expected implementation in the `__init__` function by creating configuration dataclasses.
- [ ] Add compatibility with underactuated hands (e.g., Barrett or Shadow).
- [ ] Add usage examples + a timing script.
- [ ] Clean up unused scripts and files.

### Later Changes
There are a few changes that won't be immediately addressed, but are on the radar:
- [ ] Add back in the simulation pipeline with controllers. This was removed because it cluttered the grasp refinement API and the simulator implementation isn't universal. However, if you'd like, you can check the main idea out in the old branch.
- [ ] Add back in baseline code
- [ ] Improve sampling heuristic
- [ ] Refactor from `numba` to `jax`
- [ ] Refactor solver from `nlopt` to Drake's SNOPT bindings.

## Reproducibility
This code has been refactored from the original paper implementation to emphasize usability for custom robots. However, the original code can be found on the branch `iros-release`.

## Citation
If you found our work useful (either the paper or the code), please use the following citation:

```
Albert H. Li, Preston Culbertson, Joel W. Burdick, Aaron D. Ames, "FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric," arXiv:2302.13687, Feb. 2023.
```

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

from dataclasses import dataclass, fields, make_dataclass
from typing import Callable, Tuple

import numpy as np
import torch
from qpth.qp import QPFunction

from frogger.robots.robot_core import RobotModel, RobotModelConfig

# ##### #
# UTILS #
# ##### #


def combine_dataclasses(cls_a: type, cls_b: type) -> type:
    """Combines two config dataclasses into a new one, prioritizing attributes of cls_b.

    Parameters
    ----------
    cls_a : type
        The first dataclass type.
    cls_b : type
        The second dataclass type.

    Returns
    -------
    new_cls : type
        The new dataclass type.
    """
    # extract attributes from cls_a
    attributes = {
        field.name: (field.type, getattr(cls_a, field.name)) for field in fields(cls_a)
    }

    # remove fields that are methods in cls_b
    for k in list(attributes.keys()):
        if k in cls_b.__dict__ and isinstance(getattr(cls_b, k), Callable):
            del attributes[k]

    # include attributes from cls_b, prioritizing non-callable attributes
    for field in fields(cls_b):
        if field.name not in attributes and not isinstance(
            getattr(cls_b, field.name), Callable
        ):
            attributes[field.name] = field.type, getattr(cls_b, field.name)

    # combine methods, prioritizing those from B
    # note that we look through ALL attributes of cls_a, but only through the __dict__ of cls_b.
    # this is so that we have inherited methods but only override them with cls_b impls if they
    # are explicitly overridden.
    methods = {}
    for method_name in dir(cls_a):
        method = getattr(cls_a, method_name)
        if isinstance(method, Callable) and method_name not in cls_b.__dict__:
            methods[method_name] = method
    for method_name, method in cls_b.__dict__.items():
        if isinstance(method, Callable):
            methods[method_name] = method

    # generate the new class
    new_class_name = cls_a.__name__ + cls_b.__name__.replace("Config", "")
    new_cls = make_dataclass(
        new_class_name,
        [(name, type_, default) for name, (type_, default) in attributes.items()],
        kw_only=True,
        bases=(cls_a, cls_b),
    )

    # add methods to the new dataclass
    for method_name, method in methods.items():
        if method_name != "__class__":
            setattr(new_cls, method_name, method)

    return new_cls


# ######### #
# BASELINES #
# ######### #


@dataclass(kw_only=True)
class BaselineConfig(RobotModelConfig):
    """Configuration for a generic baseline.

    Typically, we want to modify some RobotModelConfig that has already been
    instantiated with desirable properties. To do this, we will use the `from_cfg`
    method of BaselineConfig to load all of the attributes of an existing config
    into a baseline config that is the combination of the baseline attributes and
    the original config attributes.
    """

    @classmethod
    def from_cfg(cls, config_cls: type, **kwargs) -> "BaselineConfig":
        """Creates a baseline config from an existing robot model config class."""
        CombinedDataClass = combine_dataclasses(config_cls, cls)
        return CombinedDataClass


@dataclass(kw_only=True)
class WuBaselineConfig(BaselineConfig):
    """Configuration for the Wu baseline solver."""

    n_g_extra: int = 0
    n_h_extra: int = 1

    def _init_baseline_cons(self, model: RobotModel) -> None:
        """Initializes constraints for the baseline.

        Equation (5) of the paper:
        "Learning Diverse and Physically Feasible Dexterous Grasps with Generative
        Model and Bilevel Optimization"
        """
        # setting up the feasibility QP
        # (i) friction cone constraint
        Lambda_i = np.vstack(
            (
                np.append(np.cos(2 * np.pi * np.arange(model.ns) / model.ns), 0.0),
                np.append(np.sin(2 * np.pi * np.arange(model.ns) / model.ns), 0.0),
                -np.append(
                    model.mu * np.cos(np.pi * np.ones(model.ns) / model.ns), 1.0
                ),
            )
        ).T  # pyramidal friction cone approx in contact frame + min normal force

        # Ain has shape((ns + 1) * nc, 3 * nc)
        fn_min = 1.0  # hard code min normal force to 1, that's what they use
        A_in = torch.tensor(np.kron(np.eye(model.nc), Lambda_i))
        b_in = torch.zeros((model.ns + 1) * model.nc).double()
        b_in[model.ns :: model.ns + 1] = -fn_min

        A_eq = torch.Tensor().double()  # empty
        b_eq = torch.Tensor().double()

        # (ii) setting up QP constraint function
        def bilevel_constraint_func(G: torch.Tensor) -> torch.Tensor:
            """Constraint function for bilevel optimization.

            Takes in the grasp map and returns a cost.
            """
            if G.dtype == torch.float32:
                G = G.double()
            Q = G.T @ G + 1e-7 * torch.eye(3 * model.nc).double()
            f_opt = QPFunction(verbose=-1, check_Q_spd=False)(
                Q, torch.zeros(3 * model.nc).double(), A_in, b_in, A_eq, b_eq
            ).squeeze()
            h = f_opt @ Q @ f_opt
            return h

        # adding the bilevel constraint to the model
        model._bilevel_cons = bilevel_constraint_func

    @staticmethod
    def custom_compute_l(robot: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Cost function for Wu baseline. Since it's a feas program, does nothing."""
        return 0.0, np.zeros(robot.n)

    @staticmethod
    def custom_compute_h(robot: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Extra equality constraints for Wu baseline. This is the force closure QP.

        Warning
        -------
        Requires robot.DG to have been computed and cached already!
        """
        # equality constraint
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        G = torch.tensor(robot.G).double()
        h = (robot._bilevel_cons(G)).detach().numpy()[..., None]  # (1,)

        # gradient
        Dh_G = (
            torch.autograd.functional.jacobian(
                robot._bilevel_cons, G, create_graph=True, strict=True
            )
            .detach()
            .numpy()
            .astype(np.float64)
        )
        DG = robot.DG
        Dh = (Dh_G.reshape(-1) @ DG.reshape((-1, DG.shape[-1])))[None, ...]  # (1, n)

        return h, Dh

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Initializes the baseline constraints."""
        self._init_baseline_cons(model)

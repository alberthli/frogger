from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from qpth.qp import QPFunction

from frogger.baselines.base import BaselineConfig
from frogger.robots.robot_core import RobotModel


@dataclass(kw_only=True)
class WuBaselineConfig(BaselineConfig):
    """Configuration for the Wu baseline solver.

    Note
    ----
    All of the torch functions use the default device, which is CPU unless changed
    elsewhere. In testing, the device did not have a large difference on run time.
    """

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
    def custom_compute_l(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Cost function for Wu baseline. Since it's a feas program, does nothing."""
        return 0.0, np.zeros(model.n)

    @staticmethod
    def custom_compute_h(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Extra equality constraints for Wu baseline. This is the force closure QP.

        Warning
        -------
        Requires model.DG to have been computed and cached already!
        """
        # equality constraint
        G = torch.tensor(model.G).double()
        h = (model._bilevel_cons(G)).cpu().detach().numpy()[..., None]  # (1,)

        # gradient
        Dh_G = (
            torch.autograd.functional.jacobian(
                model._bilevel_cons, G, create_graph=True, strict=True
            )
            .cpu()
            .detach()
            .numpy()
        )
        DG = model.DG
        Dh = (Dh_G.reshape(-1) @ DG.reshape((-1, DG.shape[-1])))[None, ...]  # (1, n)

        return h, Dh

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Initializes the baseline constraints."""
        self._init_baseline_cons(model)

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_kinematics as pk
import torch

from frogger import ROOT
from frogger.baselines.base import BaselineConfig
from frogger.robots.robot_core import RobotModel
from frogger.robots.robots import AlgrModel, FR3AlgrModel, FR3AlgrZed2iModel

try:
    import nerf_grasping

    # from nerf_grasping.classifier import Classifier
    from nerf_grasping.optimizer_utils import AllegroGraspConfig, GraspMetric
except ImportError:
    raise ImportError(
        "The NeRF grasping baseline requires the `nerf_grasping` package. "
        "Please install it with frogger using `pip install -e .[ng]` or "
        "`pip install -e .[all]`."
    )


@dataclass(kw_only=True)
class NerfGraspingBaselineConfig(BaselineConfig):
    """Configuration for the NeRF grasping baseline."""

    metric: GraspMetric  # TODO(ahl): replace this with the inputs to load it

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.model_class in [AlgrModel, FR3AlgrModel, FR3AlgrZed2iModel]
        # TODO(ahl): for now, we use pytorch_kinematics, so we must ensure we have
        # URDFs of all the relevant models for compatibility reasons.
        assert self.model_class == FR3AlgrZed2iModel
        if self.model_class in AlgrModel:
            assert self.hand == "rh"  # TODO(ahl): for now, rh only
            raise NotImplementedError
        elif self.model_class in FR3AlgrModel:
            raise NotImplementedError
        elif self.model_class in FR3AlgrZed2iModel:
            model_path = Path(ROOT) / "models/fr3_algr_zed2i/fr3_algr_zed2i.urdf"
            with open(model_path) as f:
                self.chain = pk.build_chain_from_urdf(f.read())
        super().__post_init__()

    def _init_baseline_obj(self, model: RobotModel) -> None:
        """Initialize the objective function."""

        def q_to_failure_prob(
            q: torch.Tensor, grasp_orientation: torch.Tensor
        ) -> torch.Tensor:
            """From the model configuration, computes the failure prob."""
            # computing the entries of the grasp config dict
            if self.model_class in AlgrModel and self.hand == "lh":
                body_name = "algr_lh_palm"
            else:
                body_name = "algr_rh_palm"
            X_WWrist = self.chain.forward_kinematics(
                q, self.chain.get_frame_indices(body_name)
            )
            X_WO = torch.tensor(model.obj.X_WO.GetAsMatrix4())  # (4, 4)
            X_OWrist = torch.inverse(X_WO) @ X_WWrist  # TODO(ahl): not optimized
            p_OWrist = X_OWrist[:3, 3]
            _R_OWrist = X_OWrist[:3, :3]
            R_OOyup = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )  # rotate to y up
            R_OWrist = R_OOyup.T @ _R_OWrist

            # defining the grasp config dict
            gcd = {
                "trans": torch.tensor(p_OWrist),
                "rot": torch.tensor(R_OWrist),
                "joint_angles": torch.tensor(q[-16:]),
                "grasp_orientation": grasp_orientation,
            }

            # passing the grasp config dict to the metric object
            grasp_config = AllegroGraspConfig.from_grasp_config_dict(gcd)
            failure_prob = self.grasp_metric.get_failure_probability(grasp_config)
            return failure_prob

        # adding this function to the model attributes
        model._q_to_failure_prob = q_to_failure_prob

    @staticmethod
    def custom_compute_l(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Cost function for NeRF grasping baseline."""
        R_OOyup = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )  # rotate to y up

        # computing the failure probability
        grasp_orientation = torch.tensor(R_OOyup.T @ model.R_cf_O, requires_grad=True)
        q = torch.tensor(model.q, requires_grad=True)
        _l = model._q_to_failure_prob(q, grasp_orientation).cpu().detach().numpy()

        # compute gradients using chain rule + total derivative
        _l.backward()
        Dl_q = q.grad.cpu().detach().numpy()  # (23,), this is a partial derivative
        Dl_go = grasp_orientation.grad.cpu().detach().numpy()  # (4, 3, 3)
        Dgo_q = np.einsum(
            "jk,iklm->ijlm", R_OOyup.T.cpu().numpy(), model.DR_cf_O
        )  # (4, 3, 3, 23)
        _Dl = (Dl_q + torch.einsum("ijk,ijkl->l", Dl_go, Dgo_q)).cpu().detach().numpy()
        return -_l, -_Dl

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Initializes the baseline constraints."""
        self._init_baseline_obj(model)

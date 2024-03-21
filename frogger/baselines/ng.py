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
    from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
    from nerf_grasping.optimizer_utils import AllegroGraspConfig, GraspMetric
except ImportError:
    raise ImportError(
        "The NeRF grasping baseline requires the `nerf_grasping` package. "
        "Please install it with frogger using `pip install -e .[ng]` or "
        "`pip install -e .[all]`."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(kw_only=True)
class NerfGraspingBaselineConfig(BaselineConfig):
    """Configuration for the NeRF grasping baseline."""

    # def __post_init__(self) -> None:
    #     """Post-initialization checks."""
    #     # TODO(ahl): figure out how to annotate the model class correctly
    #     super().__post_init__()

    def _init_baseline_obj(self, model: RobotModel) -> None:
        """Initialize the objective function."""
        # checking the model class to initialize the kinematics chain
        model_class = type(self).__name__
        substrings = ["AlgrModel", "FR3AlgrModel", "FR3AlgrZed2iModel"]
        assert any(sub in model_class for sub in substrings)

        # TODO(ahl): for now, we use pytorch_kinematics, so we must ensure we have
        # URDFs of all the relevant models for compatibility reasons.
        assert "FR3AlgrZed2iModel" in model_class
        if "AlgrModel" in model_class:
            assert self.hand == "rh"  # TODO(ahl): for now, rh only
            raise NotImplementedError
        elif "FR3AlgrModel" in model_class:
            raise NotImplementedError
        elif "FR3AlgrZed2iModel" in model_class:
            model_path = Path(ROOT) / "models/fr3_algr_zed2i/fr3_algr_zed2i.urdf"
            with open(model_path) as f:
                chain = pk.build_chain_from_urdf(f.read())
                chain = chain.to(device="cuda", dtype=torch.float32)

        # initializing the grasp metric
        base_path = Path(ROOT) / "frogger/baselines"
        grasp_metric_config = GraspMetricConfig(
            # classifier_config=None,  # use default
            classifier_config_path=base_path / "classifier/config.yaml",
            classifier_checkpoint=-1,  # load latest checkpoint
            nerf_checkpoint_path=base_path / "nerf/config.yml",
            object_transform_world_frame=None,  # TODO(ahl): fill this in
        )
        grasp_metric = GraspMetric.from_config(grasp_metric_config)
        grasp_metric = grasp_metric.to(device)
        grasp_metric.eval()

        def q_to_failure_prob(
            q: torch.Tensor, grasp_orientations: torch.Tensor
        ) -> torch.Tensor:
            """From the model configuration, computes the failure prob."""
            # computing the entries of the grasp config dict
            if "AlgrModel" in model_class and self.hand == "lh":
                body_name = "algr_lh_palm"
            else:
                body_name = "algr_rh_palm"
            X_WWrist = chain.forward_kinematics(q, chain.get_frame_indices(body_name))[
                "algr_rh_palm"
            ].get_matrix()
            X_WO = torch.tensor(
                model.obj.X_WO.GetAsMatrix4(), device=device, dtype=torch.float32
            )  # (4, 4)
            X_OWrist = torch.inverse(X_WO) @ X_WWrist  # TODO(ahl): not optimized
            p_OWrist = X_OWrist[..., :3, 3]
            _R_OWrist = X_OWrist[..., :3, :3]
            R_OOyup = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                device=device,
                dtype=torch.float32,
            )  # rotate to y up
            R_OWrist = R_OOyup.T @ _R_OWrist

            # defining the grasp config dict
            gcd = {
                "trans": p_OWrist,  # (1, 3)
                "rot": R_OWrist,  # (1, 3, 3)
                "joint_angles": q[None, -16:],  # (1, 16)
                "grasp_orientations": grasp_orientations[None, ...],  # (1, 4, 3, 3)
            }

            # passing the grasp config dict to the metric object
            grasp_config = AllegroGraspConfig.from_grasp_config_dict(
                gcd, numpy_inputs=False
            )[0:1]
            failure_prob = grasp_metric.get_failure_probability(grasp_config)

            ##################
            # def func(t, r, j, g):
            #     gcd = {
            #         "trans": t,  # (1, 3)
            #         "rot": r,  # (1, 3, 3)
            #         "joint_angles": j,  # (1, 16)
            #         "grasp_orientations": g,  # (1, 4, 3, 3)
            #     }
            #     grasp_config = AllegroGraspConfig.from_grasp_config_dict(gcd, numpy_inputs=False)[0:1]
            #     grasp_config.wrist_pose.requires_grad = True
            #     grasp_config.grasp_orientations.requires_grad = True
            #     grasp_config.joint_angles.requires_grad = True
            #     failure_prob = grasp_metric.get_failure_probability(grasp_config)
            #     return failure_prob, grasp_config

            # t = p_OWrist.clone().detach().requires_grad_(True)
            # r = R_OWrist.clone().detach().requires_grad_(True)
            # j = q[None, -16:].clone().detach().requires_grad_(True)
            # g = grasp_orientations[None, ...].clone().detach().requires_grad_(True)
            # asdf, grasp_config = func(t, r, j, g)
            # asdf.backward()
            # print(t.grad)
            # print(r.grad)
            # print(j.grad)
            # print(g.grad)
            # print(f"grasp_config.wrist_pose.grad = {grasp_config.wrist_pose.grad}")
            ##################

            breakpoint()
            return failure_prob

        # adding this function to the model attributes
        model._q_to_failure_prob = q_to_failure_prob

    @staticmethod
    def custom_compute_l(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Cost function for NeRF grasping baseline."""
        R_OOyup = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )  # rotate to y up

        # computing the failure probability
        grasp_orientations = (
            R_OOyup.T @ torch.tensor(model.R_cf_O, device=device, dtype=torch.float32)
        ).requires_grad_(True)
        q = torch.tensor(model.q, device=device, dtype=torch.float32).requires_grad_(
            True
        )
        _l_torch = model._q_to_failure_prob(q, grasp_orientations)
        _l = _l_torch.cpu().detach().numpy()

        # compute gradients using chain rule + total derivative
        _l_torch.backward()
        Dl_q = q.grad.cpu().detach().numpy()  # (23,), this is a partial derivative
        Dl_go = grasp_orientations.grad.cpu().detach().numpy()  # (4, 3, 3)
        Dgo_q = np.einsum(
            "jk,iklm->ijlm", R_OOyup.T.cpu().numpy(), model.DR_cf_O
        )  # (4, 3, 3, 23)
        _Dl = (Dl_q + torch.einsum("ijk,ijkl->l", Dl_go, Dgo_q)).cpu().detach().numpy()
        return -_l, -_Dl

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Initializes the baseline constraints."""
        self._init_baseline_obj(model)

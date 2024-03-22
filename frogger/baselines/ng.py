from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pypose as pp
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

    n_g_extra: int = 1
    min_success_prob: float = 0.0

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
        # TODO(ahl): expose this somehow
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

        FINGERTIP_LINK_NAMES = [
            "algr_rh_if_ds_tip",
            "algr_rh_mf_ds_tip",
            "algr_rh_rf_ds_tip",
            "algr_rh_th_ds_tip",
        ]

        R_OOyup = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )  # rotate to y up

        def q_to_failure_prob(
            q: torch.Tensor, grasp_orientations: torch.Tensor
        ) -> torch.Tensor:
            """From the model configuration, computes the failure prob."""
            # computing the entries of the grasp config dict
            if "AlgrModel" in model_class and self.hand == "lh":
                body_name = "algr_lh_palm"
            else:
                body_name = "algr_rh_palm"
            link_poses_hand_frame = chain.forward_kinematics(q)
            X_WWrist = link_poses_hand_frame[body_name].get_matrix()
            X_WO = torch.tensor(
                model.obj.X_WO.GetAsMatrix4(), device=device, dtype=torch.float32
            )  # (4, 4)
            X_OWrist = torch.inverse(X_WO) @ X_WWrist  # TODO(ahl): not optimized
            R_OyupWrist = R_OOyup.T @ X_OWrist[:3, :3]
            X_OyupWrist = X_OWrist
            X_OyupWrist[:3, :3] = R_OyupWrist

            # computing fingertip transforms
            fingertip_poses = [link_poses_hand_frame[ln] for ln in FINGERTIP_LINK_NAMES]
            fingertip_pyposes = [
                pp.from_matrix(fp.get_matrix(), pp.SE3_type) for fp in fingertip_poses
            ]
            wrist_pose = pp.from_matrix(X_OyupWrist, ltype=pp.SE3_type)
            X_WristW_pp = pp.from_matrix(torch.inverse(X_WWrist), ltype=pp.SE3_type)
            fingertip_transforms = torch.stack(
                [wrist_pose @ X_WristW_pp @ fp for fp in fingertip_pyposes], dim=1
            )

            # call alternative forward pass to avoid using grasp_config, which breaks the
            # compute graph and prevents us from getting gradients
            fingertip_positions_pp = fingertip_transforms.translation()  # (1, nc, 3)
            grasp_orientations_pp = pp.from_matrix(
                grasp_orientations[None, ...], pp.SO3_type
            )  # (1, nc, 4)
            grasp_frame_transforms = pp.SE3(
                torch.cat(
                    [
                        fingertip_positions_pp,
                        grasp_orientations_pp,
                    ],
                    dim=-1,
                )
            )
            failure_prob = grasp_metric.forward_alt(grasp_frame_transforms)
            return failure_prob

        # adding this function to the model attributes
        model._q_to_failure_prob = q_to_failure_prob

    @staticmethod
    def custom_compute_g(model: RobotModel) -> Tuple[np.ndarray, np.ndarray]:
        """Adding minimum success probability constraint."""
        g_extra = -model.l + model.min_success_prob  # success probability
        Dg_extra = -model.Dl
        return g_extra, Dg_extra

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
        _l = _l_torch.item()

        # compute gradients using chain rule + total derivative
        _l_torch.backward()
        Dl_q = q.grad.cpu().detach().numpy()  # (23,), this is a partial derivative
        Dl_go = grasp_orientations.grad.cpu().detach().numpy()  # (4, 3, 3)
        Dgo_q = np.einsum(
            "jk,iklm->ijlm", R_OOyup.T.cpu().numpy(), model.DR_cf_O
        )  # (4, 3, 3, 23)
        _Dl = Dl_q + np.einsum("ijk,ijkl->l", Dl_go, Dgo_q)
        return 1.0 - _l, -_Dl

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Initializes the baseline constraints."""
        model.min_success_prob = self.min_success_prob
        self._init_baseline_obj(model)

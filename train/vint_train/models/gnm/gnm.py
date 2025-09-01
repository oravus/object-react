import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.base_model import BaseModel
from vint_train.goal_encoder import GoalEncoder


class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        **kwargs,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        self.kwargs = kwargs
        self.goal_type = self.kwargs.get("goal_type", "image")
        self.obs_type = self.kwargs.get("obs_type", "image")
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = goal_encoding_size

        if self.obs_type == "image":
            mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
            self.obs_mobilenet = mobilenet.features
            self.compress_observation = nn.Sequential(
                nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
                nn.ReLU(),
            )
        elif self.obs_type == "image_mask_enc":
            dims = self.kwargs["dims"]
            self.obs_mobilenet = GoalEncoder(
                obs_encoding_size,
                in_channels=dims * (1 + self.context_size),
                numLayers=2,
                preProjDims=None,
            )
        elif self.obs_type == "disabled":
            self.obs_mobilenet = None
            self.obs_encoding_size = 0
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")

        if self.goal_type == "image":
            stacked_mobilenet = MobileNetEncoder(
                num_images=2 + self.context_size
            )  # stack the goal and the current observation
            self.goal_mobilenet = stacked_mobilenet.features
            self.compress_goal = nn.Sequential(
                nn.Linear(stacked_mobilenet.last_channel, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.goal_encoding_size),
                nn.ReLU(),
            )
        elif self.goal_type == "image_mask_enc":
            self.goal_mobilenet = GoalEncoder(
                self.goal_encoding_size,
                in_channels=self.kwargs["dims"] + int(kwargs["use_mask_grad"]),
            )
        elif self.goal_type == "disabled":
            self.goal_mobilenet = None
            self.goal_encoding_size = 0
        else:
            raise ValueError(f"Unknown goal type: {self.goal_type}")

        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        if self.kwargs.get("predict_dists", True):
            self.dist_predictor = nn.Sequential(
                nn.Linear(32, 1),
            )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.obs_type == "image":
            obs_encoding = self.obs_mobilenet(obs_img)
            obs_encoding = self.flatten(obs_encoding)
            obs_encoding = self.compress_observation(obs_encoding)
        elif self.obs_type == "image_mask_enc":
            obs_encoding = self.obs_mobilenet(obs_img)
        elif self.obs_type == "disabled":
            obs_encoding = None

        if self.goal_type == "image":
            obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
            goal_encoding = self.goal_mobilenet(obs_goal_input)
            goal_encoding = self.flatten(goal_encoding)
            goal_encoding = self.compress_goal(goal_encoding)
        elif self.goal_type == "image_mask_enc":
            if self.kwargs["goal_uses_context"]:
                goal_encoding = [
                    self.goal_mobilenet(goal_img_)
                    for goal_img_ in goal_img.split(
                        goal_img.shape[1] // (self.context_size + 1), dim=1
                    )
                ]
                goal_encoding = torch.mean(torch.stack(goal_encoding), dim=0)
            else:
                goal_encoding = self.goal_mobilenet(goal_img)
        elif self.goal_type == "disabled":
            goal_encoding = None

        if obs_encoding is None:
            z = goal_encoding
        elif goal_encoding is None:
            z = obs_encoding
        else:
            z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        if self.kwargs.get("predict_dists", True):
            dist_pred = self.dist_predictor(z)
        else:
            dist_pred = None
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2].cpu(), dim=1).to(
            action_pred.device
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred

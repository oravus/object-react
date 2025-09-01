import torch
import torchvision.transforms.functional as TF

import tqdm

from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.training.train_utils import get_goal_image, get_obs_image
from vint_train.visualizing.action_utils import visualize_traj_pred
from vint_train.visualizing.visualize_utils import to_numpy


def visualize(
    config, model, dataloader, epoch, device, transform, use_tqdm=True, **kwargs
):
    model.eval()

    goal_type = kwargs.get("goal_type", "image")
    obs_type = kwargs.get("obs_type", "image")

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Visualizing epoch {epoch}",
    )

    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_image, viz_obs_image = get_obs_image(obs_image, obs_type, transform, device)

        goal_image, viz_goal_image = get_goal_image(
            goal_image, goal_type, transform, device, obs_image
        )

        model_outputs = model(obs_image, goal_image)
        dist_pred, action_pred = model_outputs

        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        visualize_traj_pred(
            to_numpy(viz_obs_image),
            (
                to_numpy(viz_goal_image)
                if goal_type != "disabled"
                else to_numpy(viz_obs_image)
            ),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            eval_type="visualize",
            normalized=config["normalize"],
            save_folder=config["project_folder"],
            epoch=epoch,
            num_images_preds=obs_image.shape[0],
            use_wandb=False,
            display=False,
        )
        break

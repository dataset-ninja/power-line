# https://universe.roboflow.com/ritsumeikan-university/electric-pole/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

import os

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)

# if sly.is_development():
# load_dotenv("local.env")
# load_dotenv(os.path.expanduser("~/supervisely.env"))

# api = sly.Api.from_env()
# team_id = sly.env.team_id()
# workspace_id = sly.env.workspace_id()


project_name = "Power Line"
dataset_path = "APP_DATA"
batch_size = 30
data_ext = ".bmp"
image_prefix_to_mask_prefix = {"VL_ORG": "VL_GT_", "IR_ORG": "IR_GT_"}


def fix_masks(image_np: np.ndarray) -> np.ndarray:
    lower_bound = np.array([200, 200, 200])
    upper_bound = np.array([255, 255, 255])
    condition_white = np.logical_and(
        np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
    )

    lower_bound = np.array([1, 1, 1])
    upper_bound = np.array([100, 100, 100])
    condition_black = np.logical_and(
        np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
    )

    image_np[np.where(condition_white)] = (255, 255, 255)
    image_np[np.where(condition_black)] = (0, 0, 0)

    return image_np


def create_ann(image_path, masks_path):
    labels = []

    image_np = sly.imaging.image.read(image_path)[:, :, 0]
    img_height = image_np.shape[0]
    img_wight = image_np.shape[1]

    image_name_parts = get_file_name_with_ext(image_path).split("_")
    mask_name = (
        image_prefix_to_mask_prefix[image_name_parts[0] + "_" + image_name_parts[1]]
        + image_name_parts[2]
    )
    mask_path = os.path.join(masks_path, mask_name)
    ann_np = sly.imaging.image.read(mask_path)[:, :, 0]

    obj_mask = ann_np != 0

    ret, curr_mask = connectedComponents(obj_mask.astype("uint8"), connectivity=8)
    for i in range(1, ret):
        obj_mask = curr_mask == i
        curr_bitmap = sly.Bitmap(obj_mask)
        if curr_bitmap.area > 200:
            curr_label = sly.Label(curr_bitmap, obj_class)
            labels.append(curr_label)

    return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


obj_class = sly.ObjClass("power line", sly.Bitmap)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    api.project.update_meta(project.id, meta.to_json())

    ds_to_image_anns_data = {
        "Visible Light (VL)": ("VL_Original (VL_ORG)", "VL_Ground_Truth (VL_GT)"),
        "Infrared (IR)": ("IR_Original (IR_ORG)", "IR_Ground_Truth (IR_GT)"),
    }

    for ds_name in os.listdir(dataset_path):
        curr_ds_path = os.path.join(dataset_path, ds_name)
        if dir_exists(curr_ds_path):
            images_path = os.path.join(curr_ds_path, ds_to_image_anns_data[ds_name][0])
            masks_path = os.path.join(curr_ds_path, ds_to_image_anns_data[ds_name][1])

            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            images_names = [
                im_name for im_name in os.listdir(images_path) if get_file_ext(im_name) == data_ext
            ]

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [
                    os.path.join(images_path, image_name) for image_name in images_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path, masks_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))
    return project

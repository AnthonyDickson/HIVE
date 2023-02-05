import argparse
import os
from zipfile import ZipFile

import torch
import wget
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo


def get_model_from_url(
    url: str, local_path: str, is_zip: bool = False, path_root: str = "checkpoints"
) -> str:
    local_path = os.path.join(path_root, local_path)
    if os.path.exists(local_path):
        print(f"Found cache {local_path}")
        return local_path

    # download
    local_path = local_path.rstrip(os.sep)
    download_path = local_path if not is_zip else f"{local_path}.zip"
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if os.path.isfile(download_path):
        print(f"Found cache {download_path}")
    else:
        print(f"Dowloading {url} to {download_path} ...")
        wget.download(url, download_path)

    if is_zip:
        print(f"Unziping {download_path} to {local_path}")
        with ZipFile(download_path, 'r') as f:
            f.extractall(local_path)
        os.remove(download_path)

    return local_path



def get_adabins_weights(basemodel_name='tf_efficientnet_b5_ap'):
    print('Loading base model {}...'.format(basemodel_name), end='')
    torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)


def get_mc_weights():
    get_model_from_url(
        "https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth",
        local_path="mc.pth",
        path_root=os.environ["WEIGHTS_PATH"]
    )


def get_detectron2_weights():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'

    # TODO: Save instance segmentation model name to global config file.
    model_config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_file)
    DefaultPredictor(cfg)  # The initialiser triggers the download of the weights.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mc', help='Download the weights for the "Mannequin Challenge" depth estimation model.', action='store_true')
    parser.add_argument('--adabins', help='Download the weights for the AdaBins depth estimation model.', action='store_true')

    args = parser.parse_args()

    get_detectron2_weights()

    if args.mc:
        get_mc_weights()

    if args.adabins:
        get_adabins_weights()

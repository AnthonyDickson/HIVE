from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

if __name__ == '__main__':
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'

    # TODO: Save instance segmentation model name to global config file.
    model_config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_file)
    predictor = DefaultPredictor(cfg)

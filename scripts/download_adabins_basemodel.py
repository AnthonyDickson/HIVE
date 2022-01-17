import torch

if __name__ == '__main__':
    basemodel_name = 'tf_efficientnet_b5_ap'

    print('Loading base model {}...'.format(basemodel_name), end='')
    basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
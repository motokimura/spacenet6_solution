import segmentation_models_pytorch as smp
import torch


def get_model(config):
    """
    """
    arch = config.MODEL.ARCHITECTURE
    backbone = config.MODEL.BACKBONE
    encoder_weights = config.MODEL.ENCODER_PRETRAINED_FROM
    n_classes = len(config.INPUT.CLASSES)
    activation = config.MODEL.ACTIVATION
    in_channels = config.MODEL.IN_CHANNELS

    if arch == 'unet':
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=activation,
            in_channels=in_channels
        )
    elif arch == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            classes=n_classes,
            activation=activation,
            in_channels=in_channels
        )
    else:
        raise ValueError()

    model = torch.nn.DataParallel(model)

    if config.MODEL.WEIGHT:
        # load weight from file
        model.load_state_dict(
            torch.load(
                config.MODEL.WEIGHT,
                map_location=torch.device('cpu')
            )
        )

    return model

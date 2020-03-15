import segmentation_models_pytorch as smp
import torch


def get_model(config):
    """
    """
    arch = config.MODEL.ARCHITECTURE
    backbone = config.MODEL.BACKBONE
    encoder_weights = config.MODEL.ENCODER_PRETRAINED_FROM
    in_channels = config.MODEL.IN_CHANNELS
    n_classes = len(config.INPUT.CLASSES)
    activation = config.MODEL.ACTIVATION

    if arch == 'unet':
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation
        )
    elif arch == 'fpn':
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            decoder_dropout=config.MODEL.FPN_DECODER_DROPOUT,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation
        )
    elif arch == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation
        )
    else:
        raise ValueError()

    model = torch.nn.DataParallel(model)

    if config.MODEL.WEIGHT and config.MODEL.WEIGHT != 'none':
        # load weight from file
        model.load_state_dict(
            torch.load(
                config.MODEL.WEIGHT,
                map_location=torch.device('cpu')
            )
        )

    return model

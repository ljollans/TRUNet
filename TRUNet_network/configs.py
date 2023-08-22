import ml_collections


def trunetConfigs(mode, img_size):
    config = ml_collections.ConfigDict()

    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.transformer_mlp_dim = 3072
    config.transformer_num_heads = 12
    config.transformer_num_layers = 12
    config.transformer_attention_dropout_rate = 0.0
    config.transformer_dropout_rate = 0.1
    config.classifier = 'seg'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 7
    config.n_skip = 3
    config.skip_channels = [512, 256, 64, 16]
    config.activation = 'softmax'
    config.patches = ml_collections.ConfigDict()
    config.patches.grid = None

    config.hidden_size = 768
    config.patches.size = 16


    config.patch_size = config.patches.size  # (results in 14 by 14 grid of patches for input size 224)

    config.hybrid = False
    if mode == '3d':
        config.patches.grid = (int(img_size / config.patches.size), int(img_size / config.patches.size),
                               int(img_size / config.patches.size))
        config.hybrid = True
    elif mode == '2d':
        config.patches.grid = (int(img_size / config.patches.size), int(img_size / config.patches.size))
        config.hybrid = True

    return config

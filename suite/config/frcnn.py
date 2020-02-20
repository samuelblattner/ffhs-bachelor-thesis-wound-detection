from keras_frcnn.config import Config


class FRCNNConfig(Config):

    def __init__(self):
        super(FRCNNConfig, self).__init__()
        # self.anchor_box_scales = [8, 16, 32]

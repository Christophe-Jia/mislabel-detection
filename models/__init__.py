from models.conv4 import Conv4
from models.densenet import DenseNet
from models.lenet import LeNet, LeNetMNIST
from models.resnet import ResNet
from models.vgg import VGG
from models.wide_resnet import WideResNet
from models.PreResNet import PreResNet

models = {
    "densenet": DenseNet,
    "resnet": ResNet,
    "wide_resnet": WideResNet,
    "vgg": VGG,
    "lenet": LeNet,
    "lenet_mnist": LeNetMNIST,
    "conv4": Conv4,
    "preresnet": PreResNet
}

__all__ = [
    "DenseNet",
    "WideResNet",
    "LeNet"
    "models",
]

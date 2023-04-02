import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models

""" This class is used to load pretrained models that are very popular for image recognition (resnet and vgg). """


class PretrainedModel(nn.Module):
    # define error messages
    ERR_TRUNC_MSG = ("{} currently only supports to be truncated "
                     "by its last {} FC layer(s). Please choose a value "
                     "between 0 and {}.")

    ERR_MODEL = "{} is currently not supported as a pretrained model."

    # define supported models
    SUPPORTED_MODEL_NAMES = ['resnet18', 'resnet34', 'resnet50',
                             'resnet101', 'resnet152',
                             'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                             'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']


    def __init__(self, model_name, layers_to_truncate=1):
        """Original Comment: Load the pretrained ResNet-152 and replace top fc layer."""
        super(PretrainedModel, self).__init__()

        # if the model is not one of the expected ones, raise error
        if model_name not in self.SUPPORTED_MODEL_NAMES:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        # set parameters according to what the chosen model requires
        if model_name.startswith('resnet'):
            self.input_size = (3, 224, 224)
            # number of fully connected layers in resnet architecture
            layer_size = 1
            max_trunc = 1
        elif model_name.startswith('vgg'):
            self.input_size = (3, 224, 224)
            # number of fully connected layers in VGG architecture
            layer_size = 3
            max_trunc = 3
        else:
            raise NotImplementedError(self.ERR_MODEL.format(model_name))

        # make sure that not more than the maximal number of layers to truncate are truncated
        if layers_to_truncate > max_trunc:
            raise ValueError(self.ERR_TRUNC_MSG.format(model_name, max_trunc, max_trunc))

        # get pretrained model from torchvision
        model = getattr(torchvision.models, model_name)(pretrained=True)

        if layers_to_truncate < 1:
            # use model without truncation
            self.pretrained_model = model
        else:
            # TODO (Original) Truncate last FC layer(s)
            # get the layers for each model
            if model_name.startswith('vgg'):
                layers = list(model.classifier.children())
            else:
                layers = list(model.children())

            trunc = self._get_num_truncated_layers(layers_to_truncate, layer_size)
            last_layer = layers[trunc]

            # Original Comment: Delete the last layer(s).
            modules = layers[:trunc]
            if model_name.startswith('vgg'):
                self.pretrained_model = model
                self.pretrained_model.classifier = nn.Sequential(*modules)
            else:
                self.pretrained_model = nn.Sequential(*modules)

        # Original Comment: Freeze all parameters of pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Original Comment: Switch model to eval mode (affects Dropout & BatchNorm)
        self.pretrained_model.eval()

        self.output_size = self._get_output_size()

    def _get_output_size(self):
        # get the output size by passing a dummy input through the model and inspecting the output
        dummy_input = Variable(torch.rand(1, *self.input_size))
        output = self(dummy_input)
        output_size = output.data.view(-1).size(0)
        return output_size

    def _get_num_truncated_layers(self, num_to_trunc, layer_size, initial_layer_size=1):

        # get the number of truncated numbers by adding layer_size until num_to_trunc is 0
        # TODO: I do not completely understand what layer_size is, do you have an idea?
        # TODO: It is the number of fully-connected layers at the end of the VGG and ResNet architectures, I think.
        num = 0
        if num_to_trunc > 0:
            num += initial_layer_size
            num_to_trunc -= 1
        while num_to_trunc > 0:
            num += layer_size
            num_to_trunc -= 1
        # return a negative number because this is used as an index until which the layers are preserved
        return -num

    def forward(self, images):
        """Original Comment: Extract the image feature vectors."""
        # TODO can it be deleted?
        features = self.pretrained_model(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        return features

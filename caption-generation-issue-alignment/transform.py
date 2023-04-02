import torchvision.transforms as transforms

def get_transform(net, train=True):
    # TODO: I do not understand what happens here exactly, but have tried to insert comments nevertheless (maybe you can improve them?)
    """
    Gets the transform of the vision model (implemented for VGG models only)

    :param net: the vision model
    :param train: whether we are in training, as a boolean
    :return: the transform
    """

    # if no vision model is given, nothing is returned
    if net is None:
        transform = None
    # if a vgg model is given, get the transform, which is different for training vs. the other conditions
    # vgg is the only possible model we can use
    elif net.startswith('vgg'):
        if train:
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
        else:
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
    # if an unknown vision model is given, raise an error
    else:
        raise NotImplementedError("{} is missing a data transform "
                                  "implementation".format(net))

    return transform

import torchvision.transforms as transforms

def get_transform(net, train=True):
    """
    Defines transformation pipeline for the image so that it can be presented in tensor form. Finally, the tensor is
    normalized
    """
    if net is None:
        transform = None

    elif net.startswith('vgg'): # the only possible model we can use is this class for is vgg (not ResNet)
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
    else: # No transforms implemented for ResNet
        raise NotImplementedError("{} is missing a data transform "
                                  "implementation".format(net))

    return transform
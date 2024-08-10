import torchvision.transforms as transforms

def get_transform(augment: bool = False) -> transforms.Compose:
    t_list = [transforms.ToTensor(), ]
    if augment:
        t_list.append(transforms.RandomAffine(degrees=15,
                                              translate=(0.1, 0.1),
                                              shear=15))
    t_list.extend([transforms.Resize((480, 270)),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    t = transforms.Compose(t_list)

    return t

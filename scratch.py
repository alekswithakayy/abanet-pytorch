from models.dilated_resnet import dilated_resnext107_32x8d
resnext = dilated_resnext107_32x8d()
for name, param in resnext.named_parameters():
    print(name)

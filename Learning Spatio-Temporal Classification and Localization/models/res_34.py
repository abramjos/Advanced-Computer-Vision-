import torch.nn.init as init
import os


from models.resnet import resnet34
import torch.nn as nn
import torch

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)


def copy_weight_bias(l1,l2,basic_block=False):
    if basic_block == True:
        l1.conv1.weight = l2.conv1.weight
        l1.bn1.weight = l2.bn1.weight
        l1.bn1.bias = l2.bn1.bias

        l1.conv2.weight = l2.conv2.weight
        l1.bn2.weight = l2.bn2.weight
        l1.bn2.bias = l2.bn2.bias
    else:
        l1.weight = l2.weight
        if isinstance(l1,(nn.BatchNorm3d,nn.Linear)):
            l1.bias = l2.bias

    return(True)


def get_res_34_features(pretrained_path = '../models/res_34_features.pth', fix_block = False, fix_weights = False):

    #pretrained_path = '../models/res_34_features.pth'
    model_34,model_34_new = resnet34(num_classes=101, shortcut_type='A',sample_duration=16,sample_size = 112)
    model_34.load_state_dict(torch.load(pretrained_path))

    print('Copying feature extractor weights to new network...')
    for i,j in zip(model_34.children(),model_34_new.children()):
        # import ipdb;ipdb.set_trace()
        if i.__str__() == j.__str__():
            if isinstance(i,nn.Sequential):
                for b1,b2 in zip(i,j):                    
                    copy_weight_bias(b2,b1,basic_block=True)

            else:
                if not isinstance(i,(nn.ReLU,nn.MaxPool3d,nn.AvgPool3d)):
                    copy_weight_bias(j,i)
    if fix_weights == True:
        for param1,param2 in  zip(model_34.parameters(),model_34_new.parameters()):
            param2.requires_grad = False
    return(model_34_new,model_34)

def model(**kwargs):
    model_34,_ = get_res_34_features(**kwargs)
    model_34 = model_34.cuda() 
    return(model_34)

if __name__ == '__main__':
    
    model_34,resnet_34 = get_res_34_features()
    model_34 = model_34.cuda() 
    # c3d.apply(weights_init)
    inx =torch.rand(( 16, 3, 16, 112, 112 )).to('cuda')
    model_34.eval()
    ans = model_34(inx)
    import ipdb;ipdb.set_trace()
    print(ans)


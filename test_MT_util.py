import math
import numpy as np
# from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import os
from utils.util import crf_refine
import torchvision.utils as vutils
import pdb

def test_all_case(net, image_list, num_classes=1, save_result=True, test_save_path=None, trans_scale=416, GT_access=True):
    normal = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # SBU
    # normal = transforms.Normalize([0.517, 0.514, 0.492], [0.186, 0.173, 0.181]) # ISTD
    # normal = transforms.Normalize([0.723, 0.616, 0.569], [0.169, 0.177, 0.197]) # ISIC2017
    img_transform = transforms.Compose([
        transforms.Resize((trans_scale, trans_scale)),
        transforms.ToTensor(),
        normal,
    ])
    to_pil = transforms.ToPILImage()
    TP, TN, Np, Nn = 0, 0, 0, 0
    ber_mean = 0
    sum_ber_mean = 0.0
    iter = 0
    net.eval()

    for (img_path, target_path) in tqdm(image_list):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_var = img_transform(img).unsqueeze(0).cuda()
        # res = net(img_var).softmax(dim=1) # shape(1, 2, 416, 416)
        # prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()[1, :, :])))
        # up_edge, up_shadow, up_subitizing, up_shadow_final = net(img_var) #subitizing

        # up_edge, up_sal, up_subitizing, up_sal_final
        with torch.no_grad():
            prediction = net(img_var)  # subiziting
        pdb.set_trace()

        # len(up_edge) -- 1
        # (Pdb) up_edge[0].size() -- torch.Size([1, 1, 416, 416])
        # (Pdb) len(up_shadow) -- 4
        # (Pdb) len(up_shadow), up_shadow[0].size(), up_shadow[1].size(), up_shadow[2].size(), up_shadow[3].size()
        # [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416]
        # (Pdb) len(up_shadow_final), up_shadow_final[0].size()
        # (1, torch.Size([1]))
        # len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size()
        # (5, [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416], [1, 1, 416, 416])


        # up_sal_final = net(img_var)
        # up_shadow_final, up_subitizing = net(img_var)  # subiziting
        # res = torch.sigmoid(up_shadow_final[-1])
        ''' # original size crf
        prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
        # prediction = np.uint8(prediction>=100)*255
        prediction = crf_refine(np.array(img.convert('RGB')), prediction)
        '''
        # resized image crf
        prediction = np.array(to_pil(prediction.squeeze(0).cpu()))
        prediction = np.uint8(prediction>=90)*255 # trick just for SBU
        # prediction = crf_refine(np.array(img.convert('RGB').resize((trans_scale, trans_scale))), prediction)
        prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))
        
        # cal metric
        if GT_access:
            target = np.array(Image.open(target_path).convert('L'))
            TP_single, TN_single, Np_single, Nn_single, Union = cal_acc(prediction, target)
            '''
            Calculate BER 
            '''
            TP = TP + TP_single
            TN = TN + TN_single
            Np = Np + Np_single
            Nn = Nn + Nn_single
            ber_shadow = (1 - TP / Np) * 100
            ber_unshadow = (1 - TN / Nn) * 100
            ber_mean = 0.5 * (2 - TP / Np - TN / Nn) * 100
            sum_ber_mean += ber_mean
            # print("{} ber is {}, shadow_ber is {}, unshadow ber is {}".format(img_path, ber_mean, ber_shadow, ber_unshadow))
        '''
        Save prediction
        '''
        if save_result:
            output_filename = os.path.join(test_save_path, img_name[:-4]+'.png')
            mask_image = Image.fromarray(prediction)
            F = transforms.ToTensor()(img)
            M = 1.0 - transforms.ToTensor()(mask_image) * 0.5
            output_image = transforms.ToPILImage()(torch.cat((F,M), dim=0))


            width, height = img.size
            result = Image.new(output_image.mode, (width * 2, height))
            result.paste(img, box=(0, 0))
            result.paste(output_image, box=(width, 0))
            result.save(output_filename, "PNG")

    return sum_ber_mean

def cal_acc(prediction, label, thr = 128):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union


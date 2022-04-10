"""Image Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F

import redos
import todos

from . import deshadow
import pdb


def model_forward(model, device, input_tensor):
    H, W = input_tensor.size(2), input_tensor.size(3)
    input_tensor_new = input_tensor.clone()
    input_tensor_new[0] = todos.data.normal_tensor(input_tensor_new[0], 
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return todos.model.forward(model, device, input_tensor_new)

    # SHADOW_MODEL_HEIGHT = 416
    # SHADOW_MODEL_WIDTH = 416
    # if H == SHADOW_MODEL_HEIGHT and W == SHADOW_MODEL_WIDTH:
    #     return todos.model.forward(model, device, input_tensor_new)

    # # else, we need resize
    # input_tensor_new = F.interpolate(
    #     input_tensor_new, size=(SHADOW_MODEL_HEIGHT, SHADOW_MODEL_WIDTH), mode="bilinear", align_corners=False
    # )
    # output_tensor = todos.model.forward(model, device, input_tensor_new)
    # output_tensor = F.interpolate(output_tensor, size=(H, W), mode="bilinear", align_corners=False)
    # return output_tensor


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.deshadow(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    device = todos.model.get_device()
    model = deshadow.get_model()
    model = model.to(device)

    def do_service(input_file, output_file, targ):
        print(f"  deshadow {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_deshadow", do_service, host, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    device = todos.model.get_device()
    model = deshadow.get_model()
    model = model.to(device)

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        predict_tensor = model_forward(model, device, input_tensor)
        # shadow_tensor = (predict_tensor >= 90.0/255.0).float()

        mask_tensor = 1.0 - predict_tensor * 0.5
        shadow_tensor = torch.cat((input_tensor, mask_tensor.cpu()), dim=1)

        # orig input
        orig_tensor = todos.data.load_rgba_tensor(filename)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        output_file = output_file.replace(".jpg", ".png")

        todos.data.save_tensor([orig_tensor, shadow_tensor], output_file)

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
from tqdm import tqdm
import torch

import redos
import todos

from . import deshadow
import pdb

DESHADOW_ZEROPAD_TIMES = 2


def model_forward(model, device, input_tensor, multi_times):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0:H, 0:W]


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
    print(f"Running on {device} ...")

    def do_service(input_file, output_file, targ):
        print(f"  deshadow {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, DESHADOW_ZEROPAD_TIMES)
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
    print(f"Running on {device} ...")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        predict_tensor = model_forward(model, device, input_tensor, DESHADOW_ZEROPAD_TIMES)
        # shadow_tensor = (predict_tensor >= 90.0/255.0).float()

        mask_tensor = 1.0 - predict_tensor * 0.5
        shadow_tensor = torch.cat((input_tensor, mask_tensor.cpu()), dim=1)

        # orig input
        orig_tensor = todos.data.load_rgba_tensor(filename)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        output_file = output_file.replace(".jpg", ".png")

        todos.data.save_tensor([orig_tensor, shadow_tensor], output_file)

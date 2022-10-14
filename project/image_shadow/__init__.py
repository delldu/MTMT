"""Image Shadow Package."""  # coding=utf-8
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

from . import shadow
import pdb


def load_weight(model, path):
    """Load model."""

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))

    target_state_dict = model.state_dict()
    for n in target_state_dict.keys():
        n2 = n.replace("seqlist", "")
        if n2 in state_dict.keys():
            p2 = state_dict[n2]
            target_state_dict[n].copy_(p2)
        else:
            raise KeyError(n)


def get_model():
    """Create model."""

    model_path = "models/image_shadow.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = shadow.ShadowModel()

    load_weight(model, checkpoint)
    model.eval()

    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_shadow.torch"):
        model.save("output/image_shadow.torch")

    return model


def model_forward(model, device, input_tensor, multi_times=2):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    output_tensor = todos.model.forward(model, device, input_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.shadow(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    device = todos.model.get_device()
    model = get_model()
    model = model.to(device)
    print(f"Running on {device} ...")

    def do_service(input_file, output_file, targ):
        print(f"  shadow {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_shadow", do_service, host, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    device = todos.model.get_device()
    model = get_model()
    model = model.to(device)
    print(f"Running on {device} ...")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_tensor(filename)
        output_tensor = model_forward(model, device, input_tensor)

        # orig input
        orig_tensor = todos.data.load_rgba_tensor(filename)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        output_file = output_file.replace(".jpg", ".png")

        todos.data.save_tensor([orig_tensor, output_tensor], output_file)

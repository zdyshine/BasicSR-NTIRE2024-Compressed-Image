import os, glob
import numpy as np
import torch
import cv2
from PIL import Image
from copy import deepcopy
import argparse

def load_checkpoint_basicsr(model, model_path, strict=True):
    # load checkpoint
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # param_key = 'params'
    param_key = 'params_ema'
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]
    # # ------------ load model
    print('==================== model path ====================')
    print(model_path)
    print('=========== param_key: ', param_key)

    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=strict)


def test_selfensemble(net, lq):
    def _transform(v, op):
        # if self.precision != 'single': v = v.float()
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to('cuda')
        # if self.precision == 'half': ret = ret.half()

        return ret

    # prepare augmented data
    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    with torch.no_grad():
        out_list = [net(aug) for aug in lq_list]

    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')
    output = torch.cat(out_list, dim=0)
    output = output.mean(dim=0, keepdim=True)

    return output

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # set path
    # ----------------------------------------
    test_path = args.test_path

    ckpt, ext = os.path.splitext(os.path.basename(args.checkpoint))

    result_name = f'SCUNet_{ckpt}'
    if args.enhance:
        result_name = result_name + '_enhance'
    E_path = os.path.join('results', result_name)  # E_path, for Estimated images
    os.makedirs(E_path, exist_ok=True)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from archs.scunet_arch import SCUNet as net
    model = net()
    load_checkpoint_basicsr(model, args.checkpoint, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    test_paths = glob.glob(os.path.join(test_path, '*.jpg'))

    runtime = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for idx, img in enumerate(test_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        print('{:->4d}--> {:>10s}'.format(idx + 1, img_name + ext), end=' ')
        img_L = np.array(Image.open(img).convert("RGB")) # RGB, 0~255
        img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        start.record()
        img_E = model(img_L)
        if args.enhance:
            img_E += torch.flip(model(torch.flip(img_L, dims=[2])), dims=[2])
            img_E += torch.flip(model(torch.flip(img_L, dims=[3])), dims=[3])
            img_E += torch.flip(model(torch.flip(img_L, dims=[2, 3])), dims=[2, 3])
            img_E = img_E / 4

        end.record()
        torch.cuda.synchronize()
        time_ = start.elapsed_time(end)
        runtime.append(time_)  # milliseconds
        print('time ', time_, 'ms')
        img_E = img_E.data.squeeze().float().cpu().numpy()
        img_E = np.transpose(img_E, (1, 2, 0))
        img_E = np.uint8((img_E.clip(0, 1)*255.).round())
        cv2.imwrite(os.path.join(E_path, img_name + '.png'), img_E[:, :, [2, 1, 0]])

    ave_runtime = sum(runtime) / len(test_paths)  # / 1000.0
    print('ave_runtime: ', ave_runtime, 'ms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specify dirs
    parser.add_argument("--test_path", type=str, default='/JPEG_Image/validation_JPEG')
    parser.add_argument("--checkpoint", type=str, default='../experiments/JPEG_002_SCUNet/models/net_g_ema_198000.pth') # best
    parser.add_argument("--enhance", action="store_true")
    args = parser.parse_args()
    main(args)

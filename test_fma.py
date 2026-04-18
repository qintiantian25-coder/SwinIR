import os
import argparse
import re
import csv
import cv2
import numpy as np
import torch
import glob

from models.network_swinir import SwinIR as Net
from utils import util_calculate_psnr_ssim as util


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def build_model(device, in_chans=3):
    model = Net(upscale=1, in_chans=in_chans, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    return model.to(device)


def to_rgb_tensor_gray(img_gray):
    # img_gray: HxW uint8 or float in [0,255] -> produce 3xHxW float32 in [0,1]
    if img_gray.dtype != np.float32:
        img = img_gray.astype(np.float32)
    else:
        img = img_gray
    if img.max() > 1.0:
        img = img / 255.0
    img3 = np.stack([img, img, img], axis=2)
    # transpose to CHW
    img3 = img3.transpose(2, 0, 1)
    return img3


def rgb_to_gray_from_tensor(out_np):
    # out_np: C x H x W in [0,1]
    if out_np.ndim == 3:
        r = out_np[0]
        g = out_np[1]
        b = out_np[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = np.clip(gray * 255.0, 0, 255).round().astype(np.uint8)
        return gray
    else:
        # single channel
        gray = np.clip(out_np[0] * 255.0, 0, 255).round().astype(np.uint8)
        return gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='dataset root')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, default='results/fma_test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_mask_csv', type=str, default=None, help='csv of blind pixel coords')
    parser.add_argument('--image_border', type=int, default=0, help='crop border for PSNR/SSIM')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_triple = os.path.join(args.save_dir, 'triple_comparison')
    save_pure = os.path.join(args.save_dir, 'test')
    os.makedirs(save_triple, exist_ok=True)
    os.makedirs(save_pure, exist_ok=True)

    # build model and load checkpoint
    model = build_model(device, in_chans=3)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # ckpt may contain 'model' or full state dict
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()

    # gather gt map
    gt_root = os.path.join(args.data_root, 'test_sharp')
    gt_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                gt_map[f] = os.path.join(root, f)

    # gather input images (test_blur)
    input_root = os.path.join(args.data_root, 'test_blur')
    input_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith('.png'):
                input_files.append(os.path.join(root, f))
    input_files = sorted(input_files, key=natural_sort_key)

    # load blind coords (CSV) if provided
    blind_coords = load_blind_coords(args.test_mask_csv) if args.test_mask_csv else None

    per_image_logs = []
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0

    print(f'===> Start testing {len(input_files)} images...')
    with torch.no_grad():
        for idx, in_path in enumerate(input_files):
            name = os.path.basename(in_path)
            # load input (gray)
            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if in_img is None:
                print('WARN: failed to load', in_path)
                continue
            H, W = in_img.shape[:2]

            # prepare input tensor (3 channels)
            inp_np = to_rgb_tensor_gray(in_img)
            inp_tensor = torch.from_numpy(inp_np).float().unsqueeze(0).to(device)

            out = model(inp_tensor)
            out = out.clamp(0, 1).cpu().numpy()[0]  # C,H,W

            out_gray = rgb_to_gray_from_tensor(out)

            # save output and triple
            # load gt
            gt_path = gt_map.get(name)
            if gt_path and os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is None:
                    print('WARN: failed to load gt for', name)
                    continue
                # align sizes
                if out_gray.shape != gt_img.shape:
                    out_gray = cv2.resize(out_gray, (gt_img.shape[1], gt_img.shape[0]))
                if in_img.shape != gt_img.shape:
                    in_resized = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                else:
                    in_resized = in_img

                # save triple visualization (input|output|gt)
                triple = np.concatenate([in_resized, out_gray, gt_img], axis=1)
                cv2.imwrite(os.path.join(save_triple, f'triple_{name}'), triple)
                cv2.imwrite(os.path.join(save_pure, name), out_gray)

                # compute full-frame metrics
                full_psnr = util.calculate_psnr(out_gray, gt_img, crop_border=args.image_border)
                full_ssim = util.calculate_ssim(out_gray, gt_img, crop_border=args.image_border)

                row = {
                    'image': name,
                    'psnr': float(full_psnr),
                    'ssim': float(full_ssim),
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'blind_mae_input': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                    'blind_count': 0
                }

                # blind metrics via coords CSV (if provided)
                if blind_coords is not None and blind_coords.size > 0:
                    h, w = gt_img.shape[:2]
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]
                        gt_vals = gt_img[y, x].astype(np.float64)
                        out_vals = out_gray[y, x].astype(np.float64)
                        err = out_vals - gt_vals
                        blind_abs = np.abs(err)
                        blind_sq = err ** 2

                        blind_abs_sum += float(blind_abs.sum())
                        blind_sq_sum += float(blind_sq.sum())
                        blind_pix_sum += int(len(err))

                        # input blind error
                        in_vals = in_resized[y, x].astype(np.float64)
                        in_err = in_vals - gt_vals
                        in_abs = np.abs(in_err)
                        in_sq = in_err ** 2
                        blind_abs_in_sum += float(in_abs.sum())
                        blind_sq_in_sum += float(in_sq.sum())

                        row.update({
                            'blind_mae': float(blind_abs.mean()),
                            'blind_rmse': float(np.sqrt(blind_sq.mean())),
                            'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                            'blind_mae_input': float(in_abs.mean()),
                            'blind_count': int(len(err))
                        })
                        if row['blind_mae_input'] is not None:
                            row['blind_mae_gain_abs'] = row['blind_mae_input'] - row['blind_mae']
                            row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (row['blind_mae_input'] + 1e-12)

                per_image_logs.append(row)
            else:
                # no gt, just save output
                cv2.imwrite(os.path.join(save_pure, name), out_gray)

            if (idx + 1) % 10 == 0:
                print(f'Processed {idx+1}/{len(input_files)}')

    # write per-image CSV
    csv_path = os.path.join(args.save_dir, 'per_image_metrics.csv')
    if len(per_image_logs) > 0:
        keys = list(per_image_logs[0].keys())
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in per_image_logs:
                writer.writerow(r)
        print('Saved per-image metrics to', csv_path)

    # aggregate blind metrics
    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = float(np.sqrt(blind_mse))
        blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))

        print('===> Blind-Pixel Focused Metrics')
        print(f'BlindCount(total sampled): {blind_pix_sum}')
        print(f'Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}')

        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse_in, 1e-12)))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(f'Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)')


if __name__ == '__main__':
    main()

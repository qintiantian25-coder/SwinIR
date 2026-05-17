import os
import argparse
import re
import csv
import cv2
import numpy as np
import torch
import glob
from collections import defaultdict

from models.network_swinir import SwinIR as Net
from utils import util_calculate_psnr_ssim as util


class TestReport:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border
        self.total_rgb_psnr = []
        self.total_ssim = []

    def update_metric(self, gt_img, out_img, img_name=None):
        self.total_rgb_psnr.append(float(util.calculate_psnr(out_img, gt_img, crop_border=self.crop_border)))
        self.total_ssim.append(float(util.calculate_ssim(out_img, gt_img, crop_border=self.crop_border)))

    def print_final_result(self):
        if len(self.total_rgb_psnr) == 0:
            print('No valid images were evaluated.')
            return
        print(f'Average PSNR: {np.mean(self.total_rgb_psnr):.4f} dB')
        print(f'Average SSIM: {np.mean(self.total_ssim):.4f}')


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


def resolve_csv_path(csv_path, data_root):
    if not csv_path:
        return None
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path
    if os.path.exists(csv_path):
        return csv_path
    if data_root:
        candidate = os.path.join(data_root, csv_path)
        if os.path.exists(candidate):
            return candidate
    return csv_path


def build_model(device, in_chans=3):
    model = Net(upscale=1, in_chans=in_chans, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    return model.to(device)


def to_rgb_tensor_gray(img_gray, in_chans=3):
    # img_gray: HxW uint8 or float in [0,255]
    # if in_chans==3 -> produce 3xHxW float32 in [0,1]
    # if in_chans==1 -> produce 1xHxW float32 in [0,1]
    if img_gray.dtype != np.float32:
        img = img_gray.astype(np.float32)
    else:
        img = img_gray
    if img.max() > 1.0:
        img = img / 255.0
    if in_chans == 3:
        img3 = np.stack([img, img, img], axis=2)
        img3 = img3.transpose(2, 0, 1)
        return img3
    else:
        img1 = img[np.newaxis, ...]
        return img1


def rgb_to_gray_from_tensor(out_np):
    # out_np: C x H x W in [0,1]
    # if 3-channel output, convert to gray; if single-channel, scale directly
    if out_np.ndim == 3 and out_np.shape[0] == 3:
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


def get_group_name(rel_path):
    parts = os.path.normpath(rel_path).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return 'root'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='dataset root')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint (.pth)')
    parser.add_argument('--in_chans', type=int, default=3, help='input channels used to build the model (1 or 3)')
    parser.add_argument('--save_dir', type=str, default='results/fma_test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_mask_csv', type=str, default=None, help='csv of blind pixel coords')
    parser.add_argument('--image_border', type=int, default=0, help='crop border for PSNR/SSIM')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_triple = os.path.join(args.save_dir, 'triple_comparison')
    save_pure = os.path.join(args.save_dir, 'test')
    save_blind_dir = os.path.join(args.save_dir, 'blind_eval')
    os.makedirs(save_triple, exist_ok=True)
    os.makedirs(save_pure, exist_ok=True)
    os.makedirs(save_blind_dir, exist_ok=True)

    # build model and load checkpoint
    model = build_model(device, in_chans=args.in_chans)
    # Try normal safe loading first; if it fails (new PyTorch weights_only restrictions),
    # fall back to allowing the needed numpy global and load with weights_only=False.
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
    except Exception as e:
        try:
            # allowlist numpy scalar type used by some older checkpoints
            try:
                torch.serialization.add_safe_globals([np._core.multiarray.scalar])
            except Exception:
                pass
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        except Exception:
            raise

    # ckpt may contain 'model' or be a plain state dict
    state = ckpt.get('model', ckpt)
    # if checkpoint was saved from DataParallel, strip the 'module.' prefix
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state[new_k] = v
        state = new_state
    model.load_state_dict(state)
    model.eval()

    # gather gt map (use relative path keys to avoid basename collisions across subfolders)
    gt_root = os.path.join(args.data_root, 'test_sharp')
    gt_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                full = os.path.join(root, f)
                rel = os.path.normpath(os.path.relpath(full, gt_root))
                gt_map[rel] = full
                # also register basename as a fallback if not already present
                if f not in gt_map:
                    gt_map[f] = full

    # gather input images (test_blur)
    input_root = os.path.join(args.data_root, 'test_blur')
    input_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith('.png'):
                input_files.append(os.path.join(root, f))
    input_files = sorted(input_files, key=natural_sort_key)

    grouped_inputs = defaultdict(list)
    for in_path in input_files:
        rel_in = os.path.normpath(os.path.relpath(in_path, input_root))
        grouped_inputs[get_group_name(rel_in)].append(in_path)

    # load blind coords (CSV) if provided
    resolved_test_mask_csv = resolve_csv_path(args.test_mask_csv, args.data_root)
    blind_coords = load_blind_coords(resolved_test_mask_csv) if resolved_test_mask_csv else None
    if args.test_mask_csv and blind_coords is None:
        print(f'WARN: blind coords CSV not loaded: {resolved_test_mask_csv}')
        print('WARN: blind metrics will stay empty until the CSV path is correct and the file has x,y columns.')
    elif blind_coords is not None:
        print(f'Loaded blind coords from: {resolved_test_mask_csv} ({len(blind_coords)} unique points)')

    report = TestReport(crop_border=args.image_border)
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []
    per_group_logs = defaultdict(list)

    print(f'===> 开始定量打分，准备比对 {len(input_files)} 张图片...')
    with torch.no_grad():
        for group_name, group_files in grouped_inputs.items():
            print(f'===> Processing group {group_name} ({len(group_files)} images) ...')
            group_rows = []

            for idx, in_path in enumerate(group_files):
                name = os.path.basename(in_path)
                rel_in = os.path.normpath(os.path.relpath(in_path, input_root))

                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                if in_img is None:
                    print('WARN: failed to load', in_path)
                    continue

                inp_np = to_rgb_tensor_gray(in_img, in_chans=args.in_chans)
                inp_tensor = torch.from_numpy(inp_np).float().unsqueeze(0).to(device)

                out = model(inp_tensor)
                out = out.clamp(0, 1).cpu().numpy()[0]
                out_gray = rgb_to_gray_from_tensor(out)

                gt_path = gt_map.get(rel_in, gt_map.get(name))
                if gt_path and os.path.exists(gt_path):
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_img is None:
                        print('WARN: failed to load gt for', name)
                        continue

                    if out_gray.shape != gt_img.shape:
                        out_gray = cv2.resize(out_gray, (gt_img.shape[1], gt_img.shape[0]))
                    if in_img.shape != gt_img.shape:
                        in_resized = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                    else:
                        in_resized = in_img

                    triple = np.concatenate([in_resized, out_gray, gt_img], axis=1)
                    cv2.imwrite(os.path.join(save_triple, f'triple_{name}'), triple)
                    cv2.imwrite(os.path.join(save_pure, name), out_gray)

                    report.update_metric(gt_img, out_gray, name)
                    full_psnr = float(report.total_rgb_psnr[-1])
                    full_ssim = float(report.total_ssim[-1])

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
                        else:
                            print(f'WARN: no valid blind coords inside image bounds for {name}')

                    per_image_logs.append(row)
                    group_rows.append(row)
                else:
                    cv2.imwrite(os.path.join(save_pure, name), out_gray)

                if (idx + 1) % 10 == 0:
                    print(f'Processed {idx+1}/{len(group_files)} in group {group_name}')

            if len(group_rows) > 0:
                group_dir = os.path.join(save_blind_dir, group_name)
                os.makedirs(group_dir, exist_ok=True)
                group_csv = os.path.join(group_dir, 'test_blind_metrics.csv')
                keys = [
                    'image', 'psnr', 'ssim',
                    'blind_mae', 'blind_rmse', 'blind_psnr',
                    'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
                ]
                with open(group_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for row in group_rows:
                        writer.writerow(row)
                print(f'Per-image test metrics saved to: {group_csv}')

            per_group_logs[group_name].extend(group_rows)

    report.print_final_result()

    # write per-image CSV
    csv_path = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = [
            'image', 'psnr', 'ssim',
            'blind_mae', 'blind_rmse', 'blind_psnr',
            'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count'
        ]
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in per_image_logs:
                writer.writerow(r)
        print('Per-image test metrics saved to:', csv_path)

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

        if len(per_image_logs) > 0:
            print(f'Blind per-image metrics saved to: {csv_path}')


if __name__ == '__main__':
    main()

import os
import sys
import time
import json
import argparse
import configparser
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from utils.util_calculate_psnr_ssim import calculate_psnr

from datasets.fma_dataset import FMADataset
from models.network_swinir import SwinIR as Net


def load_config(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if path.lower().endswith('.yml') or path.lower().endswith('.yaml'):
        try:
            import yaml

            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            raise RuntimeError('PyYAML is required to load YAML config files. Install with `pip install pyyaml`.')

    # fallback to INI-like parser
    cfg = configparser.ConfigParser()
    cfg.read(path)
    out = {}
    for section in cfg.sections():
        out[section] = {}
        for k, v in cfg.items(section):
            # try to cast types: int, float, bool, otherwise keep string
            lv = v
            if v.lower() in ('true', 'false'):
                lv = cfg.getboolean(section, k)
            else:
                try:
                    if '.' in v:
                        lv = cfg.getfloat(section, k)
                    else:
                        lv = cfg.getint(section, k)
                except Exception:
                    lv = v
            out[section][k] = lv
    return out


def build_model(device, in_chans=3, img_size=128):
    model = Net(upscale=1, in_chans=in_chans, img_size=img_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    return model.to(device)


def train_from_config(cfg: Dict[str, Any]):
    ds_cfg = cfg.get('dataset', {})
    tr_cfg = cfg.get('train', {})

    data_root = ds_cfg.get('data_root')
    if data_root is None:
        raise ValueError('dataset.data_root must be specified in config')
    split = ds_cfg.get('split', 'train')
    gray_mode = ds_cfg.get('gray_mode', 'replicate')

    patch_size = int(tr_cfg.get('patch_size', 128))
    batch_size = int(tr_cfg.get('batch_size', 8))
    epochs = int(tr_cfg.get('epochs', 10))
    lr = float(tr_cfg.get('lr', 2e-4))
    alpha = float(tr_cfg.get('alpha', 5.0))
    mask_loss_weight = float(tr_cfg.get('mask_loss_weight', 0.0))
    val_freq = int(tr_cfg.get('val_freq', 5))
    num_workers = int(tr_cfg.get('num_workers', 4))
    save_dir = tr_cfg.get('save_dir', 'experiments/fma_checkpoints')
    device_str = tr_cfg.get('device', 'cuda')
    in_chans = int(tr_cfg.get('in_chans', 1))
    use_mask = bool(tr_cfg.get('use_mask', False))

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and 'cuda' in device_str else 'cpu')

    # 日志文件（保存在 save_dir）
    train_log_path = os.path.join(save_dir, 'train.log')
    val_log_path = os.path.join(save_dir, 'val.log')
    # 如果日志文件不存在，写入表头
    if not os.path.exists(train_log_path):
        with open(train_log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,avg_train_loss,time_s\n')
    # val log includes a flag whether this validation saved a new best model
    if not os.path.exists(val_log_path):
        with open(val_log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,avg_val_loss,avg_psnr,best_saved\n')

    ds = FMADataset(data_root, split=split, patch_size=patch_size, augment=True, gray_mode=gray_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 可选的验证集（使用 data_root 下的 val_split_blur 文件夹）
    val_split = ds_cfg.get('val_split', 'val')
    val_loader = None
    val_blur_dir = os.path.join(data_root, f"{val_split}_blur")
    if os.path.isdir(val_blur_dir):
        val_ds = FMADataset(data_root, split=val_split, patch_size=patch_size, augment=False, gray_mode=gray_mode)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers//2), pin_memory=True)
        print(f'Validation loader created with split="{val_split}" and {len(val_ds)} items')

    model = build_model(device, in_chans=in_chans, img_size=patch_size)
    # 如果有多张 GPU，则使用 DataParallel 包装模型以利用多卡并减小单卡显存压力
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and 'cuda' in device_str:
            model = torch.nn.DataParallel(model)
            print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
    except Exception:
        # 任何包装失败都不应阻止训练；保持原模型
        pass
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_best = os.path.join(save_dir, 'best_model.pth')
    best_loss = float('inf')
    if os.path.exists(ckpt_best):
        try:
            prev = torch.load(ckpt_best, map_location='cpu')
            if isinstance(prev, dict) and 'best_loss' in prev:
                best_loss = float(prev['best_loss'])
                print(f'Loaded existing best_loss={best_loss} from {ckpt_best}')
        except Exception:
            print('Warning: failed to load existing best model; starting with best_loss=inf')

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(loader):
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)

            if inp.dim() == 3:
                inp = inp.unsqueeze(0)
                gt = gt.unsqueeze(0)

            pred = model(inp)

            if pred.shape != gt.shape:
                if pred.shape[1] == 1 and gt.shape[1] == 3:
                    gt = gt[:, 0:1, ...]
                elif pred.shape[1] == 3 and gt.shape[1] == 1:
                    gt = gt.repeat(1, 3, 1, 1)

            loss_map = torch.abs(pred - gt)
            # 基础损失（L1）——不使用任何 mask 加权或额外的 mask 损失项
            base_loss = loss_map.mean()
            loss = base_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} Step {global_step} Loss {loss.item():.6f}')

        t1 = time.time()
        avg_loss = epoch_loss / len(loader)
        epoch_time = t1 - t0
        print(f'Epoch {epoch+1} finished. Avg loss: {avg_loss:.6f}. Time: {epoch_time:.1f}s')
        # 写训练日志
        try:
            with open(train_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{epoch_time:.1f}\n")
        except Exception:
            pass

        # 仅在存在验证集且为验证轮（每 val_freq 轮）时，根据验证平均损失决定是否保存最优模型
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            model.eval()
            val_loss_acc = 0.0
            val_steps = 0
            psnr_acc = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vinp = vbatch['inp'].to(device)
                    vgt = vbatch['gt'].to(device)
                    if vinp.dim() == 3:
                        vinp = vinp.unsqueeze(0)
                        vgt = vgt.unsqueeze(0)
                    vpred = model(vinp)
                    if vpred.shape != vgt.shape:
                        if vpred.shape[1] == 1 and vgt.shape[1] == 3:
                            vgt = vgt[:, 0:1, ...]
                        elif vpred.shape[1] == 3 and vgt.shape[1] == 1:
                            vgt = vgt.repeat(1, 3, 1, 1)
                    vloss_map = torch.abs(vpred - vgt)
                    vbase_loss = vloss_map.mean()
                    vloss = vbase_loss
                    val_loss_acc += vloss.item()
                    # 计算批内每张图像的 PSNR
                    try:
                        vpred_np = (vpred.clamp(0, 1).cpu().numpy() * 255.0).astype('float32')
                        vgt_np = (vgt.clamp(0, 1).cpu().numpy() * 255.0).astype('float32')
                        # vpred_np shape: (B, C, H, W)
                        for j in range(vpred_np.shape[0]):
                            pred_img = vpred_np[j].transpose(1, 2, 0)  # HWC
                            gt_img = vgt_np[j].transpose(1, 2, 0)
                            try:
                                p = calculate_psnr(pred_img, gt_img, crop_border=0, input_order='HWC', test_y_channel=False)
                            except Exception:
                                p = float('nan')
                            if not (p != p):
                                psnr_acc += p
                    except Exception:
                        pass
                    val_steps += 1
            avg_val_loss = val_loss_acc / max(1, val_steps)
            # 计算并打印 PSNR
            try:
                avg_psnr = psnr_acc / max(1, val_steps)
            except Exception:
                avg_psnr = float('nan')

            is_best = avg_val_loss < best_loss
            print(f'Validation after epoch {epoch+1}: avg_val_loss={avg_val_loss:.6f} avg_psnr={avg_psnr:.3f} best_saved={int(is_best)}')
            # 写验证日志（包括是否保存为新的 best）
            try:
                with open(val_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch+1},{avg_val_loss:.6f},{avg_psnr:.3f},{int(is_best)}\n")
            except Exception:
                pass
            if is_best:
                best_loss = avg_val_loss
                ckpt_path = os.path.join(save_dir, 'best_model.pth')
                tmp_path = ckpt_path + '.tmp'
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'best_loss': best_loss, 'best_psnr': avg_psnr}, tmp_path)
                try:
                    os.replace(tmp_path, ckpt_path)
                except Exception:
                    os.rename(tmp_path, ckpt_path)
                print(f'Saved best model (by val loss) {ckpt_path} best_psnr={avg_psnr:.3f}')
        else:
            # 非验证轮或没有验证集：不进行保存操作
            pass


def run_test_with_config(cfg: Dict[str, Any]):
    # delegate to test_fma.py CLI by building argv
    test_cfg = cfg.get('test', {})
    args = ['test_fma.py']
    if 'data_root' in test_cfg:
        args += ['--data_root', str(test_cfg['data_root'])]
    if 'checkpoint' in test_cfg:
        args += ['--checkpoint', str(test_cfg['checkpoint'])]
    if 'save_dir' in test_cfg:
        args += ['--save_dir', str(test_cfg['save_dir'])]
    if 'device' in test_cfg:
        args += ['--device', str(test_cfg['device'])]
    if 'test_mask_csv' in test_cfg:
        args += ['--test_mask_csv', str(test_cfg['test_mask_csv'])]
    if 'image_border' in test_cfg:
        args += ['--image_border', str(test_cfg['image_border'])]
    if 'in_chans' in test_cfg:
        args += ['--in_chans', str(test_cfg['in_chans'])]

    # replace sys.argv and import test_fma main
    old_argv = sys.argv
    try:
        sys.argv = args
        import importlib
        spec = importlib.import_module('test_fma')
        if hasattr(spec, 'main'):
            spec.main()
        else:
            raise RuntimeError('test_fma.main not found')
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description='Unified entry: train/test with config')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, required=True, help='path to config (INI/JSON/YAML)')
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    if args.train:
        train_from_config(cfg)
    if args.test:
        run_test_with_config(cfg)


if __name__ == '__main__':
    main()

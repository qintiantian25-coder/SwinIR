import os
import sys
import time
import json
import argparse
import configparser
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

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
    num_workers = int(tr_cfg.get('num_workers', 4))
    save_dir = tr_cfg.get('save_dir', 'experiments/fma_checkpoints')
    device_str = tr_cfg.get('device', 'cuda')
    in_chans = int(tr_cfg.get('in_chans', 1))
    use_mask = bool(tr_cfg.get('use_mask', False))

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and 'cuda' in device_str else 'cpu')

    ds = FMADataset(data_root, split=split, patch_size=patch_size, augment=True, gray_mode=gray_mode)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = build_model(device, in_chans=in_chans, img_size=patch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(loader):
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)
            mask = batch['mask'].to(device)

            if inp.dim() == 3:
                inp = inp.unsqueeze(0)
                gt = gt.unsqueeze(0)
                mask = mask.unsqueeze(0)

            pred = model(inp)

            if pred.shape != gt.shape:
                if pred.shape[1] == 1 and gt.shape[1] == 3:
                    gt = gt[:, 0:1, ...]
                elif pred.shape[1] == 3 and gt.shape[1] == 1:
                    gt = gt.repeat(1, 3, 1, 1)

            loss_map = torch.abs(pred - gt)
            if use_mask:
                mask_w = 1.0 + (alpha - 1.0) * mask
                mask_w3 = mask_w.expand_as(loss_map)
                loss = (loss_map * mask_w3).mean()
            else:
                loss = loss_map.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} Step {global_step} Loss {loss.item():.6f}')

        t1 = time.time()
        print(f'Epoch {epoch+1} finished. Avg loss: {epoch_loss/len(loader):.6f}. Time: {t1-t0:.1f}s')

        ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}, ckpt_path)
        print('Saved', ckpt_path)


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

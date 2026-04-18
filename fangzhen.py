import cv2
import numpy as np
import random
import os
import glob
import csv


# --- 基础工具 ---
def get_random_dark_color():
    """通用暗色采样 (0-100灰度)"""
    return random.randint(0, 10) if random.random() < 0.3 else random.randint(10, 100)


def get_mostly_black_color():
    """专为暗主导块采样的深度黑色 (0-15灰度，确保视觉上极明显)"""
    # 绝大部分为纯黑(0-5)，极少部分为深黑(5-15)
    return random.randint(0, 5) if random.random() < 0.9 else random.randint(5, 15)


# ==========================================
# 仿真模块：深度强化型混合聚合块 (无缝粘合，无外圈)
# ==========================================

def gen_mostly_black_tight_blob(w, h, target_pts, dominant_type, forbidden):
    """
    生成高度粘合且视觉上极明显的黑主导块或白主导块
    """
    pts_dict = {}
    cx, cy = 0, 0
    # 1. 选址避让
    for _ in range(50):
        tx, ty = random.randint(120, w - 120), random.randint(120, h - 120)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty;
            break
    else:
        cx, cy = random.randint(120, w - 120), random.randint(120, h - 120)

    # 2. 紧凑生长：核心思想是新点必须在已有点的 1 像素邻域内
    current_pts = [(cx, cy)]

    # 颜色分配逻辑：在这里应用 dominant_type 的极性变化
    if dominant_type == 'white':
        pts_dict[(cx, cy)] = 255
    else:
        # 暗主导初始点直接使用深度黑色
        pts_dict[(cx, cy)] = get_mostly_black_color()

    while len(pts_dict) < target_pts:
        # 从现有像素中选基点，强制 8 邻域紧密生长
        base_x, base_y = random.choice(current_pts)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = base_x + dx, base_y + dy

        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in pts_dict:
            # 修改颜色比例：暗主导块更稠密，白主导维持80/20
            if dominant_type == 'white':
                # 80% 白色，20% 通用随机灰
                c = 255 if random.random() < 0.8 else get_random_dark_color()
            else:
                # --- 修改逻辑：90% 深度黑，10% 白色穿插 ---
                if random.random() < 0.90:
                    c = get_mostly_black_color()
                else:
                    c = 255

            pts_dict[(nx, ny)] = c
            current_pts.append((nx, ny))

    # 转换格式
    pts = [(x, y, c) for (x, y), c in pts_dict.items()]
    return pts, (cx, cy)


# ==========================================
# 其他固定模块 (保持优化后的物理特性)
# ==========================================

def gen_extra_long_lines(w, h):
    """超长破碎线：单像素主干，极少毛刺"""
    pts = []
    for _ in range(random.randint(2, 3)):
        cx, cy = random.randint(30, w // 2), random.randint(30, h // 2)
        dx, dy = random.choice([(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)])
        length = random.randint(120, 180)
        for _ in range(length):
            if 5 <= cx < w - 5 and 5 <= cy < h - 5:
                # 通用采样的破碎效果
                c = get_random_dark_color() if random.random() < 0.4 else 255
                pts.append((cx, cy, c))
            if random.random() < 0.05:
                sx, sy = cx + random.randint(-1, 1), cy + random.randint(-1, 1)
                if 5 <= sx < w - 5 and 5 <= sy < h - 5:
                    pts.append((sx, sy, get_random_dark_color()))
            cx, cy = cx + dx, cy + dy
            if not (0 <= cx < w and 0 <= cy < h): break
    return pts


def grow_compact_blob(w, h, target_w, target_d, forbidden):
    """大型/中型块：带污染边缘 (使用通用暗色采样)"""
    pts = []
    cx, cy = 0, 0
    for _ in range(50):
        tx, ty = random.randint(120, w - 120), random.randint(120, h - 120)
        if not any(abs(tx - ex) < r and abs(ty - ey) < r for ex, ey, r in forbidden):
            cx, cy = tx, ty;
            break
    else:
        cx, cy = random.randint(120, w - 120), random.randint(120, h - 120)

    w_pts = set([(cx, cy)]);
    bnd = [(cx, cy)]
    while len(w_pts) < target_w and bnd:
        px, py = random.choice(bnd)
        dirs = [(1, 0)] * 4 + [(-1, 0)] * 4 + [(0, 1)] * 4 + [(0, -1)] * 4 + [(1, 1)]
        dx, dy = random.choice(dirs)
        nx, ny = px + dx, py + dy
        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in w_pts:
            w_pts.add((nx, ny));
            bnd.append((nx, ny))
        if len(bnd) > target_w // 2: bnd.pop(0)

    d_pts = set();
    bnd_d = list(w_pts)
    while len(d_pts) < target_d and bnd_d:
        px, py = random.choice(bnd_d)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])
        nx, ny = px + dx, py + dy
        if 5 <= nx < w - 5 and 5 <= ny < h - 5 and (nx, ny) not in w_pts and (nx, ny) not in d_pts:
            d_pts.add((nx, ny));
            bnd_d.append((nx, ny))

    # 使用通用采样确保边缘污染的多样性
    for x, y in d_pts: pts.append((x, y, get_random_dark_color()))
    for x, y in w_pts:
        c = get_random_dark_color() if random.random() < 0.05 else 255
        pts.append((x, y, c))
    return pts, (cx, cy)


# ==========================================
# 批处理执行引擎
# ==========================================

def run_consistent_simulation(src_dir, dst_dir, mask_dir):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not img_paths: return

    h, w = 512, 640
    all_static_blind_params = []
    forbidden = [(w // 4, h // 4, 180)]

    # 1. 基础散布 (0-100 通用暗色)
    for wl, hl, cnt in [(w, h, 800), (w // 2, h // 2, 2000)]:
        for _ in range(cnt):
            tx, ty = random.randint(0, wl - 2), random.randint(0, hl - 2)
            all_static_blind_params.append((tx, ty, get_random_dark_color()))
            all_static_blind_params.append((tx + random.choice([0, 1]), ty + random.choice([0, 1]), 255))

    # 2. 超长破碎线
    all_static_blind_params += gen_extra_long_lines(w, h)

    # 3. 四个高度粘合的 30 像素不规则块
    # 使用深度强化后的黑采样 gen_mostly_black_tight_blob
    for _ in range(2):  # 两个白主导
        p, center = gen_mostly_black_tight_blob(w, h, 32, 'white', forbidden)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], 80))
    for _ in range(2):  # 两个深度黑主导
        p, center = gen_mostly_black_tight_blob(w, h, 32, 'dark', forbidden)
        all_static_blind_params += p
        forbidden.append((center[0], center[1], 80))

    # 4. 大型/中型块 (带通用污染边缘)
    configs = [(130, 150, 2), (45, 60, 2)]
    for wt, dt, n in configs:
        for _ in range(n):
            p, center = grow_compact_blob(w, h, wt, dt, forbidden)
            all_static_blind_params += p
            forbidden.append((center[0], center[1], 100))

    # 5. 渲染保存掩膜和CSV
    mask_img = np.zeros((h, w), dtype=np.uint8)
    csv_records = []
    for idx, p in enumerate(img_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out_img = img.copy()
        for x, y, c in all_static_blind_params:
            if 0 <= y < h and 0 <= x < w:
                out_img[y, x] = c
                if idx == 0:
                    mask_img[y, x] = 255
                    csv_records.append([x, y, img[y, x], c])
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(p)), out_img)

    cv2.imwrite(os.path.join(mask_dir, "blind_pixel_mask.png"), mask_img)
    with open(os.path.join(mask_dir, "blind_pixel_coords.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'original_gray', 'simulated_gray'])
        writer.writerows(csv_records)


if __name__ == "__main__":
    DATA_BASE = r"/home/tianyu/Pythonproject/SwinIR/data"
    run_consistent_simulation(
        os.path.join(DATA_BASE, "val_sharp", "001"),
        os.path.join(DATA_BASE, "val_blur", "001"),
        os.path.join(DATA_BASE, "val_mask", "001")
    )
    print("仿真完成：暗主导块已改为稠密、深度黑色簇（0-15灰度），视觉上极清晰。")
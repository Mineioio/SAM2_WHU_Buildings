# ================================================================
# Cell 9（更新版）：提示策略对比 + 保存全部结果 + 提示位置可视化
# 在每张图上画出质心/多点/外接框的位置，保存到 Drive
# ================================================================

import os, time, cv2, torch, gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ========== 配置 ==========
if os.path.exists("/content/data/sam2_project/dense"):
    DATA_ROOT = "/content/data/sam2_project"
elif os.path.exists("/content/data/dense"):
    DATA_ROOT = "/content/data"
else:
    raise FileNotFoundError("找不到数据！")

CATEGORIES = ["dense", "sparse", "large", "small", "irregular"]
CAT_NAMES_CN = {"dense": "密集建筑", "sparse": "稀疏建筑", "large": "大型建筑",
                "small": "小型建筑", "irregular": "不规则建筑"}
SAVE_DIR = "/content/drive/MyDrive/SAM2_WHU/results"

# ========== 工具函数 ==========
def compute_pixel_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool); gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum(); gt_area = gt.sum()
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}

def extract_buildings(gt_binary):
    num_labels, labels = cv2.connectedComponents(gt_binary)
    buildings = []
    for i in range(1, num_labels):
        bm = (labels == i).astype(np.uint8)
        if bm.sum() < 50: continue
        buildings.append(bm)
    return buildings

def get_centroid(bm):
    ys, xs = np.where(bm > 0)
    return np.array([[xs.mean(), ys.mean()]])

def get_multi_points(bm, n=5):
    ys, xs = np.where(bm > 0)
    if len(xs) <= n: indices = np.arange(len(xs))
    else: indices = np.random.choice(len(xs), n, replace=False)
    return np.stack([xs[indices], ys[indices]], axis=1)

def get_bbox(bm):
    ys, xs = np.where(bm > 0)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])

# ========== 提示位置可视化函数（核心新增） ==========
def draw_prompts_on_image(image_rgb, buildings, prompt_type):
    """
    在图像上画出提示位置，返回标注后的图像。
    - 单点：红色大圆点
    - 多点：蓝色小圆点
    - 框：绿色矩形框
    """
    vis = image_rgb.copy()

    for bld in buildings:
        if prompt_type == "single_point":
            pt = get_centroid(bld)[0]
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)       # 红色实心大圆
            cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 1)     # 白色外圈

        elif prompt_type == "multi_point":
            pts = get_multi_points(bld, n=5)
            for p in pts:
                px, py = int(p[0]), int(p[1])
                cv2.circle(vis, (px, py), 4, (0, 100, 255), -1)  # 蓝色实心小圆
                cv2.circle(vis, (px, py), 5, (255, 255, 255), 1)  # 白色外圈

        elif prompt_type == "box":
            bbox = get_bbox(bld)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形

    return vis

def save_prompt_comparison(image_rgb, gt_255, buildings, preds, metrics_dict, save_path):
    """
    保存一张 6 列对比大图：
    原图 | 真值 | 单点提示位置+预测 | 多点提示位置+预测 | 框提示位置+预测 | 三者叠加对比
    """
    h, w = image_rgb.shape[:2]

    def add_label(img, text):
        img_c = img.copy()
        cv2.putText(img_c, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return img_c

    # 列1：原图
    col1 = add_label(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), "Original")

    # 列2：真值
    gt_3ch = cv2.cvtColor(gt_255, cv2.COLOR_GRAY2BGR)
    col2 = add_label(gt_3ch, "Ground Truth")

    cols = [col1, col2]

    # 列3-5：三种提示（提示位置画在图上 + 预测 mask 半透明叠加）
    prompt_types = ["single_point", "multi_point", "box"]
    labels = ["Single Pt", "Multi Pt(5)", "BBox"]
    colors_overlay = [(255, 100, 100), (100, 100, 255), (100, 255, 100)]

    for pt, label, color in zip(prompt_types, labels, colors_overlay):
        # 画提示位置
        prompt_vis = draw_prompts_on_image(image_rgb, buildings, pt)
        # 把预测 mask 半透明叠加到图上
        pred = preds[pt]
        pred_bool = pred > 127
        prompt_vis[pred_bool] = (np.array(prompt_vis[pred_bool], dtype=np.float32) * 0.5 +
                                  np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
        m = metrics_dict[pt]
        col = add_label(cv2.cvtColor(prompt_vis, cv2.COLOR_RGB2BGR),
                        f"{label} IoU={m['IoU']:.3f}")
        cols.append(col)

    combined = np.hstack(cols)
    cv2.imwrite(save_path, combined)

# ========== 加载模型 ==========
os.chdir("/content/sam2")
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                        "checkpoints/sam2.1_hiera_base_plus.pt", device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
print("✅ SAM2 base+ ImagePredictor 加载完成\n")

# ========== 收集图像 ==========
all_items = []
for cat in CATEGORIES:
    img_dir = os.path.join(DATA_ROOT, cat, "image")
    lbl_dir = os.path.join(DATA_ROOT, cat, "label")
    if not os.path.exists(img_dir): continue
    for fname in sorted(os.listdir(img_dir)):
        lbl_path = os.path.join(lbl_dir, fname)
        if os.path.exists(lbl_path):
            all_items.append({"image": os.path.join(img_dir, fname),
                              "label": lbl_path, "category": cat, "filename": fname})
print(f"📊 共 {len(all_items)} 张图像\n")

# ========== 创建保存目录 ==========
prompt_types = ["single_point", "multi_point", "box"]
prompt_names = {"single_point": "单点提示", "multi_point": "多点提示（5点）", "box": "框提示"}

for pt in prompt_types:
    os.makedirs(os.path.join(SAVE_DIR, "cell9_prompts", pt, "predictions"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "cell9_prompts", "prompt_visualizations"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "cell9_prompts", "comparisons"), exist_ok=True)

# ========== 主实验 ==========
all_metrics = {pt: [] for pt in prompt_types}  # 每种提示的逐图指标
np.random.seed(42)

for idx, item in enumerate(all_items):
    image = cv2.imread(item["image"])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    if gt_binary.sum() == 0: continue

    buildings = extract_buildings(gt_binary)
    if len(buildings) == 0: continue

    fname_base = os.path.splitext(item["filename"])[0] + ".png"

    # ===== 保存三种提示位置的可视化（不需要跑模型） =====
    prompt_vis_combined_cols = [cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)]  # 原图

    for pt, color_name in zip(prompt_types, ["SinglePt(red)", "MultiPt(blue)", "BBox(green)"]):
        vis = draw_prompts_on_image(image_rgb, buildings, pt)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.putText(vis_bgr, color_name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        prompt_vis_combined_cols.append(vis_bgr)

    prompt_vis_path = os.path.join(SAVE_DIR, "cell9_prompts", "prompt_visualizations", fname_base)
    cv2.imwrite(prompt_vis_path, np.hstack(prompt_vis_combined_cols))

    # ===== 对每种提示策略跑 SAM2 =====
    predictor.set_image(image_rgb)  # 只编码一次，三种提示共享
    preds = {}
    metrics_dict = {}

    for pt in prompt_types:
        combined_pred = np.zeros_like(gt_binary, dtype=bool)

        for bld in buildings:
            if pt == "single_point":
                coords = get_centroid(bld)
                labels_arr = np.array([1])
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels_arr, multimask_output=False)
            elif pt == "multi_point":
                coords = get_multi_points(bld, n=5)
                labels_arr = np.ones(len(coords))
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels_arr, multimask_output=False)
            elif pt == "box":
                bbox = get_bbox(bld)
                masks, scores, _ = predictor.predict(box=bbox, multimask_output=False)

            combined_pred = np.logical_or(combined_pred, masks[0])

        pred_out = combined_pred.astype(np.uint8) * 255
        m = compute_pixel_metrics(pred_out, gt_binary * 255)
        preds[pt] = pred_out
        metrics_dict[pt] = m
        all_metrics[pt].append({**m, "filename": item["filename"], "category": item["category"]})

        # 保存每种提示的预测 mask
        cv2.imwrite(os.path.join(SAVE_DIR, "cell9_prompts", pt, "predictions", fname_base), pred_out)

    # ===== 保存三策略对比大图（含提示位置 + 预测叠加）=====
    save_prompt_comparison(
        image_rgb, gt_binary * 255, buildings, preds, metrics_dict,
        os.path.join(SAVE_DIR, "cell9_prompts", "comparisons", fname_base))

    if (idx+1) % 10 == 0 or idx == len(all_items) - 1:
        avg_ious = {pt: np.mean([x['IoU'] for x in all_metrics[pt]]) for pt in prompt_types}
        print(f"  进度: {idx+1}/{len(all_items)} | "
              f"单点={avg_ious['single_point']:.4f} "
              f"多点={avg_ious['multi_point']:.4f} "
              f"框={avg_ious['box']:.4f}")

# ========== 输出总表 ==========
rows = []
for pt in prompt_types:
    mlist = all_metrics[pt]
    avg = {k: np.mean([x[k] for x in mlist]) for k in ["IoU","F1","Precision","Recall"]}
    rows.append({"提示方式": prompt_names[pt],
                 "IoU": f"{avg['IoU']:.4f}", "F1": f"{avg['F1']:.4f}",
                 "Precision": f"{avg['Precision']:.4f}", "Recall": f"{avg['Recall']:.4f}"})
df = pd.DataFrame(rows)
print("\n" + "="*70)
print("📊 三种提示策略对比总结")
print("="*70)
print(df.to_string(index=False))
df.to_csv(os.path.join(SAVE_DIR, "cell9_prompt_summary.csv"), index=False, encoding="utf-8-sig")

# ========== 交叉分析：建筑类型 × 提示方式 ==========
cross_rows = []
for cat in CATEGORIES:
    row = {"建筑物类型": CAT_NAMES_CN.get(cat, cat)}
    for pt in prompt_types:
        cat_ious = [x["IoU"] for x in all_metrics[pt] if x["category"] == cat]
        row[prompt_names[pt]] = f"{np.mean(cat_ious):.4f}" if cat_ious else "N/A"
    cross_rows.append(row)
df_cross = pd.DataFrame(cross_rows)
print("\n📊 交叉分析：建筑类型 × 提示方式 IoU")
print(df_cross.to_string(index=False))
df_cross.to_csv(os.path.join(SAVE_DIR, "cell9_cross_analysis.csv"), index=False, encoding="utf-8-sig")

# 保存逐图明细
for pt in prompt_types:
    pd.DataFrame(all_metrics[pt]).to_csv(
        os.path.join(SAVE_DIR, "cell9_prompts", pt, "detail_metrics.csv"),
        index=False, encoding="utf-8-sig")

del sam2_model, predictor; gc.collect(); torch.cuda.empty_cache()

# ========== 最终输出汇总 ==========
print(f"\n{'='*70}")
print("🎉 Cell 9 完成！保存的文件结构：")
print(f"{'='*70}")
print(f"""
{SAVE_DIR}/cell9_prompts/
├── prompt_visualizations/     ← 每张图的提示位置可视化（红点/蓝点/绿框）
│   ├── xxx.png                   四列：原图 | 单点位置 | 多点位置 | 框位置
│   └── ...
├── comparisons/               ← 每张图的三策略对比大图
│   ├── xxx.png                   五列：原图 | 真值 | 单点预测 | 多点预测 | 框预测
│   └── ...
├── single_point/
│   ├── predictions/           ← 单点提示的预测 mask
│   └── detail_metrics.csv     ← 逐图指标明细
├── multi_point/
│   ├── predictions/
│   └── detail_metrics.csv
├── box/
│   ├── predictions/
│   └── detail_metrics.csv
├── cell9_prompt_summary.csv   ← 三策略总表
└── cell9_cross_analysis.csv   ← 建筑类型×提示方式交叉表
""")

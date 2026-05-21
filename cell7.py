# ================================================================
# Cell 7 ：四种模型大小对比 + 保存全部分割结果
# 每个模型对每张图的预测 mask 和对比图全部保存到 Drive
# ================================================================

import os, time, gc, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ========== 配置 ==========
if os.path.exists("/content/data/sam2_project/dense"):
    DATA_ROOT = "/content/data/sam2_project"
elif os.path.exists("/content/data/dense"):
    DATA_ROOT = "/content/data"
else:
    raise FileNotFoundError("找不到数据！运行 !ls /content/data/ 查看")

CATEGORIES = ["dense", "sparse", "large", "small", "irregular"]
SAVE_DIR = "/content/drive/MyDrive/SAM2_WHU/results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 下载全部权重 ==========
import subprocess
weights = {
    "tiny":  "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base+": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}
base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
for name, fname in weights.items():
    fpath = f"/content/sam2/checkpoints/{fname}"
    if not os.path.exists(fpath):
        print(f"⬇️  下载 {name}...")
        subprocess.run(["wget", "-q", base_url + fname, "-P", "/content/sam2/checkpoints/"], check=True)
    else:
        print(f"✅ {name} 已存在")

model_configs = {
    "tiny":  ("checkpoints/sam2.1_hiera_tiny.pt",       "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("checkpoints/sam2.1_hiera_small.pt",      "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base+": ("checkpoints/sam2.1_hiera_base_plus.pt",  "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("checkpoints/sam2.1_hiera_large.pt",      "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

# ========== 工具函数 ==========
def collect_all_images(data_root, categories):
    all_items = []
    for cat in categories:
        img_dir = os.path.join(data_root, cat, "image")
        lbl_dir = os.path.join(data_root, cat, "label")
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            lbl_path = os.path.join(lbl_dir, fname)
            if os.path.exists(lbl_path):
                all_items.append({"image": os.path.join(img_dir, fname),
                                  "label": lbl_path, "category": cat, "filename": fname})
    return all_items

def compute_pixel_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}

def oracle_filter_masks(sam_masks, gt_binary, iou_thresh=0.3):
    combined = np.zeros_like(gt_binary, dtype=bool)
    for m in sam_masks:
        seg = m["segmentation"]
        overlap = np.logical_and(seg, gt_binary).sum()
        mask_area = seg.sum()
        if mask_area > 0 and (overlap / mask_area) > iou_thresh:
            combined = np.logical_or(combined, seg)
    return combined.astype(np.uint8) * 255

def make_overlay(image_rgb, gt_binary_255, pred_mask_255):
    """生成错误分析叠加图：绿=正确 红=漏检 蓝=误检"""
    overlay = image_rgb.copy()
    gt_bool = gt_binary_255 > 127
    pred_bool = pred_mask_255 > 127
    overlay[np.logical_and(gt_bool, pred_bool)] = [0, 200, 0]
    overlay[np.logical_and(gt_bool, ~pred_bool)] = [200, 0, 0]
    overlay[np.logical_and(~gt_bool, pred_bool)] = [0, 0, 200]
    return overlay

def save_comparison_image(image_rgb, gt_255, pred_255, metrics, save_path):
    """用 cv2 拼接四列对比图并保存（比 matplotlib 快很多）"""
    h, w = image_rgb.shape[:2]
    # 把单通道转三通道用于拼接
    gt_3ch = cv2.cvtColor(gt_255, cv2.COLOR_GRAY2BGR) if len(gt_255.shape) == 2 else gt_255
    pred_3ch = cv2.cvtColor(pred_255, cv2.COLOR_GRAY2BGR) if len(pred_255.shape) == 2 else pred_255
    overlay = make_overlay(image_rgb, gt_255, pred_255)

    # 在每列顶部加文字标签
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return img_copy

    col1 = add_label(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), "Original")
    col2 = add_label(gt_3ch, "Ground Truth")
    col3 = add_label(pred_3ch, f"SAM2 Pred IoU={metrics['IoU']:.3f}")
    col4 = add_label(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                     f"G=TP R=FN B=FP F1={metrics['F1']:.3f}")

    combined = np.hstack([col1, col2, col3, col4])
    cv2.imwrite(save_path, combined)

# ========== 主实验 ==========
all_items = collect_all_images(DATA_ROOT, CATEGORIES)
print(f"\n📊 共 {len(all_items)} 张图像\n")

results_list = []

for model_name, (ckpt, cfg) in model_configs.items():
    safe_name = model_name.replace("+", "_plus")  # 文件夹名不能有+号
    print(f"\n{'='*50}")
    print(f"🔄 模型: {model_name}")
    print(f"{'='*50}")

    # 创建保存目录
    pred_dir = os.path.join(SAVE_DIR, "cell7_models", safe_name, "predictions")
    comp_dir = os.path.join(SAVE_DIR, "cell7_models", safe_name, "comparisons")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    # 加载模型
    os.chdir("/content/sam2")
    sam2_model = build_sam2(cfg, ckpt, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model, points_per_side=32, pred_iou_thresh=0.86,
        stability_score_thresh=0.90, min_mask_region_area=100,
    )
    torch.cuda.reset_peak_memory_stats()

    metrics_all = []
    times_all = []

    for idx, item in enumerate(all_items):
        image = cv2.imread(item["image"])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 127).astype(np.uint8)

        if gt_binary.sum() == 0:
            continue

        start_time = time.time()
        sam_masks = mask_generator.generate(image_rgb)
        elapsed = time.time() - start_time
        times_all.append(elapsed)

        pred_mask = oracle_filter_masks(sam_masks, gt_binary)
        m = compute_pixel_metrics(pred_mask, gt_binary)
        metrics_all.append({**m, "filename": item["filename"], "category": item["category"]})

        # ===== 保存预测 mask =====
        fname_base = os.path.splitext(item["filename"])[0] + ".png"
        cv2.imwrite(os.path.join(pred_dir, fname_base), pred_mask)

        # ===== 保存对比图 =====
        save_comparison_image(image_rgb, gt_binary * 255, pred_mask, m,
                              os.path.join(comp_dir, fname_base))

        if (idx + 1) % 20 == 0 or idx == len(all_items) - 1:
            avg_iou = np.mean([x['IoU'] for x in metrics_all])
            print(f"  进度: {idx+1}/{len(all_items)} | 平均IoU: {avg_iou:.4f}")

    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_metrics = {k: np.mean([x[k] for x in metrics_all]) for k in ["IoU", "F1", "Precision", "Recall"]}
    avg_time = np.mean(times_all) * 1000

    results_list.append({
        "模型": model_name, "参数量": {"tiny":"38.9M","small":"46M","base+":"80.8M","large":"224.4M"}[model_name],
        "IoU": f"{avg_metrics['IoU']:.4f}", "F1": f"{avg_metrics['F1']:.4f}",
        "Precision": f"{avg_metrics['Precision']:.4f}", "Recall": f"{avg_metrics['Recall']:.4f}",
        "推理时间(ms)": f"{avg_time:.1f}", "GPU显存(MB)": f"{gpu_mem:.0f}",
    })

    # 保存该模型的逐图指标明细
    df_detail = pd.DataFrame(metrics_all)
    df_detail.to_csv(os.path.join(SAVE_DIR, "cell7_models", safe_name, "detail_metrics.csv"),
                     index=False, encoding="utf-8-sig")

    print(f"  ✅ {model_name} 完成 | IoU={avg_metrics['IoU']:.4f} | {avg_time:.1f}ms/图")
    print(f"     预测 mask 保存到: {pred_dir}")
    print(f"     对比图保存到: {comp_dir}")

    del sam2_model, mask_generator
    gc.collect()
    torch.cuda.empty_cache()

# ========== 输出总表 ==========
df = pd.DataFrame(results_list)
print("\n" + "="*70)
print("📊 四种模型对比总结")
print("="*70)
print(df.to_string(index=False))
df.to_csv(os.path.join(SAVE_DIR, "cell7_model_comparison.csv"), index=False, encoding="utf-8-sig")

print(f"\n💾 全部结果已保存到: {SAVE_DIR}/cell7_models/")
print("   每个模型下有 predictions/（原始 mask）和 comparisons/（四列对比图）")

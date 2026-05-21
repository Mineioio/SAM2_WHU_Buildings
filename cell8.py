# ================================================================
# Cell 8（更新版）：不同建筑物类型分层分析 + 保存全部结果
# ================================================================

import os, time, cv2, torch, gc
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

def oracle_filter_masks(sam_masks, gt_binary, iou_thresh=0.3):
    combined = np.zeros_like(gt_binary, dtype=bool)
    for m in sam_masks:
        seg = m["segmentation"]
        overlap = np.logical_and(seg, gt_binary).sum()
        mask_area = seg.sum()
        if mask_area > 0 and (overlap / mask_area) > iou_thresh:
            combined = np.logical_or(combined, seg)
    return combined.astype(np.uint8) * 255

def save_comparison_image(image_rgb, gt_255, pred_255, metrics, save_path):
    h, w = image_rgb.shape[:2]
    gt_3ch = cv2.cvtColor(gt_255, cv2.COLOR_GRAY2BGR) if len(gt_255.shape)==2 else gt_255
    pred_3ch = cv2.cvtColor(pred_255, cv2.COLOR_GRAY2BGR) if len(pred_255.shape)==2 else pred_255
    overlay = image_rgb.copy()
    gt_bool = gt_255 > 127; pred_bool = pred_255 > 127
    overlay[np.logical_and(gt_bool, pred_bool)] = [0, 200, 0]
    overlay[np.logical_and(gt_bool, ~pred_bool)] = [200, 0, 0]
    overlay[np.logical_and(~gt_bool, pred_bool)] = [0, 0, 200]
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        return img_copy
    col1 = add_label(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), "Original")
    col2 = add_label(gt_3ch, "Ground Truth")
    col3 = add_label(pred_3ch, f"Pred IoU={metrics['IoU']:.3f}")
    col4 = add_label(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), f"G=TP R=FN B=FP")
    cv2.imwrite(save_path, np.hstack([col1, col2, col3, col4]))

# ========== 加载模型 ==========
os.chdir("/content/sam2")
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                        "checkpoints/sam2.1_hiera_base_plus.pt", device="cuda")
mask_generator = SAM2AutomaticMaskGenerator(
    sam2_model, points_per_side=32, pred_iou_thresh=0.86,
    stability_score_thresh=0.90, min_mask_region_area=100,
)
print("✅ SAM2 base+ 加载完成\n")

# ========== 主实验 ==========
category_results = {}

for cat in CATEGORIES:
    img_dir = os.path.join(DATA_ROOT, cat, "image")
    lbl_dir = os.path.join(DATA_ROOT, cat, "label")
    if not os.path.exists(img_dir):
        print(f"⚠️  跳过: {cat}"); continue

    # 创建保存目录
    pred_dir = os.path.join(SAVE_DIR, "cell8_types", cat, "predictions")
    comp_dir = os.path.join(SAVE_DIR, "cell8_types", cat, "comparisons")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    fnames = sorted(os.listdir(img_dir))
    print(f"{'='*50}")
    print(f"📁 {CAT_NAMES_CN[cat]} ({cat}) - {len(fnames)} 张")
    print(f"{'='*50}")

    metrics_list = []

    for idx, fname in enumerate(fnames):
        image = cv2.imread(os.path.join(img_dir, fname))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(lbl_dir, fname), cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        if gt_binary.sum() == 0: continue

        sam_masks = mask_generator.generate(image_rgb)
        pred_mask = oracle_filter_masks(sam_masks, gt_binary)
        m = compute_pixel_metrics(pred_mask, gt_binary)
        metrics_list.append({**m, "filename": fname})

        # 保存预测 mask
        fname_base = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(pred_dir, fname_base), pred_mask)
        # 保存对比图
        save_comparison_image(image_rgb, gt_binary*255, pred_mask, m,
                              os.path.join(comp_dir, fname_base))

        if (idx+1) % 5 == 0:
            print(f"  进度: {idx+1}/{len(fnames)}")

    avg = {k: np.mean([x[k] for x in metrics_list]) for k in ["IoU","F1","Precision","Recall"]}
    category_results[cat] = {"count": len(metrics_list), **avg}

    # 保存该类别逐图明细
    pd.DataFrame(metrics_list).to_csv(
        os.path.join(SAVE_DIR, "cell8_types", cat, "detail_metrics.csv"),
        index=False, encoding="utf-8-sig")

    print(f"  ✅ {CAT_NAMES_CN[cat]}: IoU={avg['IoU']:.4f}")
    print(f"     保存到: {comp_dir}\n")

# ========== 总表 ==========
rows = []
for cat in CATEGORIES:
    if cat in category_results:
        r = category_results[cat]
        rows.append({"建筑物类型": CAT_NAMES_CN[cat], "图片数量": r["count"],
                      "IoU": f"{r['IoU']:.4f}", "F1": f"{r['F1']:.4f}",
                      "Precision": f"{r['Precision']:.4f}", "Recall": f"{r['Recall']:.4f}"})
df = pd.DataFrame(rows)
print("\n" + "="*70)
print("📊 不同建筑物类型分层分析结果")
print("="*70)
print(df.to_string(index=False))
df.to_csv(os.path.join(SAVE_DIR, "cell8_building_type_summary.csv"), index=False, encoding="utf-8-sig")

del sam2_model, mask_generator; gc.collect(); torch.cuda.empty_cache()
print(f"\n💾 全部结果已保存到: {SAVE_DIR}/cell8_types/")
print("   每个类别下有 predictions/ 和 comparisons/")

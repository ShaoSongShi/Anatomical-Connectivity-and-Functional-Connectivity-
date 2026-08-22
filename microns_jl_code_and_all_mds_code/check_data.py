import pandas as pd

# 你的文件路径
roi_path = "/home/user/ShenRuihong/bishe/corr_principal_validation/fly/traced-roi-connections.csv"

# 读取只需要的两列
df = pd.read_csv(roi_path, usecols=["bodyId_post", "roi"])

# 把 ID 转成字符串（避免匹配失败）
df["bodyId_post"] = df["bodyId_post"].astype(str)

# ==========================================
# 统计：每个神经元对应多少个不同的 ROI
# ==========================================
id_roi_count = df.groupby("bodyId_post")["roi"].nunique()

# 找出对应多个 ROI 的神经元
multi_roi_ids = id_roi_count[id_roi_count > 1]

print("=" * 60)
print(f"总神经元数量: {df['bodyId_post'].nunique()}")
print(f"对应 **多个不同 ROI** 的神经元数量: {len(multi_roi_ids)}")
print("=" * 60)

# 显示前20个例子
if len(multi_roi_ids) > 0:
    print("\n前 20 个「1个神经元对应多个ROI」的例子：")
    example_ids = multi_roi_ids.index[:20]
    for bid in example_ids:
        rois = df[df["bodyId_post"] == bid]["roi"].unique()
        print(f"bodyId: {bid}  -->  {list(rois)}")
else:
    print("\✅ 没有神经元对应多个 ROI，数据干净！")
#!/usr/bin/env python3
"""修复后的 fvecs 读取 + 快速验证"""
import numpy as np
import os

def load_fvecs(fname, max_n=None):
    """正确的 fvecs 读取方式"""
    with open(fname, 'rb') as f:
        # 先读维度 (int32)
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        # 每行字节数 = 4(维度int32) + d*4(float32数据)
        row_bytes = 4 + d * 4
        file_size = os.path.getsize(fname)
        n = file_size // row_bytes
        if max_n:
            n = min(n, max_n)
        
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            f.read(4)  # 跳过维度字段
            vectors[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return vectors

def load_fvecs_fast(fname, max_n=None):
    """更快的 fvecs 批量读取"""
    with open(fname, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
    
    # 用 numpy 一次性读取，按 int32+float32 混合格式
    row_floats = 1 + d  # 每行 = 1个int32(当作float32占位) + d个float32
    data = np.fromfile(fname, dtype=np.float32)
    n = len(data) // row_floats
    if max_n:
        n = min(n, max_n)
    data = data[:n * row_floats].reshape(n, row_floats)
    vectors = data[:, 1:].copy()  # 跳过第一列（维度字段）
    return vectors

# 测试
sift_dir = "./data/sift"

print("测试 load_fvecs_fast...")
base = load_fvecs_fast(os.path.join(sift_dir, "sift_base.fvecs"), max_n=10000)
print(f"  base shape: {base.shape}")
print(f"  base[0][:5]: {base[0][:5]}")

queries = load_fvecs_fast(os.path.join(sift_dir, "sift_query.fvecs"), max_n=1000)
print(f"  queries shape: {queries.shape}")

print("\n测试 hnswlib...")
import hnswlib

dim = base.shape[1]
index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=20000, M=16, ef_construction=200, allow_replace_deleted=True)
index.set_ef(64)
index.add_items(base, list(range(len(base))))
print(f"  构建 OK: {len(base)} vectors, dim={dim}")

labels, dists = index.knn_query(queries[:5], k=10)
print(f"  查询 OK: {labels[0][:5]}")

# 测试删除+插入+rebuild
index.mark_deleted(0)
index.mark_deleted(1)
new_vec = np.random.randn(1, dim).astype(np.float32)
index.add_items(new_vec, [10001], replace_deleted=True)
print(f"  删除+插入 OK")

active_ids = list(range(2, 10000)) + [10001]
active_data = np.array(index.get_items(active_ids))
print(f"  get_items OK: shape={active_data.shape}")

new_index = hnswlib.Index(space='l2', dim=dim)
new_index.init_index(max_elements=20000, M=16, ef_construction=200, allow_replace_deleted=True)
new_index.set_ef(64)
new_index.add_items(active_data, active_ids)
labels2, _ = new_index.knn_query(queries[:1], k=10)
print(f"  Rebuild OK: {labels2[0][:5]}")

print("\n全部通过! fvecs 读取和 hnswlib 都正常工作。")
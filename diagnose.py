#!/usr/bin/env python3
"""快速诊断 segfault 原因"""
import numpy as np
import os
import sys

print("Step 1: 检查 Python 和 numpy")
print(f"  Python: {sys.version}")
print(f"  Numpy: {np.__version__}")

print("\nStep 2: 检查 hnswlib")
try:
    import hnswlib
    print(f"  hnswlib 已安装")
except:
    print("  hnswlib 未安装，正在安装...")
    os.system("pip3 install hnswlib --break-system-packages")
    import hnswlib
print(f"  hnswlib OK")

print("\nStep 3: 检查 SIFT 数据文件")
sift_dir = "./data/sift"
for f in ["sift_base.fvecs", "sift_query.fvecs"]:
    path = os.path.join(sift_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {f}: {size} bytes ({size/1024/1024:.1f} MB)")
    else:
        print(f"  {f}: 不存在!")

print("\nStep 4: 尝试用 numpy 直接读取 fvecs")
try:
    fname = os.path.join(sift_dir, "sift_base.fvecs")
    # 标准 fvecs 读取方式
    data = np.fromfile(fname, dtype=np.float32)
    d = int(data[0])  # 第一个 int32 是维度
    print(f"  首个维度值: {d}")
    # fvecs 格式: 每行 = 1个int32(维度) + d个float32(数据)
    row_size = 1 + d  # in float32 units
    n = len(data) // row_size
    print(f"  预计向量数: {n}, 维度: {d}")
    
    # reshape 并去掉维度列
    data = data.reshape(n, row_size)
    vectors = data[:, 1:]  # 去掉每行开头的维度数
    print(f"  向量 shape: {vectors.shape}")
    print(f"  前3个向量的前5维: {vectors[:3, :5]}")
    print("  fvecs 读取 OK!")
except Exception as e:
    print(f"  fvecs 读取失败: {e}")

print("\nStep 5: 尝试读取 query")
try:
    fname = os.path.join(sift_dir, "sift_query.fvecs")
    data = np.fromfile(fname, dtype=np.float32)
    d = int(data[0])
    row_size = 1 + d
    n = len(data) // row_size
    data = data.reshape(n, row_size)
    queries = data[:, 1:]
    print(f"  Query shape: {queries.shape}")
    print("  Query 读取 OK!")
except Exception as e:
    print(f"  Query 读取失败: {e}")

print("\nStep 6: 尝试构建小型 HNSW 索引")
try:
    dim = 128
    n_test = 1000
    test_data = vectors[:n_test].copy()
    
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n_test * 2, M=16, ef_construction=200,
                     allow_replace_deleted=True)
    index.set_ef(64)
    index.add_items(test_data, list(range(n_test)))
    print(f"  构建 OK, {n_test} vectors")
    
    # 测试查询
    labels, distances = index.knn_query(queries[:1], k=10)
    print(f"  查询 OK, labels={labels[0][:5]}")
    
    # 测试删除
    index.mark_deleted(0)
    print(f"  删除 OK")
    
    # 测试 replace_deleted 插入
    new_vec = np.random.randn(1, dim).astype(np.float32)
    index.add_items(new_vec, [n_test], replace_deleted=True)
    print(f"  replace_deleted 插入 OK")
    
    # 测试 get_items
    item = index.get_items([1])
    print(f"  get_items OK, shape={np.array(item).shape}")
    
    print("\n所有测试通过! hnswlib 工作正常。")
    
except Exception as e:
    print(f"  HNSW 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 7: 尝试小规模 churn 模拟")
try:
    dim = 128
    n_init = 5000
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=n_init * 3, M=16, ef_construction=200,
                     allow_replace_deleted=True)
    index.set_ef(64)
    
    init_data = vectors[:n_init].copy()
    index.add_items(init_data, list(range(n_init)))
    
    current_ids = set(range(n_init))
    next_id = n_init
    pool_idx = n_init
    
    for r in range(5):
        # 删除 500 个
        delete_ids = list(current_ids)[:500]
        for did in delete_ids:
            index.mark_deleted(did)
            current_ids.remove(did)
        
        # 插入 500 个新的
        new_data = vectors[pool_idx:pool_idx+500].copy()
        new_ids = list(range(next_id, next_id + 500))
        index.add_items(new_data, new_ids, replace_deleted=True)
        current_ids.update(new_ids)
        next_id += 500
        pool_idx += 500
        
        # 查询
        labels, distances = index.knn_query(queries[:10], k=10)
        print(f"  Churn round {r+1}: OK, active={len(current_ids)}")
    
    # 测试 rebuild
    print("  尝试 rebuild...")
    active_ids = list(current_ids)
    active_data = np.array(index.get_items(active_ids))
    
    new_index = hnswlib.Index(space='l2', dim=dim)
    new_index.init_index(max_elements=n_init * 3, M=16, ef_construction=200,
                         allow_replace_deleted=True)
    new_index.set_ef(64)
    new_index.add_items(active_data, active_ids)
    
    labels, distances = new_index.knn_query(queries[:1], k=10)
    print(f"  Rebuild + 查询 OK!")
    
    print("\n小规模 churn 模拟全部通过!")
    
except Exception as e:
    print(f"  Churn 模拟失败: {e}")
    import traceback
    traceback.print_exc()

print("\n诊断完成。")
import numpy as np
import hnswlib
import time
import os
from scipy.spatial.distance import cdist

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vector_size = 1 + dim
        data = np.fromfile(f, dtype=np.float32)
        n = len(data) // vector_size
        data = data.reshape(n, vector_size)
        vectors = data[:, 1:].astype(np.float32)
    return vectors

def compute_recall(pred, gt, k):
    recall = 0.0
    n = min(len(pred), len(gt))
    for i in range(n):
        recall += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return recall / n

def get_index_memory_mb(index, num_elements, dim, M):
    """估算索引内存占用（MB）"""
    # 每个向量的存储：向量本身 + 邻居列表
    # 向量: dim * 4 bytes
    # 邻居: 大约 M * 2 * 4 bytes (layer 0有2M个邻居)
    vector_bytes = dim * 4
    neighbor_bytes = M * 2 * 4 * 2  # 粗略估计
    total_bytes = num_elements * (vector_bytes + neighbor_bytes)
    return total_bytes / (1024 * 1024)

print("=" * 60)
print("实验1: 不同M值对性能的影响")
print("=" * 60)

# 读取数据
print("\n[加载数据...]")
base_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

test_size = 100000
base_vectors = base_vectors[:test_size]
num_query = 1000
test_queries = query_vectors[:num_query]
dim = base_vectors.shape[1]

# 计算真实最近邻
print("[计算Ground Truth...]")
dists = cdist(test_queries, base_vectors, metric='sqeuclidean')
true_neighbors = np.argsort(dists, axis=1)[:, :10]

# 测试不同的M值
M_values = [8, 16, 32, 64]
ef_construction = 200
ef_search = 50
k = 10

print("\n" + "-" * 60)
print(f"{'M':<6} {'插入时间(s)':<12} {'单次插入(ms)':<14} {'QPS':<10} {'Recall@10':<10} {'内存(MB)':<10}")
print("-" * 60)

results = []

for M in M_values:
    # 创建索引
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=test_size, M=M, ef_construction=ef_construction)
    
    # 测试插入时间
    start_time = time.time()
    index.add_items(base_vectors, list(range(test_size)))
    insert_time = time.time() - start_time
    insert_per_vec = insert_time / test_size * 1000  # 毫秒
    
    # 测试查询
    index.set_ef(ef_search)
    start_time = time.time()
    labels, _ = index.knn_query(test_queries, k=k)
    search_time = time.time() - start_time
    qps = num_query / search_time
    
    # 计算Recall
    recall = compute_recall(labels, true_neighbors, k)
    
    # 估算内存
    memory_mb = get_index_memory_mb(index, test_size, dim, M)
    
    print(f"{M:<6} {insert_time:<12.2f} {insert_per_vec:<14.3f} {qps:<10.0f} {recall:<10.4f} {memory_mb:<10.1f}")
    
    results.append({
        'M': M,
        'insert_time': insert_time,
        'insert_per_vec_ms': insert_per_vec,
        'qps': qps,
        'recall': recall,
        'memory_mb': memory_mb
    })
    
    del index

print("-" * 60)

# 保存结果
print("\n[保存结果到 results/exp1_m_impact.txt]")
with open("results/exp1_m_impact.txt", "w") as f:
    f.write("实验1: 不同M值对性能的影响\n")
    f.write("数据集: SIFT, 100,000向量, 128维\n")
    f.write(f"ef_construction={ef_construction}, ef_search={ef_search}, k={k}\n\n")
    f.write(f"{'M':<6} {'插入时间(s)':<12} {'单次插入(ms)':<14} {'QPS':<10} {'Recall@10':<10} {'内存(MB)':<10}\n")
    f.write("-" * 60 + "\n")
    for r in results:
        f.write(f"{r['M']:<6} {r['insert_time']:<12.2f} {r['insert_per_vec_ms']:<14.3f} {r['qps']:<10.0f} {r['recall']:<10.4f} {r['memory_mb']:<10.1f}\n")

print("\n" + "=" * 60)
print("实验1完成！")
print("=" * 60)
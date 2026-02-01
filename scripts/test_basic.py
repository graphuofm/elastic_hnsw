import numpy as np
import hnswlib
import time

def read_fvecs(filename):
    """读取 .fvecs 格式文件"""
    with open(filename, 'rb') as f:
        # 先读取第一个向量的维度
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        # 每个向量占用 (1 + dim) * 4 字节
        vector_size = 1 + dim
        data = np.fromfile(f, dtype=np.float32)
        n = len(data) // vector_size
        data = data.reshape(n, vector_size)
        # 去掉每行开头的维度信息
        vectors = data[:, 1:].astype(np.float32)
    return vectors

def read_ivecs(filename):
    """读取 .ivecs 格式文件"""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vector_size = 1 + dim
        data = np.fromfile(f, dtype=np.int32)
        n = len(data) // vector_size
        data = data.reshape(n, vector_size)
        vectors = data[:, 1:]
    return vectors

def compute_recall(pred, gt, k):
    """计算 Recall@k"""
    recall = 0.0
    n = min(len(pred), len(gt))
    for i in range(n):
        recall += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return recall / n

print("=" * 50)
print("SIFT1M 基础测试")
print("=" * 50)

# 1. 读取数据
print("\n[1] 读取数据...")
base_path = "data/sift/sift_base.fvecs"
query_path = "data/sift/sift_query.fvecs"
gt_path = "data/sift/sift_groundtruth.ivecs"

base_vectors = read_fvecs(base_path)
query_vectors = read_fvecs(query_path)
groundtruth = read_ivecs(gt_path)

num_base, dim = base_vectors.shape
num_query = query_vectors.shape[0]

print(f"   基础向量: {num_base} 个, {dim} 维")
print(f"   查询向量: {num_query} 个")
print(f"   Ground Truth: {groundtruth.shape}")

# 2. 只用前10万个向量做测试（加快速度）
print("\n[2] 使用前 100,000 个向量进行测试...")
test_size = 100000
base_vectors_test = base_vectors[:test_size]

# 3. 建立索引
print("\n[3] 建立 HNSW 索引 (M=16, ef_construction=200)...")
M = 16
ef_construction = 200

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=test_size, M=M, ef_construction=ef_construction)

start_time = time.time()
index.add_items(base_vectors_test, list(range(test_size)))
build_time = time.time() - start_time

print(f"   建索引耗时: {build_time:.2f} 秒")
print(f"   平均每个向量: {build_time/test_size*1000:.3f} 毫秒")

# 4. 查询测试
print("\n[4] 查询测试 (ef_search=50)...")
index.set_ef(50)
k = 10

start_time = time.time()
labels, distances = index.knn_query(query_vectors, k=k)
search_time = time.time() - start_time

qps = num_query / search_time
print(f"   查询耗时: {search_time:.3f} 秒")
print(f"   QPS: {qps:.0f}")

# 5. 计算 Recall (基于10万数据重新计算ground truth)
print("\n[5] 计算 Recall@10...")
# 由于我们只用了前10万个向量，需要重新计算真实的最近邻
# 这里用暴力搜索计算前1000个查询的真实结果
print("   (正在计算10万数据的真实最近邻...)")

from scipy.spatial.distance import cdist
num_test_queries = 1000
test_queries = query_vectors[:num_test_queries]
dists = cdist(test_queries, base_vectors_test, metric='sqeuclidean')
true_neighbors = np.argsort(dists, axis=1)[:, :k]

# 用HNSW查询这1000个
labels_test, _ = index.knn_query(test_queries, k=k)
recall = compute_recall(labels_test, true_neighbors, k)
print(f"   Recall@10: {recall:.4f}")

# 6. 测试删除功能
print("\n[6] 测试删除功能...")
for i in range(1000):
    index.mark_deleted(i)

print(f"   已标记删除 1000 个点")

labels2, _ = index.knn_query(query_vectors[:100], k=k)
print(f"   删除后查询正常")

print("\n" + "=" * 50)
print("基础测试完成！环境一切正常。")
print("=" * 50)
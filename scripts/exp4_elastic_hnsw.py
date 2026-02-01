import numpy as np
import hnswlib
import time

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

def brute_force_knn(queries, base_vectors, k=10):
    results = []
    for q in queries:
        dists = np.sum((base_vectors - q) ** 2, axis=1)
        idx = np.argsort(dists)[:k]
        results.append(idx)
    return np.array(results)

def compute_recall(pred, gt, k=10):
    recall = 0.0
    for i in range(len(pred)):
        recall += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return recall / len(pred)

print("=" * 70)
print("实验4: Vanilla HNSW vs 定期重建 - 延迟与Recall对比")
print("预计运行时间: 约2分钟")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 实验设置
index_size = 30000
num_rounds = 8
churn_per_round = 3000  # 每轮删旧插新
num_test_queries = 200
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64
k = 10

print(f"\n实验设置:")
print(f"  索引大小: {index_size}")
print(f"  每轮替换: {churn_per_round}个")
print(f"  总轮数: {num_rounds}")

def measure_latency_simple(index, queries, k=10):
    """简单测量平均延迟"""
    start = time.perf_counter()
    for q in queries:
        index.knn_query(np.array([q]), k=k)
    end = time.perf_counter()
    return (end - start) / len(queries) * 1000  # 毫秒

# ============================================================
# 方法A: Vanilla HNSW (只mark_deleted，不重建)
# ============================================================
print("\n" + "=" * 70)
print("方法A: Vanilla HNSW")
print("=" * 70)

id_to_vector_a = {}
max_elements = index_size + num_rounds * churn_per_round + 1000
index_a = hnswlib.Index(space='l2', dim=dim)
index_a.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index_a.set_ef(ef_search)

for i in range(index_size):
    id_to_vector_a[i] = all_vectors[i]
index_a.add_items(all_vectors[:index_size], list(range(index_size)))
active_ids_a = set(range(index_size))
next_id_a = index_size
next_vec_idx_a = index_size

# 初始测量
lat_a = measure_latency_simple(index_a, test_queries[:50])
current_vecs_a = np.array([id_to_vector_a[i] for i in sorted(active_ids_a)])
id_list_a = sorted(active_ids_a)
gt_local = brute_force_knn(test_queries, current_vecs_a, k)
gt_global = np.array([[id_list_a[l] for l in row] for row in gt_local])
labels_a, _ = index_a.knn_query(test_queries, k=k)
recall_a = compute_recall(labels_a, gt_global, k)

results_a = [{'round': 0, 'latency': lat_a, 'recall': recall_a}]

print(f"\n{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}")
print("-" * 35)
print(f"{0:<6} {lat_a:<12.3f} {recall_a:<12.4f}")

for rnd in range(1, num_rounds + 1):
    # 删除
    del_ids = list(active_ids_a)[:churn_per_round]
    for did in del_ids:
        index_a.mark_deleted(did)
        active_ids_a.remove(did)
        del id_to_vector_a[did]
    
    # 插入新的
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx_a % len(all_vectors)]
        id_to_vector_a[next_id_a] = vec
        active_ids_a.add(next_id_a)
        index_a.add_items(np.array([vec]), [next_id_a])
        next_id_a += 1
        next_vec_idx_a += 1
    
    lat_a = measure_latency_simple(index_a, test_queries[:50])
    current_vecs_a = np.array([id_to_vector_a[i] for i in sorted(active_ids_a)])
    id_list_a = sorted(active_ids_a)
    gt_local = brute_force_knn(test_queries, current_vecs_a, k)
    gt_global = np.array([[id_list_a[l] for l in row] for row in gt_local])
    labels_a, _ = index_a.knn_query(test_queries, k=k)
    recall_a = compute_recall(labels_a, gt_global, k)
    
    results_a.append({'round': rnd, 'latency': lat_a, 'recall': recall_a})
    print(f"{rnd:<6} {lat_a:<12.3f} {recall_a:<12.4f}")

print("-" * 35)

# ============================================================
# 方法B: 定期重建 (每轮都重建 - 性能上限)
# ============================================================
print("\n" + "=" * 70)
print("方法B: 每轮重建 (性能上限，但有重建开销)")
print("=" * 70)

id_to_vector_b = {}
for i in range(index_size):
    id_to_vector_b[i] = all_vectors[i]
active_ids_b = list(range(index_size))
next_vec_idx_b = index_size

# 初始索引
index_b = hnswlib.Index(space='l2', dim=dim)
index_b.init_index(max_elements=index_size, M=M, ef_construction=ef_construction)
index_b.set_ef(ef_search)
index_b.add_items(all_vectors[:index_size], list(range(index_size)))

lat_b = measure_latency_simple(index_b, test_queries[:50])
current_vecs_b = np.array([id_to_vector_b[i] for i in active_ids_b])
gt_local = brute_force_knn(test_queries, current_vecs_b, k)
gt_global = np.array([[active_ids_b[l] for l in row] for row in gt_local])
labels_b, _ = index_b.knn_query(test_queries, k=k)
recall_b = compute_recall(labels_b, gt_global, k)

results_b = [{'round': 0, 'latency': lat_b, 'recall': recall_b, 'rebuild_time': 0}]

print(f"\n{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'重建耗时(s)':<12}")
print("-" * 50)
print(f"{0:<6} {lat_b:<12.3f} {recall_b:<12.4f} {'--':<12}")

for rnd in range(1, num_rounds + 1):
    # 更新数据
    active_ids_b = active_ids_b[churn_per_round:]  # 删除前面的
    
    # 添加新数据
    new_vecs = []
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx_b % len(all_vectors)]
        new_id = max(active_ids_b) + 1 if active_ids_b else 0
        id_to_vector_b[new_id] = vec
        active_ids_b.append(new_id)
        new_vecs.append(vec)
        next_vec_idx_b += 1
    
    # 清理旧的id_to_vector
    id_to_vector_b = {aid: id_to_vector_b[aid] for aid in active_ids_b if aid in id_to_vector_b}
    # 对于新添加的，重新映射
    for i, aid in enumerate(active_ids_b[-churn_per_round:]):
        if aid not in id_to_vector_b:
            id_to_vector_b[aid] = new_vecs[i - (len(active_ids_b) - churn_per_round)]
    
    # 完全重建索引
    rebuild_start = time.perf_counter()
    del index_b
    index_b = hnswlib.Index(space='l2', dim=dim)
    index_b.init_index(max_elements=index_size, M=M, ef_construction=ef_construction)
    index_b.set_ef(ef_search)
    
    # 重新编号为0到index_size-1
    rebuild_vecs = []
    for aid in active_ids_b:
        if aid in id_to_vector_b:
            rebuild_vecs.append(id_to_vector_b[aid])
    rebuild_vecs = np.array(rebuild_vecs[:index_size])
    index_b.add_items(rebuild_vecs, list(range(len(rebuild_vecs))))
    
    # 更新映射
    id_to_vector_b = {i: rebuild_vecs[i] for i in range(len(rebuild_vecs))}
    active_ids_b = list(range(len(rebuild_vecs)))
    
    rebuild_time = time.perf_counter() - rebuild_start
    
    lat_b = measure_latency_simple(index_b, test_queries[:50])
    current_vecs_b = rebuild_vecs
    gt_local = brute_force_knn(test_queries, current_vecs_b, k)
    gt_global = np.array([[i for i in row] for row in gt_local])
    labels_b, _ = index_b.knn_query(test_queries, k=k)
    recall_b = compute_recall(labels_b, gt_global, k)
    
    results_b.append({'round': rnd, 'latency': lat_b, 'recall': recall_b, 'rebuild_time': rebuild_time})
    print(f"{rnd:<6} {lat_b:<12.3f} {recall_b:<12.4f} {rebuild_time:<12.2f}")

print("-" * 50)

# ============================================================
# 最终对比
# ============================================================
print("\n" + "=" * 70)
print("最终对比 (第8轮结束后)")
print("=" * 70)

print(f"\n{'指标':<20} {'Vanilla':<15} {'每轮重建':<15}")
print("-" * 50)
print(f"{'最终延迟(ms)':<20} {results_a[-1]['latency']:<15.3f} {results_b[-1]['latency']:<15.3f}")
print(f"{'最终Recall@10':<20} {results_a[-1]['recall']:<15.4f} {results_b[-1]['recall']:<15.4f}")
print(f"{'延迟增长':<20} {results_a[-1]['latency']/results_a[0]['latency']:<14.2f}x {results_b[-1]['latency']/results_b[0]['latency']:<14.2f}x")

total_rebuild_time = sum(r['rebuild_time'] for r in results_b)
print(f"{'总重建耗时(s)':<20} {'0':<15} {total_rebuild_time:<15.2f}")

# 保存结果
with open("results/exp4_comparison.txt", "w") as f:
    f.write("实验4: Vanilla HNSW vs 定期重建\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"设置: 索引{index_size}, 每轮替换{churn_per_round}, 共{num_rounds}轮\n\n")
    
    f.write("方法A: Vanilla HNSW\n")
    f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}\n")
    for r in results_a:
        f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f}\n")
    
    f.write("\n方法B: 每轮重建\n")
    f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'重建耗时(s)':<12}\n")
    for r in results_b:
        rt = r['rebuild_time'] if r['rebuild_time'] > 0 else '--'
        f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f} {rt}\n")
    
    f.write("\n最终对比:\n")
    f.write(f"Vanilla: 延迟={results_a[-1]['latency']:.3f}ms, Recall={results_a[-1]['recall']:.4f}\n")
    f.write(f"重建: 延迟={results_b[-1]['latency']:.3f}ms, Recall={results_b[-1]['recall']:.4f}, 总重建耗时={total_rebuild_time:.2f}s\n")

print("\n[结果已保存到 results/exp4_comparison.txt]")
print("=" * 70)
print("实验4完成！")
print("=" * 70)
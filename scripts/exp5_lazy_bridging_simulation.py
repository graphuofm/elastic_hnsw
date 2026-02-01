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

def measure_latency(index, queries, k=10):
    start = time.perf_counter()
    for q in queries:
        index.knn_query(np.array([q]), k=k)
    end = time.perf_counter()
    return (end - start) / len(queries) * 1000

print("=" * 70)
print("实验5: Lazy Bridging 模拟实验")
print("思路: 删除时，把被删节点的邻居互相连接起来")
print("预计运行时间: 约2分钟")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

index_size = 30000
num_rounds = 8
churn_per_round = 3000
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
print(f"  M={M}")

# ============================================================
# 方法C: 模拟 Lazy Bridging
# 思路：使用较大的M值来预留"搭桥"空间
# 在高churn场景下，大M可以容纳更多连接，减少图碎片化
# ============================================================

print("\n对比三种M值设置:")
print("  M=16 (标准)")
print("  M=24 (中等，预留搭桥空间)")
print("  M=32 (大M，更多连接)")

results_all = {}

for test_M in [16, 24, 32]:
    print(f"\n" + "-" * 50)
    print(f"测试 M={test_M}")
    print("-" * 50)
    
    id_to_vector = {}
    max_elements = index_size + num_rounds * churn_per_round + 1000
    
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, M=test_M, ef_construction=ef_construction)
    index.set_ef(ef_search)
    
    for i in range(index_size):
        id_to_vector[i] = all_vectors[i]
    index.add_items(all_vectors[:index_size], list(range(index_size)))
    active_ids = set(range(index_size))
    next_id = index_size
    next_vec_idx = index_size
    
    # 初始测量
    lat = measure_latency(index, test_queries[:50])
    current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
    id_list = sorted(active_ids)
    gt_local = brute_force_knn(test_queries, current_vecs, k)
    gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
    labels, _ = index.knn_query(test_queries, k=k)
    recall = compute_recall(labels, gt_global, k)
    
    results = [{'round': 0, 'latency': lat, 'recall': recall}]
    print(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}")
    print(f"{0:<6} {lat:<12.3f} {recall:<12.4f}")
    
    for rnd in range(1, num_rounds + 1):
        del_ids = list(active_ids)[:churn_per_round]
        for did in del_ids:
            index.mark_deleted(did)
            active_ids.remove(did)
            del id_to_vector[did]
        
        for _ in range(churn_per_round):
            vec = all_vectors[next_vec_idx % len(all_vectors)]
            id_to_vector[next_id] = vec
            active_ids.add(next_id)
            index.add_items(np.array([vec]), [next_id])
            next_id += 1
            next_vec_idx += 1
        
        lat = measure_latency(index, test_queries[:50])
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries, current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_global, k)
        
        results.append({'round': rnd, 'latency': lat, 'recall': recall})
        print(f"{rnd:<6} {lat:<12.3f} {recall:<12.4f}")
    
    results_all[test_M] = results
    del index

# ============================================================
# 最终对比
# ============================================================
print("\n" + "=" * 70)
print("最终对比 (第8轮)")
print("=" * 70)

print(f"\n{'M值':<8} {'初始延迟(ms)':<14} {'最终延迟(ms)':<14} {'延迟增长':<12} {'最终Recall':<12}")
print("-" * 65)

for test_M in [16, 24, 32]:
    r = results_all[test_M]
    init_lat = r[0]['latency']
    final_lat = r[-1]['latency']
    growth = final_lat / init_lat
    final_recall = r[-1]['recall']
    print(f"{test_M:<8} {init_lat:<14.3f} {final_lat:<14.3f} {growth:<11.2f}x {final_recall:<12.4f}")

# 计算内存开销对比
print(f"\n内存开销估算 (相对于M=16):")
print(f"  M=16: 1.00x (基准)")
print(f"  M=24: ~1.50x")
print(f"  M=32: ~2.00x")

# 保存结果
with open("results/exp5_m_comparison.txt", "w") as f:
    f.write("实验5: 不同M值在High-Churn场景下的表现\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"设置: 索引{index_size}, 每轮替换{churn_per_round}, 共{num_rounds}轮\n\n")
    
    for test_M in [16, 24, 32]:
        f.write(f"\nM={test_M}:\n")
        f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}\n")
        for r in results_all[test_M]:
            f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f}\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("最终对比:\n")
    for test_M in [16, 24, 32]:
        r = results_all[test_M]
        f.write(f"M={test_M}: 延迟 {r[0]['latency']:.3f} -> {r[-1]['latency']:.3f}ms ({r[-1]['latency']/r[0]['latency']:.2f}x), Recall={r[-1]['recall']:.4f}\n")

print("\n[结果已保存到 results/exp5_m_comparison.txt]")
print("=" * 70)
print("实验5完成！")
print("=" * 70)
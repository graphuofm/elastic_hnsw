import numpy as np
import hnswlib
import time
import copy

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
print("实验7: 公平对比 - Vanilla vs Rebuild vs 大M策略")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 实验设置
index_size = 30000
num_rounds = 10
churn_per_round = 3000
num_test_queries = 200
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

ef_construction = 200
ef_search = 64
k = 10

print(f"\n实验设置:")
print(f"  索引大小: {index_size}")
print(f"  每轮替换: {churn_per_round}个")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {num_rounds * churn_per_round} ({num_rounds * churn_per_round / index_size * 100:.0f}%)")

# 存储所有结果
all_results = {}

# ============================================================
# 方法A: Vanilla HNSW (M=16)
# ============================================================
def run_vanilla_experiment(M_value, method_name):
    print(f"\n{'='*60}")
    print(f"方法: {method_name} (M={M_value})")
    print('='*60)
    
    id_to_vector = {}
    max_elements = index_size + num_rounds * churn_per_round + 1000
    
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, M=M_value, ef_construction=ef_construction)
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
    print("-" * 35)
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
    
    print("-" * 35)
    del index
    return results

# ============================================================
# 方法B: Periodic Rebuild
# ============================================================
def run_rebuild_experiment():
    print(f"\n{'='*60}")
    print(f"方法: Periodic Rebuild (每轮重建)")
    print('='*60)
    
    M_value = 16
    id_to_vector = {}
    
    for i in range(index_size):
        id_to_vector[i] = all_vectors[i]
    active_ids = list(range(index_size))
    next_vec_idx = index_size
    
    # 初始索引
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=index_size, M=M_value, ef_construction=ef_construction)
    index.set_ef(ef_search)
    index.add_items(all_vectors[:index_size], list(range(index_size)))
    
    lat = measure_latency(index, test_queries[:50])
    current_vecs = np.array([id_to_vector[i] for i in active_ids])
    gt_local = brute_force_knn(test_queries, current_vecs, k)
    labels, _ = index.knn_query(test_queries, k=k)
    recall = compute_recall(labels, gt_local, k)
    
    results = [{'round': 0, 'latency': lat, 'recall': recall, 'rebuild_time': 0}]
    print(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'重建耗时(s)':<12}")
    print("-" * 50)
    print(f"{0:<6} {lat:<12.3f} {recall:<12.4f} {'--':<12}")
    
    for rnd in range(1, num_rounds + 1):
        # 更新数据
        del_ids = active_ids[:churn_per_round]
        active_ids = active_ids[churn_per_round:]
        for did in del_ids:
            if did in id_to_vector:
                del id_to_vector[did]
        
        # 添加新数据
        new_id = max(active_ids) + 1 if active_ids else 0
        for _ in range(churn_per_round):
            vec = all_vectors[next_vec_idx % len(all_vectors)]
            id_to_vector[new_id] = vec
            active_ids.append(new_id)
            new_id += 1
            next_vec_idx += 1
        
        # 完全重建
        rebuild_start = time.perf_counter()
        del index
        
        rebuild_vecs = np.array([id_to_vector[aid] for aid in active_ids])
        new_ids = list(range(len(rebuild_vecs)))
        
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=index_size, M=M_value, ef_construction=ef_construction)
        index.set_ef(ef_search)
        index.add_items(rebuild_vecs, new_ids)
        
        # 更新映射
        id_to_vector = {i: rebuild_vecs[i] for i in range(len(rebuild_vecs))}
        active_ids = list(range(len(rebuild_vecs)))
        
        rebuild_time = time.perf_counter() - rebuild_start
        
        lat = measure_latency(index, test_queries[:50])
        gt_local = brute_force_knn(test_queries, rebuild_vecs, k)
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_local, k)
        
        results.append({'round': rnd, 'latency': lat, 'recall': recall, 'rebuild_time': rebuild_time})
        print(f"{rnd:<6} {lat:<12.3f} {recall:<12.4f} {rebuild_time:<12.2f}")
    
    print("-" * 50)
    del index
    return results

# 运行所有实验
all_results['Vanilla_M16'] = run_vanilla_experiment(16, 'Vanilla')
all_results['Vanilla_M24'] = run_vanilla_experiment(24, 'Larger M')
all_results['Rebuild'] = run_rebuild_experiment()

# ============================================================
# 最终汇总
# ============================================================
print("\n" + "=" * 70)
print("最终对比汇总")
print("=" * 70)

print(f"\n{'方法':<20} {'初始延迟':<12} {'最终延迟':<12} {'延迟增长':<12} {'最终Recall':<12}")
print("-" * 70)

for method_name, results in all_results.items():
    init_lat = results[0]['latency']
    final_lat = results[-1]['latency']
    growth = final_lat / init_lat
    final_recall = results[-1]['recall']
    print(f"{method_name:<20} {init_lat:<12.3f} {final_lat:<12.3f} {growth:<11.2f}x {final_recall:<12.4f}")

# 额外统计：Rebuild的总重建时间
total_rebuild = sum(r.get('rebuild_time', 0) for r in all_results['Rebuild'])
print(f"\nRebuild方法总重建耗时: {total_rebuild:.2f}秒")

# 保存结果
with open("results/exp7_fair_comparison.txt", "w") as f:
    f.write("实验7: 公平对比 - Vanilla vs Rebuild vs 大M策略\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"设置: 索引{index_size}, 每轮替换{churn_per_round}, 共{num_rounds}轮\n\n")
    
    for method_name, results in all_results.items():
        f.write(f"\n{method_name}:\n")
        f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}\n")
        for r in results:
            f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f}\n")
        init_lat = results[0]['latency']
        final_lat = results[-1]['latency']
        f.write(f"延迟增长: {final_lat/init_lat:.2f}x\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("汇总:\n")
    for method_name, results in all_results.items():
        init_lat = results[0]['latency']
        final_lat = results[-1]['latency']
        f.write(f"{method_name}: {init_lat:.3f}ms -> {final_lat:.3f}ms ({final_lat/init_lat:.2f}x)\n")

print("\n[结果已保存到 results/exp7_fair_comparison.txt]")
print("=" * 70)
print("实验7完成！")
print("=" * 70)
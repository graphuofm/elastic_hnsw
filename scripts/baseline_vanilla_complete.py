import numpy as np
import hnswlib
import time
import json

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
    latencies = []
    for q in queries:
        start = time.perf_counter()
        index.knn_query(np.array([q]), k=k)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    return {
        'mean': np.mean(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }

print("=" * 70)
print("Vanilla HNSW 完整基线测试")
print("=" * 70)
print("\n此测试结果将保存为基线，用于后续与ElasticHNSW对比")

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 完整测试配置
index_size = 100000        # 10万向量
num_rounds = 50            # 50轮
churn_per_round = 5000     # 每轮换5000个 (5%)
num_test_queries = 200
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64
k = 10

total_churn = num_rounds * churn_per_round
print(f"\n测试配置:")
print(f"  索引大小: {index_size}")
print(f"  维度: {dim}")
print(f"  每轮换血: {churn_per_round} ({churn_per_round/index_size*100:.1f}%)")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {total_churn} ({total_churn/index_size*100:.0f}%)")
print(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
print(f"\n预计运行时间: 约10-15分钟")

# 初始化
print("\n" + "=" * 70)
print("开始测试...")
print("=" * 70)

id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 1000

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)

# 构建初始索引
print("\n[构建初始索引...]")
build_start = time.perf_counter()
for i in range(index_size):
    id_to_vector[i] = all_vectors[i]
index.add_items(all_vectors[:index_size], list(range(index_size)))
build_time = time.perf_counter() - build_start
print(f"  构建时间: {build_time:.2f}s")

active_ids = set(range(index_size))
next_id = index_size
next_vec_idx = index_size

# 初始测量
print("\n[测量初始性能...]")
init_latency = measure_latency(index, test_queries[:100])
current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
id_list = sorted(active_ids)
gt_local = brute_force_knn(test_queries[:100], current_vecs, k)
gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
labels, _ = index.knn_query(test_queries[:100], k=k)
init_recall = compute_recall(labels, gt_global, k)

print(f"  初始延迟: {init_latency['mean']:.3f}ms (P99: {init_latency['p99']:.3f}ms)")
print(f"  初始Recall@10: {init_recall:.4f}")

# 存储完整结果
results = {
    'config': {
        'index_size': index_size,
        'num_rounds': num_rounds,
        'churn_per_round': churn_per_round,
        'M': M,
        'ef_construction': ef_construction,
        'ef_search': ef_search,
        'dim': dim
    },
    'build_time': build_time,
    'rounds': []
}

results['rounds'].append({
    'round': 0,
    'cumulative_churn': 0,
    'churn_ratio': 0,
    'latency_mean': init_latency['mean'],
    'latency_p50': init_latency['p50'],
    'latency_p95': init_latency['p95'],
    'latency_p99': init_latency['p99'],
    'recall': init_recall,
    'delete_time': 0,
    'insert_time': 0
})

# 开始High-Churn测试
print("\n[开始High-Churn测试...]")
print(f"\n{'轮次':<6} {'累计换血':<12} {'换血比':<10} {'延迟(ms)':<12} {'P99(ms)':<12} {'Recall':<10}")
print("-" * 70)
print(f"{0:<6} {0:<12} {'0%':<10} {init_latency['mean']:<12.3f} {init_latency['p99']:<12.3f} {init_recall:<10.4f}")

for rnd in range(1, num_rounds + 1):
    # 删除
    del_start = time.perf_counter()
    del_ids = list(active_ids)[:churn_per_round]
    for did in del_ids:
        index.mark_deleted(did)
        active_ids.remove(did)
        del id_to_vector[did]
    delete_time = time.perf_counter() - del_start
    
    # 插入
    ins_start = time.perf_counter()
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.add(next_id)
        index.add_items(np.array([vec]), [next_id])
        next_id += 1
        next_vec_idx += 1
    insert_time = time.perf_counter() - ins_start
    
    # 每5轮测量一次
    if rnd % 5 == 0 or rnd == num_rounds:
        lat = measure_latency(index, test_queries[:100])
        
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries[:100], current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries[:100], k=k)
        recall = compute_recall(labels, gt_global, k)
        
        cumulative = rnd * churn_per_round
        churn_ratio = cumulative / index_size * 100
        
        results['rounds'].append({
            'round': rnd,
            'cumulative_churn': cumulative,
            'churn_ratio': churn_ratio,
            'latency_mean': lat['mean'],
            'latency_p50': lat['p50'],
            'latency_p95': lat['p95'],
            'latency_p99': lat['p99'],
            'recall': recall,
            'delete_time': delete_time,
            'insert_time': insert_time
        })
        
        print(f"{rnd:<6} {cumulative:<12} {churn_ratio:<9.0f}% {lat['mean']:<12.3f} {lat['p99']:<12.3f} {recall:<10.4f}")

print("-" * 70)

# 计算统计
init_lat = results['rounds'][0]['latency_mean']
final_lat = results['rounds'][-1]['latency_mean']
latency_growth = final_lat / init_lat

init_recall = results['rounds'][0]['recall']
final_recall = results['rounds'][-1]['recall']

print(f"\n最终统计:")
print(f"  延迟变化: {init_lat:.3f}ms -> {final_lat:.3f}ms ({latency_growth:.2f}x)")
print(f"  Recall变化: {init_recall:.4f} -> {final_recall:.4f}")
print(f"  总换血量: {total_churn} ({total_churn/index_size:.0f}x索引大小)")

results['summary'] = {
    'init_latency': init_lat,
    'final_latency': final_lat,
    'latency_growth': latency_growth,
    'init_recall': init_recall,
    'final_recall': final_recall
}

# 保存JSON格式（方便后续程序读取）
with open("results/baseline_vanilla.json", "w") as f:
    json.dump(results, f, indent=2)

# 保存可读格式
with open("results/baseline_vanilla.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("Vanilla HNSW 完整基线测试结果\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("配置:\n")
    f.write(f"  索引大小: {index_size}\n")
    f.write(f"  每轮换血: {churn_per_round}\n")
    f.write(f"  总轮数: {num_rounds}\n")
    f.write(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}\n\n")
    
    f.write("详细结果:\n")
    f.write(f"{'轮次':<6} {'累计换血':<12} {'换血比':<10} {'延迟(ms)':<12} {'P99(ms)':<12} {'Recall':<10}\n")
    f.write("-" * 70 + "\n")
    for r in results['rounds']:
        f.write(f"{r['round']:<6} {r['cumulative_churn']:<12} {r['churn_ratio']:<9.0f}% {r['latency_mean']:<12.3f} {r['latency_p99']:<12.3f} {r['recall']:<10.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("总结:\n")
    f.write(f"  延迟增长: {init_lat:.3f}ms -> {final_lat:.3f}ms ({latency_growth:.2f}x)\n")
    f.write(f"  Recall变化: {init_recall:.4f} -> {final_recall:.4f}\n")
    f.write("=" * 70 + "\n")

print("\n" + "=" * 70)
print("基线测试完成！")
print("结果已保存到:")
print("  - results/baseline_vanilla.json (程序可读)")
print("  - results/baseline_vanilla.txt (人类可读)")
print("=" * 70)
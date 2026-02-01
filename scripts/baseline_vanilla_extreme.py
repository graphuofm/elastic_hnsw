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
        'p99': np.percentile(latencies, 99),
        'max': np.max(latencies)
    }

print("=" * 80)
print("Vanilla HNSW 极限压力测试")
print("=" * 80)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 极限测试配置
index_size = 100000        # 10万向量
num_rounds = 200           # 200轮
churn_per_round = 10000    # 每轮换1万个 (10%)
num_test_queries = 100
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64
k = 10

total_churn = num_rounds * churn_per_round
print(f"\n极限测试配置:")
print(f"  索引大小: {index_size}")
print(f"  维度: {dim}")
print(f"  每轮换血: {churn_per_round} ({churn_per_round/index_size*100:.0f}%)")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {total_churn} ({total_churn/index_size:.0f}x 索引大小)")
print(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
print(f"\n预计运行时间: 约20-30分钟")

# 初始化
id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 10000

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
print("[测量初始性能...]")
init_latency = measure_latency(index, test_queries)
current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
id_list = sorted(active_ids)
gt_local = brute_force_knn(test_queries, current_vecs, k)
gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
labels, _ = index.knn_query(test_queries, k=k)
init_recall = compute_recall(labels, gt_global, k)

print(f"  初始延迟: mean={init_latency['mean']:.3f}ms, P99={init_latency['p99']:.3f}ms, max={init_latency['max']:.3f}ms")
print(f"  初始Recall@10: {init_recall:.4f}")

# 存储结果
results = []
results.append({
    'round': 0,
    'cumulative_churn': 0,
    'churn_multiplier': 0,
    'latency_mean': init_latency['mean'],
    'latency_p50': init_latency['p50'],
    'latency_p95': init_latency['p95'],
    'latency_p99': init_latency['p99'],
    'latency_max': init_latency['max'],
    'recall': init_recall
})

# 开始极限测试
print("\n" + "=" * 80)
print("开始极限High-Churn测试...")
print("=" * 80)
print(f"\n{'轮次':<6} {'累计换血':<10} {'换血倍数':<10} {'mean(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'max(ms)':<10} {'Recall':<8}")
print("-" * 95)
print(f"{0:<6} {0:<10} {'0x':<10} {init_latency['mean']:<10.3f} {init_latency['p50']:<10.3f} {init_latency['p95']:<10.3f} {init_latency['p99']:<10.3f} {init_latency['max']:<10.3f} {init_recall:<8.4f}")

test_start_time = time.perf_counter()

for rnd in range(1, num_rounds + 1):
    # 删除
    del_ids = list(active_ids)[:churn_per_round]
    for did in del_ids:
        index.mark_deleted(did)
        active_ids.remove(did)
        del id_to_vector[did]
    
    # 插入
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.add(next_id)
        index.add_items(np.array([vec]), [next_id])
        next_id += 1
        next_vec_idx += 1
    
    # 每10轮测量一次（200轮太多，每轮都测太慢）
    if rnd % 10 == 0 or rnd == num_rounds:
        lat = measure_latency(index, test_queries)
        
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries, current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_global, k)
        
        cumulative = rnd * churn_per_round
        multiplier = cumulative / index_size
        
        results.append({
            'round': rnd,
            'cumulative_churn': cumulative,
            'churn_multiplier': multiplier,
            'latency_mean': lat['mean'],
            'latency_p50': lat['p50'],
            'latency_p95': lat['p95'],
            'latency_p99': lat['p99'],
            'latency_max': lat['max'],
            'recall': recall
        })
        
        print(f"{rnd:<6} {cumulative:<10} {multiplier:<9.0f}x {lat['mean']:<10.3f} {lat['p50']:<10.3f} {lat['p95']:<10.3f} {lat['p99']:<10.3f} {lat['max']:<10.3f} {recall:<8.4f}")

test_total_time = time.perf_counter() - test_start_time
print("-" * 95)

# 最终统计
init_lat = results[0]['latency_mean']
final_lat = results[-1]['latency_mean']
latency_growth = final_lat / init_lat

init_p99 = results[0]['latency_p99']
final_p99 = results[-1]['latency_p99']
p99_growth = final_p99 / init_p99

init_recall = results[0]['recall']
final_recall = results[-1]['recall']

print(f"\n" + "=" * 80)
print("极限测试最终统计")
print("=" * 80)
print(f"\n测试总耗时: {test_total_time:.1f}秒 ({test_total_time/60:.1f}分钟)")
print(f"\n延迟变化:")
print(f"  Mean: {init_lat:.3f}ms -> {final_lat:.3f}ms (增长 {latency_growth:.2f}x)")
print(f"  P99:  {init_p99:.3f}ms -> {final_p99:.3f}ms (增长 {p99_growth:.2f}x)")
print(f"\nRecall变化: {init_recall:.4f} -> {final_recall:.4f}")
print(f"\n总换血量: {total_churn} ({total_churn/index_size:.0f}x 索引大小)")

# 找出延迟增长的关键节点
print(f"\n延迟增长里程碑:")
for r in results:
    growth = r['latency_mean'] / init_lat
    if growth >= 1.5 and 'printed_1.5' not in dir():
        print(f"  1.5x延迟: 轮次{r['round']}, 换血{r['churn_multiplier']:.0f}x")
        printed_1_5 = True
    if growth >= 2.0 and 'printed_2.0' not in dir():
        print(f"  2.0x延迟: 轮次{r['round']}, 换血{r['churn_multiplier']:.0f}x")
        printed_2_0 = True
    if growth >= 3.0 and 'printed_3.0' not in dir():
        print(f"  3.0x延迟: 轮次{r['round']}, 换血{r['churn_multiplier']:.0f}x")
        printed_3_0 = True
    if growth >= 4.0 and 'printed_4.0' not in dir():
        print(f"  4.0x延迟: 轮次{r['round']}, 换血{r['churn_multiplier']:.0f}x")
        printed_4_0 = True
    if growth >= 5.0 and 'printed_5.0' not in dir():
        print(f"  5.0x延迟: 轮次{r['round']}, 换血{r['churn_multiplier']:.0f}x")
        printed_5_0 = True

# 保存JSON
with open("results/baseline_vanilla_extreme.json", "w") as f:
    json.dump({
        'config': {
            'index_size': index_size,
            'num_rounds': num_rounds,
            'churn_per_round': churn_per_round,
            'total_churn': total_churn,
            'M': M,
            'ef_construction': ef_construction,
            'ef_search': ef_search
        },
        'build_time': build_time,
        'test_time': test_total_time,
        'results': results,
        'summary': {
            'init_latency_mean': init_lat,
            'final_latency_mean': final_lat,
            'latency_growth': latency_growth,
            'init_p99': init_p99,
            'final_p99': final_p99,
            'p99_growth': p99_growth,
            'init_recall': init_recall,
            'final_recall': final_recall
        }
    }, f, indent=2)

# 保存详细文本
with open("results/baseline_vanilla_extreme.txt", "w") as f:
    f.write("=" * 95 + "\n")
    f.write("Vanilla HNSW 极限压力测试结果\n")
    f.write("=" * 95 + "\n\n")
    
    f.write("配置:\n")
    f.write(f"  索引大小: {index_size}\n")
    f.write(f"  每轮换血: {churn_per_round} ({churn_per_round/index_size*100:.0f}%)\n")
    f.write(f"  总轮数: {num_rounds}\n")
    f.write(f"  累计换血: {total_churn} ({total_churn/index_size:.0f}x)\n")
    f.write(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}\n\n")
    
    f.write("详细结果:\n")
    f.write(f"{'轮次':<6} {'累计换血':<10} {'换血倍数':<10} {'mean(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'max(ms)':<10} {'Recall':<8}\n")
    f.write("-" * 95 + "\n")
    for r in results:
        f.write(f"{r['round']:<6} {r['cumulative_churn']:<10} {r['churn_multiplier']:<9.0f}x {r['latency_mean']:<10.3f} {r['latency_p50']:<10.3f} {r['latency_p95']:<10.3f} {r['latency_p99']:<10.3f} {r['latency_max']:<10.3f} {r['recall']:<8.4f}\n")
    
    f.write("\n" + "=" * 95 + "\n")
    f.write("总结:\n")
    f.write(f"  测试耗时: {test_total_time:.1f}秒\n")
    f.write(f"  Mean延迟增长: {init_lat:.3f}ms -> {final_lat:.3f}ms ({latency_growth:.2f}x)\n")
    f.write(f"  P99延迟增长: {init_p99:.3f}ms -> {final_p99:.3f}ms ({p99_growth:.2f}x)\n")
    f.write(f"  Recall变化: {init_recall:.4f} -> {final_recall:.4f}\n")
    f.write("=" * 95 + "\n")

print("\n" + "=" * 80)
print("极限测试完成！")
print("结果已保存到:")
print("  - results/baseline_vanilla_extreme.json")
print("  - results/baseline_vanilla_extreme.txt")
print("=" * 80)
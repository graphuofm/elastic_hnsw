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
print("ElasticHNSW 测试 (Lazy Bridging + Adaptive M)")
print("=" * 70)

# 加载之前的Vanilla基线数据
print("\n[加载Vanilla基线数据...]")
try:
    with open("results/baseline_vanilla_extreme.json", "r") as f:
        vanilla_data = json.load(f)
    print("  已加载极限测试基线")
    baseline_config = vanilla_data['config']
except:
    with open("results/baseline_vanilla.json", "r") as f:
        vanilla_data = json.load(f)
    print("  已加载标准基线")
    baseline_config = vanilla_data['config']

# 使用相同配置
index_size = baseline_config['index_size']
num_rounds = baseline_config['num_rounds']
churn_per_round = baseline_config['churn_per_round']
M = baseline_config['M']
ef_construction = baseline_config['ef_construction']
ef_search = baseline_config['ef_search']

print(f"\n配置 (与Vanilla相同):")
print(f"  索引大小: {index_size}")
print(f"  每轮换血: {churn_per_round}")
print(f"  总轮数: {num_rounds}")
print(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}")

# 加载数据
all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")
num_test_queries = 100
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]
k = 10

# 初始化ElasticHNSW (现在就是修改过的hnswlib)
print("\n[构建ElasticHNSW索引...]")
id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 10000

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)

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

print(f"  初始延迟: {init_latency['mean']:.3f}ms (P99: {init_latency['p99']:.3f}ms)")
print(f"  初始Recall@10: {init_recall:.4f}")

# 存储结果
elastic_results = []
elastic_results.append({
    'round': 0,
    'cumulative_churn': 0,
    'latency_mean': init_latency['mean'],
    'latency_p99': init_latency['p99'],
    'recall': init_recall
})

# 开始High-Churn测试
print("\n" + "=" * 70)
print("开始High-Churn测试 (ElasticHNSW)")
print("=" * 70)
print(f"\n{'轮次':<6} {'累计换血':<12} {'换血比':<10} {'延迟(ms)':<12} {'P99(ms)':<12} {'Recall':<10}")
print("-" * 70)
print(f"{0:<6} {0:<12} {'0%':<10} {init_latency['mean']:<12.3f} {init_latency['p99']:<12.3f} {init_recall:<10.4f}")

test_start = time.perf_counter()

for rnd in range(1, num_rounds + 1):
    # 删除 (这里会触发Lazy Bridging)
    del_ids = list(active_ids)[:churn_per_round]
    for did in del_ids:
        index.mark_deleted(did)
        active_ids.remove(did)
        del id_to_vector[did]
    
    # 插入新数据
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.add(next_id)
        index.add_items(np.array([vec]), [next_id])
        next_id += 1
        next_vec_idx += 1
    
    # 每10轮测量一次
    if rnd % 10 == 0 or rnd == num_rounds:
        lat = measure_latency(index, test_queries)
        
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries, current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_global, k)
        
        cumulative = rnd * churn_per_round
        churn_ratio = cumulative / index_size * 100
        
        elastic_results.append({
            'round': rnd,
            'cumulative_churn': cumulative,
            'latency_mean': lat['mean'],
            'latency_p99': lat['p99'],
            'recall': recall
        })
        
        print(f"{rnd:<6} {cumulative:<12} {churn_ratio:<9.0f}% {lat['mean']:<12.3f} {lat['p99']:<12.3f} {recall:<10.4f}")

test_time = time.perf_counter() - test_start
print("-" * 70)
print(f"测试耗时: {test_time:.1f}秒")

# ============================================================
# 与Vanilla对比
# ============================================================
print("\n" + "=" * 70)
print("ElasticHNSW vs Vanilla HNSW 对比")
print("=" * 70)

vanilla_results = vanilla_data['results']

# 找到对应轮次的数据进行对比
print(f"\n{'轮次':<8} {'Vanilla延迟':<14} {'Elastic延迟':<14} {'改进':<10} {'Vanilla Recall':<14} {'Elastic Recall':<14}")
print("-" * 80)

for er in elastic_results:
    rnd = er['round']
    # 找Vanilla对应轮次
    vr = None
    for v in vanilla_results:
        if v['round'] == rnd:
            vr = v
            break
    
    if vr:
        v_lat = vr['latency_mean']
        e_lat = er['latency_mean']
        improvement = (v_lat - e_lat) / v_lat * 100
        print(f"{rnd:<8} {v_lat:<14.3f} {e_lat:<14.3f} {improvement:+9.1f}% {vr['recall']:<14.4f} {er['recall']:<14.4f}")

# 最终总结
print("\n" + "=" * 70)
print("最终总结")
print("=" * 70)

vanilla_init = vanilla_results[0]['latency_mean']
vanilla_final = vanilla_results[-1]['latency_mean']
vanilla_growth = vanilla_final / vanilla_init

elastic_init = elastic_results[0]['latency_mean']
elastic_final = elastic_results[-1]['latency_mean']
elastic_growth = elastic_final / elastic_init

print(f"\n{'指标':<25} {'Vanilla HNSW':<20} {'ElasticHNSW':<20}")
print("-" * 65)
print(f"{'初始延迟 (ms)':<25} {vanilla_init:<20.3f} {elastic_init:<20.3f}")
print(f"{'最终延迟 (ms)':<25} {vanilla_final:<20.3f} {elastic_final:<20.3f}")
print(f"{'延迟增长':<25} {vanilla_growth:<19.2f}x {elastic_growth:<19.2f}x")
print(f"{'初始Recall':<25} {vanilla_results[0]['recall']:<20.4f} {elastic_results[0]['recall']:<20.4f}")
print(f"{'最终Recall':<25} {vanilla_results[-1]['recall']:<20.4f} {elastic_results[-1]['recall']:<20.4f}")

improvement_pct = (vanilla_growth - elastic_growth) / vanilla_growth * 100
print(f"\n延迟增长改进: {improvement_pct:+.1f}%")

if elastic_growth < vanilla_growth:
    print("✓ ElasticHNSW 成功减缓了延迟退化！")
else:
    print("✗ ElasticHNSW 未能改善延迟退化，需要调整策略")

# 保存结果
with open("results/elastic_hnsw_results.json", "w") as f:
    json.dump({
        'config': baseline_config,
        'results': elastic_results,
        'summary': {
            'init_latency': elastic_init,
            'final_latency': elastic_final,
            'latency_growth': elastic_growth,
            'init_recall': elastic_results[0]['recall'],
            'final_recall': elastic_results[-1]['recall']
        },
        'comparison': {
            'vanilla_growth': vanilla_growth,
            'elastic_growth': elastic_growth,
            'improvement_pct': improvement_pct
        }
    }, f, indent=2)

print("\n[结果已保存到 results/elastic_hnsw_results.json]")
print("=" * 70)
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
print("实验6: ElasticHNSW (Lazy Bridging) vs Vanilla HNSW")
print("=" * 70)
print("\n注意: 当前hnswlib已包含Lazy Bridging修改")

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 实验设置
index_size = 30000
num_rounds = 10
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
print(f"  每轮替换: {churn_per_round}个 (删旧插新)")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {num_rounds * churn_per_round} ({num_rounds * churn_per_round / index_size * 100:.0f}%)")

# ============================================================
# 测试 ElasticHNSW (带 Lazy Bridging)
# ============================================================
print("\n" + "=" * 70)
print("ElasticHNSW (Lazy Bridging 已启用)")
print("=" * 70)

id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 1000

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)

# 插入初始数据
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

print(f"\n{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'累计换血':<12}")
print("-" * 45)
print(f"{0:<6} {lat:<12.3f} {recall:<12.4f} {0:<12}")

total_delete_time = 0
total_insert_time = 0

for rnd in range(1, num_rounds + 1):
    # 删除（这里会触发 Lazy Bridging）
    del_start = time.perf_counter()
    del_ids = list(active_ids)[:churn_per_round]
    for did in del_ids:
        index.mark_deleted(did)
        active_ids.remove(did)
        del id_to_vector[did]
    del_time = time.perf_counter() - del_start
    total_delete_time += del_time
    
    # 插入新数据
    ins_start = time.perf_counter()
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.add(next_id)
        index.add_items(np.array([vec]), [next_id])
        next_id += 1
        next_vec_idx += 1
    ins_time = time.perf_counter() - ins_start
    total_insert_time += ins_time
    
    # 测量
    lat = measure_latency(index, test_queries[:50])
    current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
    id_list = sorted(active_ids)
    gt_local = brute_force_knn(test_queries, current_vecs, k)
    gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
    labels, _ = index.knn_query(test_queries, k=k)
    recall = compute_recall(labels, gt_global, k)
    
    cumulative_churn = rnd * churn_per_round
    results.append({'round': rnd, 'latency': lat, 'recall': recall})
    print(f"{rnd:<6} {lat:<12.3f} {recall:<12.4f} {cumulative_churn:<12}")

print("-" * 45)

# 计算统计
init_latency = results[0]['latency']
final_latency = results[-1]['latency']
latency_growth = final_latency / init_latency

print(f"\n统计:")
print(f"  初始延迟: {init_latency:.3f}ms")
print(f"  最终延迟: {final_latency:.3f}ms")
print(f"  延迟增长: {latency_growth:.2f}x")
print(f"  最终Recall: {results[-1]['recall']:.4f}")
print(f"  总删除耗时: {total_delete_time:.2f}s (平均 {total_delete_time/num_rounds:.3f}s/轮)")
print(f"  总插入耗时: {total_insert_time:.2f}s")

# 与之前 Vanilla 的结果对比（手动输入之前的结果）
print("\n" + "=" * 70)
print("与之前 Vanilla HNSW 结果对比")
print("=" * 70)

# 之前实验4的Vanilla结果
vanilla_init = 0.137
vanilla_final = 0.278
vanilla_growth = 2.03

print(f"\n{'指标':<20} {'Vanilla':<15} {'ElasticHNSW':<15} {'改进':<15}")
print("-" * 65)
print(f"{'初始延迟(ms)':<20} {vanilla_init:<15.3f} {init_latency:<15.3f}")
print(f"{'最终延迟(ms)':<20} {vanilla_final:<15.3f} {final_latency:<15.3f}")
print(f"{'延迟增长':<20} {vanilla_growth:<14.2f}x {latency_growth:<14.2f}x {(vanilla_growth-latency_growth)/vanilla_growth*100:+.1f}%")

# 保存结果
with open("results/exp6_elastic_hnsw.txt", "w") as f:
    f.write("实验6: ElasticHNSW (Lazy Bridging) 效果测试\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"设置: 索引{index_size}, 每轮替换{churn_per_round}, 共{num_rounds}轮\n")
    f.write(f"M={M}, ef_construction={ef_construction}, ef_search={ef_search}\n\n")
    
    f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12}\n")
    f.write("-" * 35 + "\n")
    for r in results:
        f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f}\n")
    
    f.write("\n" + "-" * 35 + "\n")
    f.write(f"延迟增长: {init_latency:.3f}ms -> {final_latency:.3f}ms ({latency_growth:.2f}x)\n")
    f.write(f"对比Vanilla ({vanilla_growth:.2f}x): 改进 {(vanilla_growth-latency_growth)/vanilla_growth*100:.1f}%\n")

print("\n[结果已保存到 results/exp6_elastic_hnsw.txt]")
print("=" * 70)
print("实验6完成！")
print("=" * 70)
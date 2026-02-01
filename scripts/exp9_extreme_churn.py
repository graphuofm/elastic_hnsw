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
print("实验9: 极端High-Churn场景 - 长期运行测试")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 极端设置
index_size = 20000
num_rounds = 50           # 50轮！
churn_per_round = 2000    # 每轮换10%
num_test_queries = 100
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64
k = 10

total_churn = num_rounds * churn_per_round
print(f"\n极端实验设置:")
print(f"  索引大小: {index_size}")
print(f"  每轮替换: {churn_per_round} ({churn_per_round/index_size*100:.0f}%)")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {total_churn} ({total_churn/index_size*100:.0f}%，即换血{total_churn//index_size}遍)")
print(f"\n预计运行时间: 约3-4分钟")

# ============================================================
# Vanilla HNSW
# ============================================================
print("\n" + "=" * 60)
print("Vanilla HNSW (无任何优化)")
print("=" * 60)

id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 1000

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)

for i in range(index_size):
    id_to_vector[i] = all_vectors[i]
index.add_items(all_vectors[:index_size], list(range(index_size)))
active_ids = set(range(index_size))
next_id = index_size
next_vec_idx = index_size

# 初始测量
lat = measure_latency(index, test_queries)
current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
id_list = sorted(active_ids)
gt_local = brute_force_knn(test_queries, current_vecs, k)
gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
labels, _ = index.knn_query(test_queries, k=k)
recall = compute_recall(labels, gt_global, k)

vanilla_results = [{'round': 0, 'latency': lat, 'recall': recall}]
baseline_lat = lat

print(f"\n{'轮次':<6} {'延迟(ms)':<10} {'延迟增长':<10} {'Recall':<10} {'换血倍数':<10}")
print("-" * 50)
print(f"{0:<6} {lat:<10.3f} {'1.00x':<10} {recall:<10.4f} {'0x':<10}")

# 每5轮输出一次
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
    
    # 每5轮测量一次
    if rnd % 5 == 0 or rnd == num_rounds:
        lat = measure_latency(index, test_queries)
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries, current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_global, k)
        
        churn_times = rnd * churn_per_round / index_size
        lat_growth = lat / baseline_lat
        vanilla_results.append({'round': rnd, 'latency': lat, 'recall': recall})
        print(f"{rnd:<6} {lat:<10.3f} {lat_growth:<9.2f}x {recall:<10.4f} {churn_times:<9.1f}x")

print("-" * 50)

del index

# ============================================================
# ElasticHNSW (智能重建)
# ============================================================
print("\n" + "=" * 60)
print("ElasticHNSW (智能触发重建, 阈值=1.8x)")
print("=" * 60)

LATENCY_THRESHOLD = 1.8

id_to_vector2 = {}
for i in range(index_size):
    id_to_vector2[i] = all_vectors[i]
active_ids2 = list(range(index_size))
next_vec_idx2 = index_size

index2 = hnswlib.Index(space='l2', dim=dim)
index2.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index2.set_ef(ef_search)
index2.add_items(all_vectors[:index_size], list(range(index_size)))

baseline_lat2 = measure_latency(index2, test_queries)
current_vecs2 = np.array([id_to_vector2[i] for i in active_ids2])
gt_local2 = brute_force_knn(test_queries, current_vecs2, k)
labels2, _ = index2.knn_query(test_queries, k=k)
recall2 = compute_recall(labels2, gt_local2, k)

elastic_results = [{'round': 0, 'latency': baseline_lat2, 'recall': recall2, 'rebuilt': False}]
total_rebuild_time = 0
rebuild_count = 0
next_id2 = index_size

print(f"\n{'轮次':<6} {'延迟(ms)':<10} {'延迟增长':<10} {'Recall':<10} {'重建':<6}")
print("-" * 50)
print(f"{0:<6} {baseline_lat2:<10.3f} {'1.00x':<10} {recall2:<10.4f} {'--':<6}")

for rnd in range(1, num_rounds + 1):
    # 删除
    del_ids2 = active_ids2[:churn_per_round]
    for did in del_ids2:
        index2.mark_deleted(did)
    active_ids2 = active_ids2[churn_per_round:]
    for did in del_ids2:
        if did in id_to_vector2:
            del id_to_vector2[did]
    
    # 插入
    new_vecs = []
    new_ids = []
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx2 % len(all_vectors)]
        id_to_vector2[next_id2] = vec
        active_ids2.append(next_id2)
        new_vecs.append(vec)
        new_ids.append(next_id2)
        next_id2 += 1
        next_vec_idx2 += 1
    index2.add_items(np.array(new_vecs), new_ids)
    
    # 检查是否需要重建
    current_lat = measure_latency(index2, test_queries[:30])  # 快速检测
    lat_ratio = current_lat / baseline_lat2
    
    rebuilt = False
    if lat_ratio > LATENCY_THRESHOLD:
        rebuild_start = time.perf_counter()
        del index2
        
        rebuild_vecs = np.array([id_to_vector2[aid] for aid in active_ids2])
        index2 = hnswlib.Index(space='l2', dim=dim)
        index2.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
        index2.set_ef(ef_search)
        
        new_id_to_vector = {i: rebuild_vecs[i] for i in range(len(rebuild_vecs))}
        index2.add_items(rebuild_vecs, list(range(len(rebuild_vecs))))
        
        id_to_vector2 = new_id_to_vector
        active_ids2 = list(range(len(rebuild_vecs)))
        next_id2 = len(rebuild_vecs)
        
        rebuild_time = time.perf_counter() - rebuild_start
        total_rebuild_time += rebuild_time
        rebuild_count += 1
        rebuilt = True
        
        baseline_lat2 = measure_latency(index2, test_queries[:30])
    
    # 每5轮输出
    if rnd % 5 == 0 or rnd == num_rounds:
        lat = measure_latency(index2, test_queries)
        current_vecs2 = np.array([id_to_vector2[i] for i in active_ids2])
        gt_local2 = brute_force_knn(test_queries, current_vecs2, k)
        gt_global2 = np.array([[active_ids2[l] for l in row] for row in gt_local2])
        labels2, _ = index2.knn_query(test_queries, k=k)
        recall2 = compute_recall(labels2, gt_global2, k)
        
        lat_growth = lat / elastic_results[0]['latency']
        elastic_results.append({'round': rnd, 'latency': lat, 'recall': recall2, 'rebuilt': rebuilt})
        rebuild_str = f"是({rebuild_count})" if rebuilt else "否"
        print(f"{rnd:<6} {lat:<10.3f} {lat_growth:<9.2f}x {recall2:<10.4f} {rebuild_str:<6}")

print("-" * 50)

# ============================================================
# 最终对比
# ============================================================
print("\n" + "=" * 70)
print("最终对比 (50轮, 累计换血500%)")
print("=" * 70)

vanilla_init = vanilla_results[0]['latency']
vanilla_final = vanilla_results[-1]['latency']
vanilla_growth = vanilla_final / vanilla_init

elastic_init = elastic_results[0]['latency']
elastic_final = elastic_results[-1]['latency']
elastic_growth = elastic_final / elastic_init

print(f"\n{'方法':<30} {'初始延迟':<12} {'最终延迟':<12} {'延迟增长':<12} {'重建次数':<10} {'重建耗时':<10}")
print("-" * 90)
print(f"{'Vanilla HNSW':<30} {vanilla_init:<12.3f} {vanilla_final:<12.3f} {vanilla_growth:<11.2f}x {'0':<10} {'0s':<10}")
print(f"{'ElasticHNSW (智能重建)':<30} {elastic_init:<12.3f} {elastic_final:<12.3f} {elastic_growth:<11.2f}x {rebuild_count:<10} {total_rebuild_time:<9.2f}s")

improvement = (vanilla_growth - elastic_growth) / vanilla_growth * 100
print(f"\n延迟增长改进: {improvement:.1f}%")
print(f"重建开销: {total_rebuild_time:.2f}s (平均每次 {total_rebuild_time/max(rebuild_count,1):.2f}s)")

# 保存
with open("results/exp9_extreme_churn.txt", "w") as f:
    f.write("实验9: 极端High-Churn场景 (50轮, 500%换血)\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Vanilla HNSW:\n")
    for r in vanilla_results:
        f.write(f"轮次{r['round']}: 延迟={r['latency']:.3f}ms, Recall={r['recall']:.4f}\n")
    
    f.write(f"\nElasticHNSW (重建{rebuild_count}次, 耗时{total_rebuild_time:.2f}s):\n")
    for r in elastic_results:
        f.write(f"轮次{r['round']}: 延迟={r['latency']:.3f}ms, Recall={r['recall']:.4f}\n")
    
    f.write(f"\n最终对比:\n")
    f.write(f"Vanilla: {vanilla_init:.3f}ms -> {vanilla_final:.3f}ms ({vanilla_growth:.2f}x)\n")
    f.write(f"ElasticHNSW: {elastic_init:.3f}ms -> {elastic_final:.3f}ms ({elastic_growth:.2f}x)\n")
    f.write(f"改进: {improvement:.1f}%\n")

print("\n[结果已保存到 results/exp9_extreme_churn.txt]")
print("=" * 70)
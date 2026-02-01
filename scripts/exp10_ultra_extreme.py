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
print("实验10: 超极端High-Churn - 换血2000% (20遍)")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 超极端设置
index_size = 10000         # 更小的索引，加快速度
num_rounds = 200           # 200轮
churn_per_round = 1000     # 每轮换10%
num_test_queries = 50      # 更少的查询，加快速度
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64
k = 10

total_churn = num_rounds * churn_per_round
print(f"\n超极端实验设置:")
print(f"  索引大小: {index_size}")
print(f"  每轮替换: {churn_per_round} ({churn_per_round/index_size*100:.0f}%)")
print(f"  总轮数: {num_rounds}")
print(f"  累计换血: {total_churn} ({total_churn/index_size*100:.0f}%，即换血{total_churn//index_size}遍)")
print(f"\n预计运行时间: 约5-8分钟")

# ============================================================
# Vanilla HNSW
# ============================================================
print("\n" + "=" * 60)
print("Vanilla HNSW")
print("=" * 60)

id_to_vector = {}
max_elements = index_size + num_rounds * churn_per_round + 1000

index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)

for i in range(index_size):
    id_to_vector[i] = all_vectors[i % len(all_vectors)]
index.add_items(np.array([id_to_vector[i] for i in range(index_size)]), list(range(index_size)))
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

print(f"\n{'轮次':<6} {'换血倍数':<10} {'延迟(ms)':<12} {'延迟增长':<12} {'Recall':<10}")
print("-" * 55)
print(f"{0:<6} {'0x':<10} {lat:<12.3f} {'1.00x':<12} {recall:<10.4f}")

# 每20轮输出一次
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
    
    if rnd % 20 == 0:
        lat = measure_latency(index, test_queries)
        current_vecs = np.array([id_to_vector[i] for i in sorted(active_ids)])
        id_list = sorted(active_ids)
        gt_local = brute_force_knn(test_queries, current_vecs, k)
        gt_global = np.array([[id_list[l] for l in row] for row in gt_local])
        labels, _ = index.knn_query(test_queries, k=k)
        recall = compute_recall(labels, gt_global, k)
        
        churn_times = rnd * churn_per_round / index_size
        lat_growth = lat / baseline_lat
        vanilla_results.append({'round': rnd, 'latency': lat, 'recall': recall, 'churn_times': churn_times})
        print(f"{rnd:<6} {churn_times:<9.0f}x {lat:<12.3f} {lat_growth:<11.2f}x {recall:<10.4f}")

print("-" * 55)
del index

# ============================================================
# ElasticHNSW
# ============================================================
print("\n" + "=" * 60)
print("ElasticHNSW (智能触发重建, 阈值=2.0x)")
print("=" * 60)

LATENCY_THRESHOLD = 2.0

id_to_vector2 = {}
for i in range(index_size):
    id_to_vector2[i] = all_vectors[i % len(all_vectors)]
active_ids2 = list(range(index_size))
next_vec_idx2 = index_size

index2 = hnswlib.Index(space='l2', dim=dim)
index2.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
index2.set_ef(ef_search)
index2.add_items(np.array([id_to_vector2[i] for i in range(index_size)]), list(range(index_size)))

baseline_lat2 = measure_latency(index2, test_queries)
current_vecs2 = np.array([id_to_vector2[i] for i in active_ids2])
gt_local2 = brute_force_knn(test_queries, current_vecs2, k)
labels2, _ = index2.knn_query(test_queries, k=k)
recall2 = compute_recall(labels2, gt_local2, k)

elastic_results = [{'round': 0, 'latency': baseline_lat2, 'recall': recall2}]
total_rebuild_time = 0
rebuild_count = 0
next_id2 = index_size
reference_lat = baseline_lat2  # 用于检测的基准

print(f"\n{'轮次':<6} {'换血倍数':<10} {'延迟(ms)':<12} {'延迟增长':<12} {'Recall':<10} {'累计重建':<10}")
print("-" * 70)
print(f"{0:<6} {'0x':<10} {baseline_lat2:<12.3f} {'1.00x':<12} {recall2:<10.4f} {0:<10}")

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
    
    # 每5轮检查一次是否需要重建
    if rnd % 5 == 0:
        current_lat = measure_latency(index2, test_queries[:20])
        lat_ratio = current_lat / reference_lat
        
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
            
            reference_lat = measure_latency(index2, test_queries[:20])
    
    # 每20轮输出
    if rnd % 20 == 0:
        lat = measure_latency(index2, test_queries)
        current_vecs2 = np.array([id_to_vector2[i] for i in active_ids2])
        gt_local2 = brute_force_knn(test_queries, current_vecs2, k)
        gt_global2 = np.array([[active_ids2[l] for l in row] for row in gt_local2])
        labels2, _ = index2.knn_query(test_queries, k=k)
        recall2 = compute_recall(labels2, gt_global2, k)
        
        churn_times = rnd * churn_per_round / index_size
        lat_growth = lat / elastic_results[0]['latency']
        elastic_results.append({'round': rnd, 'latency': lat, 'recall': recall2, 'churn_times': churn_times})
        print(f"{rnd:<6} {churn_times:<9.0f}x {lat:<12.3f} {lat_growth:<11.2f}x {recall2:<10.4f} {rebuild_count:<10}")

print("-" * 70)

# ============================================================
# 最终对比
# ============================================================
print("\n" + "=" * 70)
print(f"最终对比 ({num_rounds}轮, 累计换血{total_churn//index_size*100}%)")
print("=" * 70)

vanilla_init = vanilla_results[0]['latency']
vanilla_final = vanilla_results[-1]['latency']
vanilla_growth = vanilla_final / vanilla_init
vanilla_final_recall = vanilla_results[-1]['recall']

elastic_init = elastic_results[0]['latency']
elastic_final = elastic_results[-1]['latency']
elastic_growth = elastic_final / elastic_init
elastic_final_recall = elastic_results[-1]['recall']

print(f"\n{'方法':<25} {'初始延迟':<10} {'最终延迟':<10} {'延迟增长':<10} {'最终Recall':<12} {'重建次数':<8} {'重建耗时':<8}")
print("-" * 95)
print(f"{'Vanilla HNSW':<25} {vanilla_init:<10.3f} {vanilla_final:<10.3f} {vanilla_growth:<9.1f}x {vanilla_final_recall:<12.4f} {'0':<8} {'0s':<8}")
print(f"{'ElasticHNSW':<25} {elastic_init:<10.3f} {elastic_final:<10.3f} {elastic_growth:<9.1f}x {elastic_final_recall:<12.4f} {rebuild_count:<8} {total_rebuild_time:<7.1f}s")

improvement = (vanilla_growth - elastic_growth) / vanilla_growth * 100
print(f"\n关键指标:")
print(f"  延迟增长改进: {improvement:.1f}%")
print(f"  Vanilla延迟增长: {vanilla_growth:.1f}x")
print(f"  ElasticHNSW延迟增长: {elastic_growth:.1f}x")
print(f"  重建开销: {total_rebuild_time:.1f}s (共{rebuild_count}次)")

# 保存详细结果
with open("results/exp10_ultra_extreme.txt", "w") as f:
    f.write(f"实验10: 超极端High-Churn ({num_rounds}轮, {total_churn//index_size*100}%换血)\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Vanilla HNSW 详细数据:\n")
    f.write(f"{'轮次':<8} {'换血倍数':<10} {'延迟(ms)':<12} {'Recall':<10}\n")
    for r in vanilla_results:
        ct = r.get('churn_times', 0)
        f.write(f"{r['round']:<8} {ct:<10.0f} {r['latency']:<12.3f} {r['recall']:<10.4f}\n")
    
    f.write(f"\nElasticHNSW 详细数据 (重建{rebuild_count}次):\n")
    f.write(f"{'轮次':<8} {'换血倍数':<10} {'延迟(ms)':<12} {'Recall':<10}\n")
    for r in elastic_results:
        ct = r.get('churn_times', 0)
        f.write(f"{r['round']:<8} {ct:<10.0f} {r['latency']:<12.3f} {r['recall']:<10.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("最终对比:\n")
    f.write(f"Vanilla: 延迟 {vanilla_init:.3f}ms -> {vanilla_final:.3f}ms ({vanilla_growth:.1f}x增长)\n")
    f.write(f"ElasticHNSW: 延迟 {elastic_init:.3f}ms -> {elastic_final:.3f}ms ({elastic_growth:.1f}x增长)\n")
    f.write(f"改进: {improvement:.1f}%\n")
    f.write(f"重建开销: {total_rebuild_time:.1f}s ({rebuild_count}次)\n")

print("\n[结果已保存到 results/exp10_ultra_extreme.txt]")
print("=" * 70)
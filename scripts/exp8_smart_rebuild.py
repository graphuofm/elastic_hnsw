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
print("实验8: ElasticHNSW - 智能触发重建策略")
print("=" * 70)
print("\n核心思想: 监控延迟，当延迟增长超过阈值时触发局部重建")

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

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

# ElasticHNSW 参数
LATENCY_THRESHOLD = 1.5  # 当延迟增长超过50%时触发重建

print(f"\n实验设置:")
print(f"  索引大小: {index_size}")
print(f"  每轮替换: {churn_per_round}")
print(f"  延迟阈值: {LATENCY_THRESHOLD}x (超过则重建)")

# ============================================================
# 方法: ElasticHNSW - 智能重建
# ============================================================
print("\n" + "=" * 60)
print("ElasticHNSW (智能触发重建)")
print("=" * 60)

id_to_vector = {}
for i in range(index_size):
    id_to_vector[i] = all_vectors[i]
active_ids = list(range(index_size))
next_vec_idx = index_size

# 创建初始索引
index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=index_size + num_rounds * churn_per_round, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)
index.add_items(all_vectors[:index_size], list(range(index_size)))

# 测量基准延迟
baseline_latency = measure_latency(index, test_queries[:50])
current_vecs = np.array([id_to_vector[i] for i in active_ids])
gt_local = brute_force_knn(test_queries, current_vecs, k)
gt_global = np.array([[active_ids[l] for l in row] for row in gt_local])
labels, _ = index.knn_query(test_queries, k=k)
recall = compute_recall(labels, gt_global, k)

results = [{'round': 0, 'latency': baseline_latency, 'recall': recall, 'rebuilt': False}]
total_rebuild_time = 0
rebuild_count = 0

print(f"\n{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'延迟比':<10} {'重建':<8}")
print("-" * 55)
print(f"{0:<6} {baseline_latency:<12.3f} {recall:<12.4f} {'1.00x':<10} {'--':<8}")

next_id = index_size

for rnd in range(1, num_rounds + 1):
    # 删除旧数据
    del_ids = active_ids[:churn_per_round]
    for did in del_ids:
        index.mark_deleted(did)
    active_ids = active_ids[churn_per_round:]
    for did in del_ids:
        if did in id_to_vector:
            del id_to_vector[did]
    
    # 插入新数据
    new_vecs = []
    new_ids = []
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.append(next_id)
        new_vecs.append(vec)
        new_ids.append(next_id)
        next_id += 1
        next_vec_idx += 1
    
    index.add_items(np.array(new_vecs), new_ids)
    
    # 测量当前延迟
    current_latency = measure_latency(index, test_queries[:50])
    latency_ratio = current_latency / baseline_latency
    
    # 检查是否需要重建
    rebuilt = False
    if latency_ratio > LATENCY_THRESHOLD:
        rebuild_start = time.perf_counter()
        
        # 重建索引
        del index
        rebuild_vecs = np.array([id_to_vector[aid] for aid in active_ids])
        
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=index_size + (num_rounds - rnd) * churn_per_round, 
                        M=M, ef_construction=ef_construction)
        index.set_ef(ef_search)
        
        # 重新编号
        new_id_to_vector = {}
        for new_local_id, vec in enumerate(rebuild_vecs):
            new_id_to_vector[new_local_id] = vec
        
        index.add_items(rebuild_vecs, list(range(len(rebuild_vecs))))
        
        id_to_vector = new_id_to_vector
        active_ids = list(range(len(rebuild_vecs)))
        next_id = len(rebuild_vecs)
        
        rebuild_time = time.perf_counter() - rebuild_start
        total_rebuild_time += rebuild_time
        rebuild_count += 1
        rebuilt = True
        
        # 重新测量延迟和更新基准
        current_latency = measure_latency(index, test_queries[:50])
        baseline_latency = current_latency  # 重置基准
        latency_ratio = 1.0
    
    # 计算Recall
    current_vecs = np.array([id_to_vector[i] for i in active_ids])
    gt_local = brute_force_knn(test_queries, current_vecs, k)
    gt_global = np.array([[active_ids[l] for l in row] for row in gt_local])
    labels, _ = index.knn_query(test_queries, k=k)
    recall = compute_recall(labels, gt_global, k)
    
    results.append({'round': rnd, 'latency': current_latency, 'recall': recall, 'rebuilt': rebuilt})
    
    rebuild_str = "是" if rebuilt else "否"
    print(f"{rnd:<6} {current_latency:<12.3f} {recall:<12.4f} {latency_ratio:<9.2f}x {rebuild_str:<8}")

print("-" * 55)

# 统计
init_latency = results[0]['latency']
final_latency = results[-1]['latency']

print(f"\n统计:")
print(f"  初始延迟: {init_latency:.3f}ms")
print(f"  最终延迟: {final_latency:.3f}ms")
print(f"  总重建次数: {rebuild_count}")
print(f"  总重建耗时: {total_rebuild_time:.2f}s")

# ============================================================
# 与其他方法对比
# ============================================================
print("\n" + "=" * 70)
print("与其他方法对比")
print("=" * 70)

# 之前实验7的结果
vanilla_init = 0.131
vanilla_final = 0.278
vanilla_growth = vanilla_final / vanilla_init

rebuild_init = 0.136
rebuild_final = 0.126
rebuild_growth = rebuild_final / rebuild_init
rebuild_total_time = 5.30

elastic_growth = final_latency / init_latency

print(f"\n{'方法':<25} {'延迟增长':<12} {'重建次数':<10} {'重建耗时':<12}")
print("-" * 60)
print(f"{'Vanilla (M=16)':<25} {vanilla_growth:<11.2f}x {'0':<10} {'0s':<12}")
print(f"{'Periodic Rebuild':<25} {rebuild_growth:<11.2f}x {'10':<10} {rebuild_total_time:<11.2f}s")
print(f"{'ElasticHNSW (智能重建)':<25} {elastic_growth:<11.2f}x {rebuild_count:<10} {total_rebuild_time:<11.2f}s")

# 保存结果
with open("results/exp8_elastic_hnsw.txt", "w") as f:
    f.write("实验8: ElasticHNSW - 智能触发重建策略\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"设置: 索引{index_size}, 每轮替换{churn_per_round}, 共{num_rounds}轮\n")
    f.write(f"延迟阈值: {LATENCY_THRESHOLD}x\n\n")
    
    f.write(f"{'轮次':<6} {'延迟(ms)':<12} {'Recall@10':<12} {'重建':<8}\n")
    f.write("-" * 40 + "\n")
    for r in results:
        f.write(f"{r['round']:<6} {r['latency']:<12.3f} {r['recall']:<12.4f} {'是' if r['rebuilt'] else '否':<8}\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"总重建次数: {rebuild_count}\n")
    f.write(f"总重建耗时: {total_rebuild_time:.2f}s\n")
    f.write(f"延迟增长: {elastic_growth:.2f}x\n")

print("\n[结果已保存到 results/exp8_elastic_hnsw.txt]")
print("=" * 70)
print("实验8完成！")
print("=" * 70)
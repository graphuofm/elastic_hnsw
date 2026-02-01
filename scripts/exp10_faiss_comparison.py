import numpy as np
import hnswlib
import time

# 尝试导入faiss
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS 已安装")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS 未安装，请运行: pip3 install faiss-cpu --user")

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
print("实验10: hnswlib vs FAISS 对比")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 实验设置
index_size = 50000
num_test_queries = 500
test_queries = query_vectors[:num_test_queries]
dim = all_vectors.shape[1]
k = 10

print(f"\n实验设置:")
print(f"  数据量: {index_size}")
print(f"  维度: {dim}")
print(f"  查询数: {num_test_queries}")

base_vectors = all_vectors[:index_size].astype(np.float32)

# 计算Ground Truth
print("\n[计算Ground Truth...]")
gt = brute_force_knn(test_queries[:100], base_vectors, k)

# ============================================================
# 测试1: 静态索引性能对比
# ============================================================
print("\n" + "=" * 60)
print("测试1: 静态索引构建和查询性能")
print("=" * 60)

results = {}

# --- hnswlib ---
print("\n[hnswlib HNSW]")
M = 16
ef_construction = 200
ef_search = 64

start = time.perf_counter()
index_hnsw = hnswlib.Index(space='l2', dim=dim)
index_hnsw.init_index(max_elements=index_size, M=M, ef_construction=ef_construction)
index_hnsw.add_items(base_vectors, list(range(index_size)))
build_time_hnsw = time.perf_counter() - start

index_hnsw.set_ef(ef_search)
start = time.perf_counter()
labels_hnsw, _ = index_hnsw.knn_query(test_queries, k=k)
search_time_hnsw = time.perf_counter() - start

recall_hnsw = compute_recall(labels_hnsw[:100], gt, k)
qps_hnsw = num_test_queries / search_time_hnsw

results['hnswlib'] = {
    'build_time': build_time_hnsw,
    'search_time': search_time_hnsw,
    'qps': qps_hnsw,
    'recall': recall_hnsw
}
print(f"  构建时间: {build_time_hnsw:.2f}s")
print(f"  查询时间: {search_time_hnsw:.3f}s")
print(f"  QPS: {qps_hnsw:.0f}")
print(f"  Recall@10: {recall_hnsw:.4f}")

# --- FAISS HNSW ---
if FAISS_AVAILABLE:
    print("\n[FAISS HNSW]")
    
    start = time.perf_counter()
    index_faiss_hnsw = faiss.IndexHNSWFlat(dim, M)
    index_faiss_hnsw.hnsw.efConstruction = ef_construction
    index_faiss_hnsw.add(base_vectors)
    build_time_faiss_hnsw = time.perf_counter() - start
    
    index_faiss_hnsw.hnsw.efSearch = ef_search
    start = time.perf_counter()
    _, labels_faiss_hnsw = index_faiss_hnsw.search(test_queries, k)
    search_time_faiss_hnsw = time.perf_counter() - start
    
    recall_faiss_hnsw = compute_recall(labels_faiss_hnsw[:100], gt, k)
    qps_faiss_hnsw = num_test_queries / search_time_faiss_hnsw
    
    results['faiss_hnsw'] = {
        'build_time': build_time_faiss_hnsw,
        'search_time': search_time_faiss_hnsw,
        'qps': qps_faiss_hnsw,
        'recall': recall_faiss_hnsw
    }
    print(f"  构建时间: {build_time_faiss_hnsw:.2f}s")
    print(f"  查询时间: {search_time_faiss_hnsw:.3f}s")
    print(f"  QPS: {qps_faiss_hnsw:.0f}")
    print(f"  Recall@10: {recall_faiss_hnsw:.4f}")
    
    # --- FAISS IVF ---
    print("\n[FAISS IVF-Flat]")
    nlist = 100  # 聚类数
    
    start = time.perf_counter()
    quantizer = faiss.IndexFlatL2(dim)
    index_faiss_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index_faiss_ivf.train(base_vectors)
    index_faiss_ivf.add(base_vectors)
    build_time_faiss_ivf = time.perf_counter() - start
    
    index_faiss_ivf.nprobe = 10
    start = time.perf_counter()
    _, labels_faiss_ivf = index_faiss_ivf.search(test_queries, k)
    search_time_faiss_ivf = time.perf_counter() - start
    
    recall_faiss_ivf = compute_recall(labels_faiss_ivf[:100], gt, k)
    qps_faiss_ivf = num_test_queries / search_time_faiss_ivf
    
    results['faiss_ivf'] = {
        'build_time': build_time_faiss_ivf,
        'search_time': search_time_faiss_ivf,
        'qps': qps_faiss_ivf,
        'recall': recall_faiss_ivf
    }
    print(f"  构建时间: {build_time_faiss_ivf:.2f}s")
    print(f"  查询时间: {search_time_faiss_ivf:.3f}s")
    print(f"  QPS: {qps_faiss_ivf:.0f}")
    print(f"  Recall@10: {recall_faiss_ivf:.4f}")

# ============================================================
# 测试2: 动态更新能力对比 (这是我们的重点!)
# ============================================================
print("\n" + "=" * 60)
print("测试2: 动态更新能力 (High-Churn场景)")
print("=" * 60)
print("\n注意: FAISS的HNSW不支持删除操作!")

# hnswlib 动态更新测试
print("\n[hnswlib - 支持动态删除]")
num_rounds = 20
churn_per_round = 2500

index_hnsw2 = hnswlib.Index(space='l2', dim=dim)
index_hnsw2.init_index(max_elements=index_size + num_rounds * churn_per_round, M=M, ef_construction=ef_construction)
index_hnsw2.set_ef(ef_search)
index_hnsw2.add_items(base_vectors, list(range(index_size)))

id_to_vector = {i: base_vectors[i] for i in range(index_size)}
active_ids = set(range(index_size))
next_id = index_size
next_vec_idx = index_size

# 初始延迟
def measure_latency(index, queries, k=10):
    start = time.perf_counter()
    for q in queries:
        index.knn_query(np.array([q]), k=k)
    end = time.perf_counter()
    return (end - start) / len(queries) * 1000

init_latency = measure_latency(index_hnsw2, test_queries[:50])
hnsw_dynamic_results = [{'round': 0, 'latency': init_latency}]

print(f"\n{'轮次':<6} {'延迟(ms)':<12} {'累计换血':<12}")
print("-" * 35)
print(f"{0:<6} {init_latency:<12.3f} {0:<12}")

for rnd in range(1, num_rounds + 1):
    # 删除
    del_ids = list(active_ids)[:churn_per_round]
    for did in del_ids:
        index_hnsw2.mark_deleted(did)
        active_ids.remove(did)
        del id_to_vector[did]
    
    # 插入
    for _ in range(churn_per_round):
        vec = all_vectors[next_vec_idx % len(all_vectors)]
        id_to_vector[next_id] = vec
        active_ids.add(next_id)
        index_hnsw2.add_items(np.array([vec]), [next_id])
        next_id += 1
        next_vec_idx += 1
    
    if rnd % 5 == 0 or rnd == num_rounds:
        lat = measure_latency(index_hnsw2, test_queries[:50])
        cumulative = rnd * churn_per_round
        hnsw_dynamic_results.append({'round': rnd, 'latency': lat})
        print(f"{rnd:<6} {lat:<12.3f} {cumulative:<12}")

print("-" * 35)
hnsw_latency_growth = hnsw_dynamic_results[-1]['latency'] / hnsw_dynamic_results[0]['latency']
print(f"hnswlib 延迟增长: {hnsw_latency_growth:.2f}x")

# FAISS 动态更新测试 (需要重建)
if FAISS_AVAILABLE:
    print("\n[FAISS - 需要完全重建来处理删除]")
    
    # 模拟FAISS的动态更新：每次删除都需要重建
    faiss_rebuild_times = []
    
    current_vectors = base_vectors.copy()
    current_ids = list(range(index_size))
    
    for rnd in range(1, 5):  # 只测5轮，因为每轮都要重建很慢
        # 模拟删除：移除前churn_per_round个
        current_vectors = current_vectors[churn_per_round:]
        
        # 模拟插入：添加新向量
        new_vecs = all_vectors[index_size + (rnd-1)*churn_per_round : index_size + rnd*churn_per_round]
        current_vectors = np.vstack([current_vectors, new_vecs])
        
        # FAISS需要完全重建
        rebuild_start = time.perf_counter()
        index_faiss_rebuild = faiss.IndexHNSWFlat(dim, M)
        index_faiss_rebuild.hnsw.efConstruction = ef_construction
        index_faiss_rebuild.add(current_vectors)
        rebuild_time = time.perf_counter() - rebuild_start
        faiss_rebuild_times.append(rebuild_time)
    
    avg_rebuild = np.mean(faiss_rebuild_times)
    print(f"  FAISS每次重建耗时: {avg_rebuild:.2f}s")
    print(f"  如果20轮都重建: {avg_rebuild * 20:.1f}s")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("汇总对比")
print("=" * 70)

print(f"\n{'指标':<25} {'hnswlib':<15} {'FAISS HNSW':<15} {'FAISS IVF':<15}")
print("-" * 70)

if FAISS_AVAILABLE:
    print(f"{'静态构建时间(s)':<25} {results['hnswlib']['build_time']:<15.2f} {results['faiss_hnsw']['build_time']:<15.2f} {results['faiss_ivf']['build_time']:<15.2f}")
    print(f"{'静态QPS':<25} {results['hnswlib']['qps']:<15.0f} {results['faiss_hnsw']['qps']:<15.0f} {results['faiss_ivf']['qps']:<15.0f}")
    print(f"{'静态Recall@10':<25} {results['hnswlib']['recall']:<15.4f} {results['faiss_hnsw']['recall']:<15.4f} {results['faiss_ivf']['recall']:<15.4f}")
    print(f"{'支持动态删除':<25} {'是':<15} {'否':<15} {'否':<15}")
    print(f"{'动态场景延迟增长':<25} {hnsw_latency_growth:<14.2f}x {'需重建':<15} {'需重建':<15}")

print("\n关键发现:")
print("1. 静态场景: FAISS和hnswlib性能相近")
print("2. 动态场景: hnswlib支持删除，FAISS需要完全重建")
print("3. 这正是ElasticHNSW的机会 - 优化hnswlib的动态更新性能!")

# 保存结果
with open("results/exp10_faiss_comparison.txt", "w") as f:
    f.write("实验10: hnswlib vs FAISS 对比\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"数据量: {index_size}, 维度: {dim}\n\n")
    
    f.write("静态性能:\n")
    for name, r in results.items():
        f.write(f"  {name}: 构建{r['build_time']:.2f}s, QPS={r['qps']:.0f}, Recall={r['recall']:.4f}\n")
    
    f.write(f"\n动态更新 (hnswlib, {num_rounds}轮换血):\n")
    f.write(f"  延迟增长: {hnsw_latency_growth:.2f}x\n")
    f.write(f"  FAISS不支持删除，需完全重建\n")

print("\n[结果已保存到 results/exp10_faiss_comparison.txt]")
print("=" * 70)
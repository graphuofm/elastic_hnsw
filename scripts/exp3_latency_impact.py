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

print("=" * 70)
print("实验3: Tombstone累积对搜索延迟的影响")
print("预计运行时间: 约60秒")
print("=" * 70)

all_vectors = read_fvecs("data/sift/sift_base.fvecs")
query_vectors = read_fvecs("data/sift/sift_query.fvecs")

# 实验设置
index_size = 50000
num_rounds = 10
delete_per_round = 4000  # 每轮删除
num_queries = 1000
test_queries = query_vectors[:num_queries]
dim = all_vectors.shape[1]

M = 16
ef_construction = 200
ef_search = 64  # 稍大的ef以更明显看到延迟变化
k = 10

print(f"\n实验设置:")
print(f"  初始数据量: {index_size}")
print(f"  每轮删除: {delete_per_round}个")
print(f"  查询数量: {num_queries}")
print(f"  ef_search: {ef_search}")

# 初始化索引
index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=index_size, M=M, ef_construction=ef_construction)
index.set_ef(ef_search)
index.add_items(all_vectors[:index_size], list(range(index_size)))

active_ids = set(range(index_size))

# 测量初始延迟
def measure_latency(index, queries, num_runs=3):
    """测量查询延迟，返回平均值和P99"""
    latencies = []
    for _ in range(num_runs):
        for q in queries:
            start = time.perf_counter()
            index.knn_query(np.array([q]), k=k)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 毫秒
    
    latencies = np.array(latencies)
    return {
        'mean': np.mean(latencies),
        'p50': np.percentile(latencies, 50),
        'p99': np.percentile(latencies, 99),
        'max': np.max(latencies)
    }

print("\n[测量初始延迟...]")
init_latency = measure_latency(index, test_queries[:100], num_runs=2)

results = [{
    'round': 0,
    'tombstones': 0,
    'tombstone_pct': 0.0,
    'mean_latency': init_latency['mean'],
    'p99_latency': init_latency['p99']
}]

print(f"\n{'轮次':<6} {'墓碑数':<10} {'墓碑比例':<10} {'平均延迟(ms)':<14} {'P99延迟(ms)':<14}")
print("-" * 60)
print(f"{0:<6} {0:<10} {'0.0%':<10} {init_latency['mean']:<14.3f} {init_latency['p99']:<14.3f}")

# 开始删除并测量延迟
for round_num in range(1, num_rounds + 1):
    if len(active_ids) <= delete_per_round + 100:
        print(f"[轮次{round_num}] 活跃节点不足，停止")
        break
    
    # 随机删除
    delete_ids = np.random.choice(list(active_ids), size=delete_per_round, replace=False)
    for did in delete_ids:
        index.mark_deleted(did)
        active_ids.remove(did)
    
    tombstones = index_size - len(active_ids)
    tombstone_pct = tombstones / index_size * 100
    
    # 测量延迟
    lat = measure_latency(index, test_queries[:100], num_runs=2)
    
    results.append({
        'round': round_num,
        'tombstones': tombstones,
        'tombstone_pct': tombstone_pct,
        'mean_latency': lat['mean'],
        'p99_latency': lat['p99']
    })
    
    print(f"{round_num:<6} {tombstones:<10} {tombstone_pct:<9.1f}% {lat['mean']:<14.3f} {lat['p99']:<14.3f}")

print("-" * 60)

# 计算延迟增长
latency_increase = results[-1]['mean_latency'] / results[0]['mean_latency']
print(f"\n延迟变化:")
print(f"  平均延迟: {results[0]['mean_latency']:.3f}ms -> {results[-1]['mean_latency']:.3f}ms")
print(f"  延迟增长: {latency_increase:.2f}x")
print(f"  最终墓碑比例: {results[-1]['tombstone_pct']:.1f}%")

# 保存结果
with open("results/exp3_latency_impact.txt", "w") as f:
    f.write("实验3: Tombstone累积对搜索延迟的影响\n")
    f.write("=" * 60 + "\n")
    f.write(f"初始数据量: {index_size}, 每轮删除: {delete_per_round}\n")
    f.write(f"M={M}, ef_search={ef_search}\n\n")
    f.write(f"{'轮次':<6} {'墓碑数':<10} {'墓碑比例':<10} {'平均延迟(ms)':<14} {'P99延迟(ms)':<14}\n")
    f.write("-" * 60 + "\n")
    for r in results:
        f.write(f"{r['round']:<6} {r['tombstones']:<10} {r['tombstone_pct']:<9.1f}% {r['mean_latency']:<14.3f} {r['p99_latency']:<14.3f}\n")
    f.write("-" * 60 + "\n")
    f.write(f"\n延迟增长: {latency_increase:.2f}x\n")

print("\n[结果已保存到 results/exp3_latency_impact.txt]")
print("=" * 70)
print("实验3完成！")
print("=" * 70)
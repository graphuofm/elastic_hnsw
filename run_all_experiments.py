#!/usr/bin/env python3
"""
ElasticHNSW 补充实验 - 修复版
运行: cd /home/jding/elastic_hnsw && python3 run_all_experiments.py 2>&1 | tee all_results.log
预计时间: 40-60分钟
"""

import numpy as np
import time
import json
import os
import sys
import hnswlib
import traceback

RESULTS_DIR = "./results_supplementary"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 数据读取
# ============================================================
def load_fvecs_fast(fname, max_n=None):
    with open(fname, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
    row_floats = 1 + d
    data = np.fromfile(fname, dtype=np.float32)
    n = len(data) // row_floats
    if max_n:
        n = min(n, max_n)
    data = data[:n * row_floats].reshape(n, row_floats)
    return data[:, 1:].copy()

def load_sift(base_dir="./data/sift", max_base=None):
    base = load_fvecs_fast(os.path.join(base_dir, "sift_base.fvecs"), max_base)
    query = load_fvecs_fast(os.path.join(base_dir, "sift_query.fvecs"), 1000)
    return base, query

# ============================================================
# 工具函数
# ============================================================
def measure_latency(index, queries, k=10, n=200):
    q = queries[:n]
    lats = []
    for i in range(len(q)):
        t0 = time.perf_counter()
        index.knn_query(q[i:i+1], k=k)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    return {'mean': float(np.mean(lats)), 'p50': float(np.median(lats)),
            'p95': float(np.percentile(lats, 95)), 'p99': float(np.percentile(lats, 99))}

def compute_recall_bf(index, queries, active_data, active_ids, dim, k=10, n=100):
    """用BruteForce计算recall，更安全"""
    q = queries[:n]
    bf = hnswlib.BFIndex(space='l2', dim=dim)
    bf.init_index(max_elements=len(active_ids))
    bf.add_items(active_data, active_ids)
    
    recs = []
    for i in range(len(q)):
        la, _ = index.knn_query(q[i:i+1], k=k)
        lb, _ = bf.knn_query(q[i:i+1], k=k)
        recs.append(len(set(la[0]) & set(lb[0])) / k)
    return float(np.mean(recs))

def run_churn(data_pool, queries, dim, index_size, replace_pct, num_rounds,
              M=16, ef_c=200, ef_s=64, strategy='vanilla', tau=2.0):
    """
    通用 churn 实验
    strategy: 'vanilla' | 'elastic' | 'full_rebuild'
    """
    repl = int(index_size * replace_pct)
    maxel = index_size + num_rounds * repl + 10000

    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=maxel, M=M, ef_construction=ef_c)
    idx.set_ef(ef_s)

    # 插入初始数据
    init_data = data_pool[:index_size].copy()
    init_ids = list(range(index_size))
    idx.add_items(init_data, init_ids)

    # 跟踪活跃向量
    id_to_vector = {}
    for i in range(index_size):
        id_to_vector[i] = init_data[i]

    active_ids = set(range(index_size))
    nxt_id = index_size
    pool_idx = index_size
    tot_rb_time = 0.0
    tot_rb = 0

    # 初始测量
    active_data = np.array([id_to_vector[i] for i in sorted(active_ids)])
    active_id_list = sorted(active_ids)
    base_lat = measure_latency(idx, queries, n=100)
    base_rec = compute_recall_bf(idx, queries, active_data, active_id_list, dim, n=100)
    base_mean = base_lat['mean']

    results = [{'round': 0, 'latency': base_lat, 'recall': base_rec, 'rebuild': False, 'rb_time': 0}]
    sys.stdout.write(f"    R0: {base_lat['mean']:.3f}ms rec={base_rec:.4f}\n")
    sys.stdout.flush()

    measure_interval = max(1, num_rounds // 10)

    for r in range(1, num_rounds + 1):
        # 删除
        del_list = list(active_ids)[:repl]
        for d in del_list:
            try:
                idx.mark_deleted(d)
            except Exception:
                pass  # 可能已经被删除
            active_ids.discard(d)
            if d in id_to_vector:
                del id_to_vector[d]

        # 插入
        for _ in range(repl):
            if pool_idx < len(data_pool):
                v = data_pool[pool_idx]
                pool_idx += 1
            else:
                v = data_pool[pool_idx % len(data_pool)]
                pool_idx += 1
            
            id_to_vector[nxt_id] = v
            active_ids.add(nxt_id)
            idx.add_items(np.array([v]), [nxt_id])
            nxt_id += 1

        did_rb = False
        rb_t = 0

        if strategy == 'full_rebuild':
            t0 = time.perf_counter()
            aid_list = sorted(active_ids)
            adata = np.array([id_to_vector[i] for i in aid_list])
            
            new_idx = hnswlib.Index(space='l2', dim=dim)
            new_idx.init_index(max_elements=maxel, M=M, ef_construction=ef_c)
            new_idx.set_ef(ef_s)
            new_idx.add_items(adata, aid_list)
            idx = new_idx
            
            rb_t = time.perf_counter() - t0
            did_rb = True
            tot_rb_time += rb_t
            tot_rb += 1

        elif strategy == 'elastic':
            ck = measure_latency(idx, queries, n=50)
            if ck['mean'] / base_mean > tau:
                t0 = time.perf_counter()
                aid_list = sorted(active_ids)
                adata = np.array([id_to_vector[i] for i in aid_list])
                
                new_idx = hnswlib.Index(space='l2', dim=dim)
                new_idx.init_index(max_elements=maxel, M=M, ef_construction=ef_c)
                new_idx.set_ef(ef_s)
                new_idx.add_items(adata, aid_list)
                idx = new_idx
                
                rb_t = time.perf_counter() - t0
                did_rb = True
                tot_rb_time += rb_t
                tot_rb += 1
                base_mean = measure_latency(idx, queries, n=50)['mean']

        # 测量
        lat = measure_latency(idx, queries, n=100)
        
        if r % measure_interval == 0 or r == num_rounds:
            aid_list = sorted(active_ids)
            adata = np.array([id_to_vector[i] for i in aid_list])
            rec = compute_recall_bf(idx, queries, adata, aid_list, dim, n=100)
        else:
            rec = -1

        results.append({'round': r, 'latency': lat, 'recall': rec, 'rebuild': did_rb, 'rb_time': rb_t})

        if r % measure_interval == 0:
            tag = " [RB]" if did_rb else ""
            sys.stdout.write(f"    R{r}: {lat['mean']:.3f}ms p99={lat['p99']:.3f}ms rec={rec:.4f}{tag}\n")
            sys.stdout.flush()

    return results, tot_rb, tot_rb_time

def save(data, fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> {path}")

def sep(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# ============================================================
# 实验 A: τ 敏感性 (10K, 100轮, 1000% churn)
# ============================================================
def exp_a():
    sep("实验 A: τ 敏感性分析 (10K, 100轮)")
    base, queries = load_sift(max_base=200000)
    dim = 128; sz = 10000; pct = 0.10; rounds = 100

    out = {}

    print("  Vanilla...")
    rv, _, _ = run_churn(base, queries, dim, sz, pct, rounds, strategy='vanilla')
    out['vanilla'] = {'init': rv[0]['latency']['mean'], 'final': rv[-1]['latency']['mean'],
                      'growth': rv[-1]['latency']['mean']/rv[0]['latency']['mean'],
                      'final_recall': rv[-1]['recall'],
                      'rebuilds': 0, 'cost': 0}

    for tau in [1.3, 1.5, 2.0, 2.5, 3.0]:
        print(f"  ElasticHNSW tau={tau}...")
        re, rb, rt = run_churn(base, queries, dim, sz, pct, rounds, strategy='elastic', tau=tau)
        out[f'tau_{tau}'] = {'init': re[0]['latency']['mean'], 'final': re[-1]['latency']['mean'],
                             'growth': re[-1]['latency']['mean']/re[0]['latency']['mean'],
                             'final_recall': re[-1]['recall'],
                             'rebuilds': rb, 'cost': rt}

    print(f"\n  {'方案':<15} {'init(ms)':<10} {'final(ms)':<10} {'growth':<8} {'recall':<8} {'#rb':<6} {'cost(s)':<8}")
    print("  " + "-"*66)
    for k, v in out.items():
        rec_str = f"{v['final_recall']:.4f}" if v['final_recall'] >= 0 else "N/A"
        print(f"  {k:<15} {v['init']:<10.3f} {v['final']:<10.3f} {v['growth']:<7.2f}x {rec_str:<8} {v['rebuilds']:<6} {v['cost']:<8.2f}")
    save(out, "exp_a_tau.json")

# ============================================================
# 实验 B: 高维 (128d vs 384d vs 768d)
# ============================================================
def exp_b():
    sep("实验 B: 高维实验 (128/384/768)")
    out = {}
    sz = 10000; pct = 0.10; rounds = 50

    for dim, label in [(128, "128d"), (384, "384d"), (768, "768d")]:
        print(f"\n  --- {label} ---")
        if dim == 128:
            base, queries = load_sift(max_base=100000)
        else:
            np.random.seed(42)
            base = np.random.randn(100000, dim).astype(np.float32)
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            queries = np.random.randn(1000, dim).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        print(f"  Vanilla...")
        rv, _, _ = run_churn(base, queries, dim, sz, pct, rounds, strategy='vanilla')
        print(f"  Elastic tau=2.0...")
        re, rb, rt = run_churn(base, queries, dim, sz, pct, rounds, strategy='elastic', tau=2.0)

        out[label] = {
            'dim': dim,
            'vanilla': {'init': rv[0]['latency']['mean'], 'final': rv[-1]['latency']['mean'],
                        'growth': rv[-1]['latency']['mean']/rv[0]['latency']['mean'],
                        'recall': rv[-1]['recall']},
            'elastic': {'init': re[0]['latency']['mean'], 'final': re[-1]['latency']['mean'],
                        'growth': re[-1]['latency']['mean']/re[0]['latency']['mean'],
                        'recall': re[-1]['recall'], 'rebuilds': rb, 'cost': rt}
        }

    print(f"\n  {'维度':<8} {'V_growth':<10} {'E_growth':<10} {'改善%':<8} {'重建':<6}")
    print("  " + "-"*45)
    for k, v in out.items():
        vg = v['vanilla']['growth']; eg = v['elastic']['growth']
        imp = (1 - eg/vg)*100
        print(f"  {k:<8} {vg:<10.2f}x {eg:<10.2f}x {imp:<8.1f}% {v['elastic']['rebuilds']}")
    save(out, "exp_b_highdim.json")

# ============================================================
# 实验 C: 不同索引规模 (10K/30K/50K/100K)
# ============================================================
def exp_c():
    sep("实验 C: Scalability (10K-100K)")
    base, queries = load_sift(max_base=500000)
    dim = 128; pct = 0.10; rounds = 50
    out = {}

    for sz in [10000, 30000, 50000, 100000]:
        print(f"\n  --- {sz//1000}K ---")
        print(f"  Vanilla...")
        rv, _, _ = run_churn(base, queries, dim, sz, pct, rounds, strategy='vanilla')
        print(f"  Elastic tau=2.0...")
        re, rb, rt = run_churn(base, queries, dim, sz, pct, rounds, strategy='elastic', tau=2.0)

        out[f'{sz//1000}K'] = {
            'size': sz,
            'vanilla': {'init': rv[0]['latency']['mean'], 'final': rv[-1]['latency']['mean'],
                        'growth': rv[-1]['latency']['mean']/rv[0]['latency']['mean']},
            'elastic': {'init': re[0]['latency']['mean'], 'final': re[-1]['latency']['mean'],
                        'growth': re[-1]['latency']['mean']/re[0]['latency']['mean'],
                        'rebuilds': rb, 'cost': rt, 'avg_rb': rt/max(rb,1)}
        }

    print(f"\n  {'规模':<6} {'V_init':<8} {'V_final':<9} {'V_grow':<8} {'E_final':<9} {'E_grow':<8} {'#rb':<5} {'avg(s)':<8}")
    print("  " + "-"*62)
    for k, v in out.items():
        vi = v['vanilla']; ei = v['elastic']
        print(f"  {k:<6} {vi['init']:<8.3f} {vi['final']:<9.3f} {vi['growth']:<8.2f}x "
              f"{ei['final']:<9.3f} {ei['growth']:<8.2f}x {ei['rebuilds']:<5} {ei['avg_rb']:<8.2f}")
    save(out, "exp_c_scalability.json")

# ============================================================
# 实验 D: 不同 churn 强度
# ============================================================
def exp_d():
    sep("实验 D: Churn 强度 (5%/10%/20%/30%)")
    base, queries = load_sift(max_base=200000)
    dim = 128; sz = 10000
    out = {}

    for pct, rounds in [(0.05, 100), (0.10, 50), (0.20, 25), (0.30, 17)]:
        total = pct * rounds * 100
        label = f"{int(pct*100)}%x{rounds}r"
        print(f"\n  --- {label} (total ~{total:.0f}%) ---")
        print(f"  Vanilla...")
        rv, _, _ = run_churn(base, queries, dim, sz, pct, rounds, strategy='vanilla')
        print(f"  Elastic tau=2.0...")
        re, rb, rt = run_churn(base, queries, dim, sz, pct, rounds, strategy='elastic', tau=2.0)

        out[label] = {
            'pct': pct, 'rounds': rounds, 'total_churn': total,
            'vanilla': {'init': rv[0]['latency']['mean'], 'final': rv[-1]['latency']['mean'],
                        'growth': rv[-1]['latency']['mean']/rv[0]['latency']['mean'],
                        'recall': rv[-1]['recall']},
            'elastic': {'init': re[0]['latency']['mean'], 'final': re[-1]['latency']['mean'],
                        'growth': re[-1]['latency']['mean']/re[0]['latency']['mean'],
                        'recall': re[-1]['recall'], 'rebuilds': rb, 'cost': rt}
        }

    print(f"\n  {'强度':<15} {'churn%':<8} {'V_grow':<8} {'E_grow':<8} {'改善%':<8} {'#rb':<5}")
    print("  " + "-"*50)
    for k, v in out.items():
        vg = v['vanilla']['growth']; eg = v['elastic']['growth']
        print(f"  {k:<15} {v['total_churn']:<8.0f} {vg:<8.2f}x {eg:<8.2f}x {(1-eg/vg)*100:<8.1f}% {v['elastic']['rebuilds']}")
    save(out, "exp_d_intensity.json")

# ============================================================
# 实验 E: 极端 2000% + 全方案对比
# ============================================================
def exp_e():
    sep("实验 E: 极端 2000% 全方案对比")
    base, queries = load_sift(max_base=500000)
    dim = 128; sz = 10000; pct = 0.10; rounds = 200
    out = {}

    print("  Vanilla...")
    rv, _, _ = run_churn(base, queries, dim, sz, pct, rounds, strategy='vanilla')
    out['vanilla'] = {'init': rv[0]['latency']['mean'], 'final': rv[-1]['latency']['mean'],
                      'growth': rv[-1]['latency']['mean']/rv[0]['latency']['mean'],
                      'final_recall': rv[-1]['recall'],
                      'rebuilds': 0, 'cost': 0}

    for tau in [1.5, 2.0, 2.5]:
        print(f"  Elastic tau={tau}...")
        re, rb, rt = run_churn(base, queries, dim, sz, pct, rounds, strategy='elastic', tau=tau)
        out[f'elastic_{tau}'] = {'init': re[0]['latency']['mean'], 'final': re[-1]['latency']['mean'],
                                 'growth': re[-1]['latency']['mean']/re[0]['latency']['mean'],
                                 'final_recall': re[-1]['recall'],
                                 'rebuilds': rb, 'cost': rt}

    print("  Full Rebuild (every round)...")
    rf, rbf, rtf = run_churn(base, queries, dim, sz, pct, rounds, strategy='full_rebuild')
    out['full_rebuild'] = {'init': rf[0]['latency']['mean'], 'final': rf[-1]['latency']['mean'],
                           'growth': rf[-1]['latency']['mean']/rf[0]['latency']['mean'],
                           'final_recall': rf[-1]['recall'],
                           'rebuilds': rbf, 'cost': rtf}

    print(f"\n  {'方案':<18} {'init(ms)':<10} {'final(ms)':<10} {'growth':<8} {'recall':<8} {'#rb':<6} {'cost(s)':<8}")
    print("  " + "-"*66)
    for k, v in out.items():
        rec_str = f"{v['final_recall']:.4f}" if v['final_recall'] >= 0 else "N/A"
        print(f"  {k:<18} {v['init']:<10.3f} {v['final']:<10.3f} {v['growth']:<7.2f}x {rec_str:<8} "
              f"{v['rebuilds']:<6} {v['cost']:<8.2f}")
    save(out, "exp_e_extreme.json")

# ============================================================
# 实验 F: Rebuild 耗时 vs 规模
# ============================================================
def exp_f():
    sep("实验 F: Rebuild 耗时 vs 索引规模")
    base, queries = load_sift(max_base=500000)
    dim = 128
    out = []

    for n in [5000, 10000, 20000, 50000, 100000, 200000]:
        if n > len(base):
            break
        data = base[:n]
        times = []
        for _ in range(3):
            idx = hnswlib.Index(space='l2', dim=dim)
            idx.init_index(max_elements=n, M=16, ef_construction=200)
            t0 = time.perf_counter()
            idx.add_items(data, list(range(n)))
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        out.append({'n': n, 'build_s': float(avg), 'build_ms': float(avg*1000)})
        print(f"  n={n:>7}: {avg*1000:.1f}ms ({avg:.3f}s)")

    save(out, "exp_f_rebuild_cost.json")

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  ElasticHNSW 补充实验套件 (修复版)")
    print(f"  结果目录: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 70)

    if not os.path.exists("./data/sift/sift_base.fvecs"):
        print("错误: 找不到 SIFT 数据! 请在 elastic_hnsw 目录下运行")
        sys.exit(1)

    t0 = time.time()

    for name, fn in [("A", exp_a), ("B", exp_b), ("C", exp_c),
                      ("D", exp_d), ("E", exp_e), ("F", exp_f)]:
        try:
            fn()
        except Exception as e:
            print(f"\n!!! 实验 {name} 出错: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  全部完成! 耗时: {(time.time()-t0)/60:.1f} 分钟")
    print(f"  结果在: {os.path.abspath(RESULTS_DIR)}/")
    print(f"{'=' * 70}")
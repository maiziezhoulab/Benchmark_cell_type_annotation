import tangram as tg
import scanpy as sc
import numpy as np
import pandas as pd
import os
import time
import threading
import psutil
import gc

# -----------------------------
# Memory + time monitor
# -----------------------------
def run_with_monitor(func, interval=0.05):
    """
    Returns:
      result, runtime_sec, runtime_min, peak_rss_mib_inc, peak_rss_mib_abs
    """
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss
    peak_rss = rss_start
    stop_flag = False

    def monitor():
        nonlocal peak_rss
        while not stop_flag:
            rss_now = proc.memory_info().rss
            if rss_now > peak_rss:
                peak_rss = rss_now
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t0 = time.perf_counter()
    t.start()
    try:
        result = func()
    finally:
        stop_flag = True
        t.join()
    runtime_sec = time.perf_counter() - t0

    peak_rss_mib_inc = (peak_rss - rss_start) / (1024 ** 2)   # increase during run
    peak_rss_mib_abs = peak_rss / (1024 ** 2)                 # absolute process peak
    return result, runtime_sec, runtime_sec / 60.0, peak_rss_mib_inc, peak_rss_mib_abs


# -----------------------------
# Input data
# -----------------------------
rna_path = '/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad'
rna = sc.read_h5ad(rna_path)

# Get all slice IDs
unique_section = rna.obs['Section ID'].unique()
selected_0503 = [s for s in unique_section if s.startswith('0503')]
selected_0503_clean = [s for s in selected_0503 if s != "0503_nan_nan"]
selected_0503_1 = [s for s in selected_0503_clean if s != "0503_F4_C"]

# Output base path
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/0503_F4_C_output"
os.makedirs(outdir, exist_ok=True)

# Keep a raw query, copy fresh each loop to avoid repeated normalize/log1p on same object
query_data_raw = rna[rna.obs['Section ID'] == '0503_M4_S'].copy()

# Collect benchmark stats
bench_rows = []

# for k in range(2, len(selected_0503_1)+1):
for k in range(1, 18):
    subset_ids = selected_0503_clean[:k]
    print(f"\n>>> Running Tangram for first {k} sections: {subset_ids}")

    # Build data each loop
    total_data = rna[rna.obs['Section ID'].isin(subset_ids)].copy()
    reference_data = total_data
    query_data = query_data_raw.copy()

    sc.pp.normalize_total(query_data)
    sc.pp.log1p(query_data)

    sc.pp.normalize_total(reference_data)
    sc.pp.log1p(reference_data)

    # Harmonize genes
    tg.pp_adatas(reference_data, query_data, genes=None)
    def one_run():
        # Preprocessing
        tg_map = tg.map_cells_to_space(
            reference_data,
            query_data,
            density_prior='uniform',
            device='cpu'
        )

        # Project cell type annotation
        tg.project_cell_annotations(
            adata_sp=query_data,
            adata_map=tg_map,
            annotation='MERFISH cell type annotation'
        )

        return tg_map

    # Run with monitor
    tg_map, runtime_sec, runtime_min, peak_inc_mib, peak_abs_mib = run_with_monitor(one_run, interval=0.05)

    # Save predictions
    df = pd.DataFrame(query_data.obsm['tangram_ct_pred'], index=query_data.obs_names)
    outfile = os.path.join(outdir, f"tangram_ct_pred_first{k}_slices.csv")
    df.to_csv(outfile)
    print(f"Saved result: {outfile}")

    # Log metrics
    print(f"[k={k}] Runtime: {runtime_sec:.2f} s ({runtime_min:.2f} min)")
    print(f"[k={k}] Peak RSS increase: {peak_inc_mib:.2f} MiB")
    print(f"[k={k}] Absolute peak RSS: {peak_abs_mib:.2f} MiB")

    bench_rows.append({
        "k": k,
        "n_reference_cells": reference_data.n_obs,
        "n_query_cells": query_data.n_obs,
        "runtime_sec": runtime_sec,
        "runtime_min": runtime_min,
        "peak_rss_mib_increase": peak_inc_mib,
        "peak_rss_mib_absolute": peak_abs_mib,
        "output_csv": outfile
    })

    # Optional cleanup between loops
    del tg_map, df, query_data, reference_data, total_data
    gc.collect()

# Save benchmark summary
bench_df = pd.DataFrame(bench_rows)
bench_file = os.path.join(outdir, "tangram_runtime_memory_summary.csv")
bench_df.to_csv(bench_file, index=False)
print(f"\nSaved benchmark summary: {bench_file}")

import os
import sys
import time
import threading
import psutil
import pandas as pd
import scanpy as sc
import torch

# -----------------------------
# Force CPU execution
# -----------------------------
torch.cuda.is_available = lambda: False

# -----------------------------
# Import spatialID
# -----------------------------
sys.path.append('/maiziezhou_lab2/yuling/label_Transfer/spatialID/SpatialID/spatialid')
from spatialid.transfer import Transfer

# -----------------------------
# Benchmark settings
# -----------------------------
query = ['L', 'R']
time_point = ['Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6']

records = []

for tp in time_point:
    # ---------- load reference (NOT timed) ----------
    sc_adata = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Kidney/snRNA_cleaned.h5ad')
    sc_adata.layers['counts'] = sc_adata.X
    sc_adata.obs['tech'] = 'sc'

    for side in query:
        print(f"\n>>> Running SpatialID | {tp} {side}")

        outdir = f"/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/Kidney_all/{tp}{side}/"
        os.makedirs(outdir, exist_ok=True)

        # ---------- query preprocessing (NOT timed) ----------
        query_data = sc.read_h5ad(
            f"/maiziezhou_lab2/yuling/Datasets/Kidney/Xenium/time_{tp}{side}.h5ad"
        )
        query_data.obs['tech'] = 'st'
        query_data.layers['counts'] = query_data.X
        query_data.write(os.path.join(outdir, "spatial_data.h5ad"))

        single_path = os.path.join(outdir, "single_cell_data.h5ad")
        sc_adata.write(single_path)

        transfer_tool = Transfer(
            spatial_data=os.path.join(outdir, "spatial_data.h5ad"),
            single_data=single_path,
            output_path=outdir,
            device=0
        )

        # =============================
        # TIMED MODEL SECTION
        # =============================
        process = psutil.Process(os.getpid())
        peak_mem = [0]          # <- mutable container
        sampling = [True]

        def memory_sampler():
            while sampling[0]:
                try:
                    mem = process.memory_info().rss
                    peak_mem[0] = max(peak_mem[0], mem)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)

        mem_thread = threading.Thread(target=memory_sampler)
        mem_thread.start()

        t0 = time.perf_counter()

        # ---- Stage 1 ----
        transfer_tool.learn_sc(
            filter_mt=True,
            min_cell=0,
            min_gene=0,
            max_cell=98.0,
            ann_key="name",
            batch_size=409,
            epoch=200,
            lr=3e-4
        )

        # ---- Stage 2 ----
        transfer_tool.sc2st()

        # ---- Stage 3 ----
        transfer_tool.annotation(
            pca_dim=200,
            n_neigh=30,
            epochs=200,
            lr=0.01,
            show_results=False
        )

        t1 = time.perf_counter()

        sampling[0] = False
        mem_thread.join()

        records.append({
            "method": "SpatialID",
            "time_point": tp,
            "side": side,
            "runtime_sec": t1 - t0,
            "peak_memory_GB": peak_mem[0] / (1024 ** 3)
        })

        print(
            f"Time: {t1 - t0:.2f} sec | "
            f"Peak RSS: {peak_mem[0] / (1024 ** 3):.2f} GB"
        )

# -----------------------------
# Save results
# -----------------------------
df = pd.DataFrame(records)
df.to_csv(
    "/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/Kidney/dnn_time_memory.csv",
    index=False
)

print("\nBenchmark completed successfully.")

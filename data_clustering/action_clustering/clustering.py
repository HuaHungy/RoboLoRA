"""Candidate comparison with full provenance and representative windows."""

import json
from pathlib import Path
from time import monotonic

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from .config import fresh_dir, write_json
from .data import read_index
from .learning import metadata
from .progress import timed


def describe(z, labels, rows, train_ids, representative_count):
    sizes, representatives, centers = {}, {}, {}
    for label in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == label)
        sizes[str(label)] = len(ids)
        if label < 0:
            continue
        fit_ids = train_ids[labels[train_ids] == label]
        if len(fit_ids) == 0:
            continue
        center = z[fit_ids].mean(axis=0)
        centers[str(label)] = center.tolist()
        # Representative selection never uses held-out windows to define centroids.
        distances = np.linalg.norm(z[ids] - center, axis=1)
        chosen = np.argsort(distances)[:representative_count]
        representatives[str(label)] = [{**rows[int(ids[i])], "distance_to_mean": float(distances[i])}
                                       for i in chosen]
    return sizes, representatives, centers


def score_sample(z, labels, ids, seed, sample_size):
    # Keep noise out of silhouette calculations; no score for degenerate samples.
    ids = ids[labels[ids] >= 0]
    if len(ids) > sample_size:
        ids = np.random.default_rng(seed).choice(ids, sample_size, replace=False)
    unique = np.unique(labels[ids])
    if len(unique) < 2 or len(unique) >= len(ids):
        return None
    return float(silhouette_score(z[ids], labels[ids], metric="euclidean"))


def export_result(out, name, z, labels, rows, train_ids, settings, metrics, centers=None):
    folder = fresh_dir(out / name)
    np.save(folder / "labels.npy", labels)
    sizes, representatives, means = describe(z, labels, rows, train_ids, settings["representatives"])
    write_json(folder / "cluster_means.json", means)
    if centers is not None:
        np.save(folder / "centers.npy", centers)
    write_json(folder / "representatives.json", representatives)
    with (folder / "assignments.jsonl").open("w", encoding="utf-8") as f:
        for row, label in zip(rows, labels):
            f.write(json.dumps({**row, "cluster_id": int(label)}, ensure_ascii=False) + "\n")
    metrics.update({"cluster_sizes": sizes, "noise_fraction": float(np.mean(labels < 0)),
                    "occupied_clusters": len(set(labels.tolist()) - {-1})})
    write_json(folder / "metrics.json", metrics)
    return metrics


def cluster(cfg):
    folder, meta = metadata(cfg)
    root = Path(cfg["output_dir"])
    embedding_meta = json.loads((root / "embeddings/metadata.json").read_text())
    if embedding_meta["index_sha256"] != meta["index_sha256"]:
        raise ValueError("Embeddings do not match the current window index")
    z = np.load(root / "embeddings/z.npy", mmap_mode="r")
    rows = read_index(folder)
    if len(z) != len(rows) or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid embeddings")
    train_ids = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val_ids = np.array([i for i, r in enumerate(rows) if r["split"] == "validation"])
    settings = cfg["clustering"]
    out = fresh_dir(root / "clusters")
    summary = {}
    previous_cost = None
    with threadpool_limits(limits=cfg["training"]["threads"]):
        for candidate, k in enumerate(settings["k_values"], 1):
            name = f"kmeans_k{k}"
            if len(train_ids) < k:
                summary[name] = {"status": "skipped", "reason": f"only {len(train_ids)} training windows"}
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"])
            started = monotonic()
            estimate = previous_cost * k if previous_cost is not None else None
            with timed(f"K-means {candidate}/{len(settings['k_values'])}：K={k}, n_init=10", estimate):
                km.fit(z[train_ids])
            previous_cost = (monotonic() - started) / k
            print(f"[{name}] 计算验证指标并导出代表窗口", flush=True)
            labels = km.predict(z)
            metrics = {"fit_split": "train", "training_inertia": float(km.inertia_),
                       "validation_mean_squared_distance": float(np.mean(np.sum((z[val_ids] - km.cluster_centers_[labels[val_ids]])**2, axis=1))),
                       "validation_silhouette": score_sample(z, labels, val_ids, cfg["seed"], settings["silhouette_sample_size"])}
            summary[name] = export_result(out, name, z, labels, rows, train_ids, settings, metrics, km.cluster_centers_)
            print(f"{name}: {summary[name]['occupied_clusters']} occupied clusters", flush=True)
        if settings["hdbscan"]:
            from sklearn.cluster import HDBSCAN

            if len(train_ids) < settings["min_cluster_size"]:
                summary["hdbscan"] = {"status": "skipped", "reason": "too few training windows"}
            else:
                estimator = HDBSCAN(min_cluster_size=settings["min_cluster_size"], n_jobs=cfg["training"]["threads"])
                with timed("HDBSCAN 拟合"):
                    train_labels = estimator.fit_predict(z[train_ids])
                labels = np.full(len(z), -1, dtype=np.int64)
                labels[train_ids] = train_labels
                # sklearn HDBSCAN has no held-out predict. Do not fabricate it.
                selected_rows = [rows[i] for i in train_ids]
                metrics = {"fit_split": "train", "assignment_scope": "train_only; validation not predicted",
                           "training_silhouette": score_sample(z, labels, train_ids, cfg["seed"], settings["silhouette_sample_size"])}
                summary["hdbscan"] = export_result(out, "hdbscan", z[train_ids], train_labels, selected_rows,
                                                   np.arange(len(train_ids)), settings, metrics)
                np.save(out / "hdbscan/probabilities.npy", estimator.probabilities_)
                np.save(out / "hdbscan/window_ids.npy", train_ids)
    write_json(out / "comparison.json", summary)
    print("[聚类] 生成 PCA 图和结果汇总", flush=True)
    # PCA is a diagnostic, never used to calculate cluster assignments.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = np.sort(np.random.default_rng(cfg["seed"]).choice(len(z), min(len(z), 10000), replace=False))
    if min(len(train_ids), z.shape[1]) >= 2:
        pca = PCA(n_components=2, random_state=cfg["seed"]).fit(z[train_ids[:10000]])
        xy = pca.transform(z[sample])
        for name in summary:
            if not name.startswith("kmeans") or "status" in summary[name]:
                continue
            labels = np.load(out / name / "labels.npy")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels[sample], cmap="tab20", s=10, alpha=0.7)
            ax.set(xlabel="PC1", ylabel="PC2", title=name + " | latent PCA (diagnostic only)")
            fig.colorbar(scatter, ax=ax, label="Cluster ID")
            fig.tight_layout()
            fig.savefig(out / name / "latent_pca.png", dpi=150)
            plt.close(fig)
    print(f"Saved candidate comparison and representative provenance -> {out}")
    return summary

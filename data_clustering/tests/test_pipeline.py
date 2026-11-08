import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from action_clustering.__main__ import demo
from action_clustering.config import load_config
from action_clustering.data import discover, load_episode, prepare, read_index, window_features
from action_clustering.learning import fit_statistics, fit, embed
from action_clustering.clustering import cluster


class FeatureTests(unittest.TestCase):
    def test_ee_invariant_to_common_world_transform_and_quaternion_sign(self):
        rng = np.random.default_rng(5)
        length = 50
        pose = np.zeros((length + 1, 2, 7), dtype=np.float32)
        pose[:, :, :3] = rng.normal(size=(length + 1, 2, 3))
        pose[:, :, 3:] = Rotation.random((length + 1) * 2, random_state=rng).as_quat().reshape(length + 1, 2, 4)
        ep = {"ee_pose": pose, "gripper": rng.uniform(size=(length + 1, 2)).astype(np.float32)}
        before = window_features(ep, 0, length, "ee", {})
        transform = Rotation.from_rotvec([0.2, -0.4, 0.3])
        moved = pose.copy()
        moved[:, :, :3] = transform.apply(pose[:, :, :3].reshape(-1, 3)).reshape(length + 1, 2, 3) + [3, 4, -2]
        moved[:, :, 3:] = (transform * Rotation.from_quat(pose[:, :, 3:].reshape(-1, 4))).as_quat().reshape(length + 1, 2, 4)
        moved[::2, :, 3:] *= -1
        after = window_features({**ep, "ee_pose": moved}, 0, length, "ee", {})
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=2e-6)

    def test_hdf5_does_not_invent_terminal_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            q = np.arange(12, dtype=np.float32).reshape(4, 3)
            with h5py.File(path, "w") as f:
                f["observations/qpos"] = q
                f["action"] = q + 99
            ep = load_episode(path, {"format": "aloha_hdf5", "gripper_indices": [2]})
            self.assertEqual(len(ep["actions"]), 3)
            np.testing.assert_array_equal(ep["qpos"][-1], q[-1])

    def test_joint_mode_rejects_mixed_embodiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(demo(Path(tmp) / "demo"))
            cfg["datasets"].append({**cfg["datasets"][0], "id": "other", "robot_id": "different"})
            with self.assertRaisesRegex(ValueError, "embodiments"):
                discover(cfg)

    def test_mixed_sampling_rates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(demo(Path(tmp) / "demo"))
            cfg["datasets"].append({**cfg["datasets"][0], "id": "other", "fps": 50})
            with self.assertRaisesRegex(ValueError, "sampling rate"):
                discover(cfg)


class PipelineTests(unittest.TestCase):
    def test_ee_and_cached_visual_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(demo(Path(tmp) / "demo"))
            cfg["representation"] = {"mode": "ee", "visual_weight": 0.1}
            cfg["training"]["epochs"] = 1
            with self.assertRaisesRegex(ValueError, "visual_cache"):
                prepare(cfg)
            cache = Path(tmp) / "visual"
            cache.mkdir()
            source = cfg["datasets"][0]
            source["visual_cache"] = str(cache)
            rng = np.random.default_rng(0)
            for path in Path(source["root"]).glob("*.npz"):
                # Synthetic features only for verifying the optional input branch.
                np.save(cache / path.with_suffix(".npy").name, rng.normal(size=(181, 8)).astype(np.float32))
            prepare(cfg)
            fit(cfg)
            embed(cfg)
            root = Path(cfg["output_dir"])
            meta = json.loads((root / "windows/metadata.json").read_text())
            self.assertEqual(meta["visual_dim"], 8)
            self.assertEqual(meta["action_dim"], 16)
            z = np.load(root / "embeddings/z.npy")
            np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)

    def test_complete_pipeline_split_normalization_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(demo(Path(tmp) / "demo"))
            cfg["training"]["epochs"] = 1
            cfg["clustering"]["k_values"] = [3, 999]
            prepare(cfg)
            root = Path(cfg["output_dir"])
            rows = read_index(root / "windows")
            train = {r["episode_id"] for r in rows if r["split"] == "train"}
            val = {r["episode_id"] for r in rows if r["split"] == "validation"}
            self.assertFalse(train & val)
            train_ids = [r["window_id"] for r in rows if r["split"] == "train"]
            val_ids = [r["window_id"] for r in rows if r["split"] == "validation"]
            stats = fit_statistics(root / "windows", train_ids)
            array = np.load(root / "windows/action.npy", mmap_mode="r+")
            array[val_ids] += 1000
            array.flush()
            after = fit_statistics(root / "windows", train_ids)
            for key in stats:
                np.testing.assert_array_equal(stats[key], after[key])
            array[val_ids] -= 1000
            array.flush()
            fit(cfg)
            embed(cfg)
            result = cluster(cfg)
            z = np.load(root / "embeddings/z.npy")
            np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
            self.assertEqual(len(z), len(rows))
            self.assertEqual(result["kmeans_k999"]["status"], "skipped")
            assignments = [json.loads(line) for line in (root / "clusters/kmeans_k3/assignments.jsonl").read_text().splitlines()]
            self.assertEqual([r["window_id"] for r in assignments], list(range(len(rows))))
            self.assertTrue(all(0 <= r["cluster_id"] < 3 for r in assignments))
            hdb = [json.loads(line) for line in (root / "clusters/hdbscan/assignments.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["split"] == "train" for r in hdb))
            self.assertEqual(len(hdb), len(train_ids))
            self.assertTrue((root / "clusters/kmeans_k3/latent_pca.png").exists())


if __name__ == "__main__":
    unittest.main()

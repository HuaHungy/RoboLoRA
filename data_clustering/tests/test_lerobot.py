import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from action_clustering.config import load_config
from action_clustering.data import load_episode, prepare, read_index
from action_clustering.lerobot import read_columns


def fixture(root):
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    sources = []
    states = []
    for e in range(2):
        full = np.zeros((101, 16), dtype=np.float32)
        full[:, 0] = np.arange(101) * 0.01 + e
        full[:, 3] = full[:, 6] = np.sqrt(0.5)
        full[:, 11] = 1
        full[:, 7] = full[:, 15] = np.linspace(0, 1, 101)
        path = root / f"data/chunk-000/episode_{e:06d}.parquet"
        pq.write_table(pa.table({"observation.state": full[:-1].tolist(), "action": full[1:].tolist()}), path)
        sources.append({"episode_index": e, "frames": 100, "source_path": f"/native/{e}.hdf5", "instruction": "move"})
        states.append(full)
    (root / "meta/info.json").write_text(json.dumps({"fps": 250 / 15}))
    (root / "meta/robotwin_ee_conversion.json").write_text(json.dumps({
        "status": "complete", "quaternion_order": "wxyz", "representation": "absolute_world_ee_pose", "sources": sources}))
    return states


class LeRobotTests(unittest.TestCase):
    def test_quaternion_reordering_and_real_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            full = fixture(root)[0]
            source = {"format": "lerobot_ee", "root": str(root), "fps": 250 / 15}
            path = root / "data/chunk-000/episode_000000.parquet"
            ep = load_episode(path, source)
            self.assertEqual(ep["ee_pose"].shape, (101, 2, 7))
            self.assertEqual(ep["actions"].shape, (100, 16))
            np.testing.assert_array_equal(ep["actions"], full[1:])
            np.testing.assert_array_equal(ep["ee_pose"][:, 0], full[:, [0, 1, 2, 4, 5, 6, 3]])
            np.testing.assert_array_equal(ep["gripper"][-1], full[-1, [7, 15]])
            with self.assertRaisesRegex(ValueError, "fps"):
                load_episode(path, {**source, "fps": 25})
            broken = full[1:].copy()
            broken[0, 0] += 1
            pq.write_table(pa.table({"observation.state": full[:-1].tolist(), "action": broken.tolist()}), path)
            with self.assertRaisesRegex(ValueError, "next recorded state"):
                read_columns(path)

    def test_split_per_dataset_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = []
            for name in ("a", "b"):
                folder = root / name
                fixture(folder)
                sources.append({"id": name, "root": str(folder), "format": "lerobot_ee",
                                "glob": "data/chunk-*/*.parquet", "fps": 250 / 15, "robot_id": "aloha"})
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"datasets": sources, "output_dir": str(root / "run"),
                                              "representation": {"mode": "ee"},
                                              "training": {"stratify_by_dataset": True}}))
            meta = prepare(load_config(config_path))
            self.assertEqual((meta["action_dim"], meta["state_dim"], meta["visual_dim"]), (16, 14, 0))
            rows = read_index(root / "run/windows")
            for name in ("a", "b"):
                splits = {split: {r["episode_id"] for r in rows if r["dataset_id"] == name and r["split"] == split}
                          for split in ("train", "validation")}
                self.assertEqual(len(splits["train"]), 1)
                self.assertEqual(len(splits["validation"]), 1)
                self.assertFalse(splits["train"] & splits["validation"])
            self.assertTrue(all(r["instruction"] == "move" and "native_source_path" in r for r in rows))


if __name__ == "__main__":
    unittest.main()

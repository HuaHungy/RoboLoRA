import os
import sys
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from script.common import data_loader
from script.common import pi0_forward
from script import compose
from vla_lorahub import algorithm


class DataLogicTest(unittest.TestCase):
    def test_pi0_tail_action_indices_repeat_episode_end(self):
        indices = data_loader.make_padded_action_indices(7, 10, 6)
        np.testing.assert_array_equal(indices, [7, 8, 9, 9, 9, 9])

    def test_delta_action_windows_match_openpi_anchor_and_tail_padding(self):
        states = np.zeros((3, 14), dtype=np.float32)
        states[:, 0] = [1.0, 2.0, 3.0]
        actions = np.zeros((3, 14), dtype=np.float32)
        actions[:, 0] = [1.5, 3.0, 5.0]
        actions[:, 6] = [0.0, 1.0, 1.0]
        windows = data_loader.make_delta_action_windows(
            states, actions, action_horizon=4, action_dim=14)
        np.testing.assert_allclose(windows[1, :, 0], [1.0, 3.0, 3.0, 3.0])
        np.testing.assert_allclose(windows[1, :, 6], [1.0, 1.0, 1.0, 1.0])

    def test_random_batch_is_shared_within_cma_generation(self):
        class FakeDataset:
            def __len__(self):
                return 100

            def get_batch(self, indices):
                values = [int(x) for x in indices]
                return values, values

        sampler = data_loader.Pi0RandomBatchSampler(
            FakeDataset(), batch_size=4, reuse_candidates=3, seed=7)
        first = sampler.get_batch(0)
        self.assertIs(first, sampler.get_batch(2))
        second = sampler.get_batch(3)
        self.assertNotEqual(first[2]["dataset_indices"],
                            second[2]["dataset_indices"])
        self.assertEqual(first[2]["num_samples"], 4)

    def test_motion_keyframes_cover_all_temporal_phases(self):
        n_frames, horizon = 90, 10
        states = np.zeros((n_frames, 14), dtype=np.float32)
        actions = np.zeros((n_frames, 14), dtype=np.float32)
        # Put a distinct motion spike in each quarter of the valid start range.
        for index in (5, 25, 45, 70):
            actions[index, 0] = 1.0
        starts = data_loader.select_keyframe_starts(
            states, actions, horizon, 4, "motion_stratified")
        self.assertEqual(starts.shape, (4,))
        self.assertTrue(np.all(np.diff(starts) > 0))
        self.assertTrue(np.all(starts <= n_frames - horizon))

    def test_uniform_starts_cover_full_episode_with_complete_chunks(self):
        starts = data_loader.select_uniform_frame_starts(
            episode_length=200, action_horizon=50, frames_per_ep=5)
        np.testing.assert_array_equal(starts, [0, 38, 75, 112, 150])
        self.assertTrue(np.all(starts <= 200 - 50))

    def test_delta_stats_floor_marks_inactive_dimensions(self):
        output = np.zeros((5, 14), dtype=np.float32)
        output[:, 0] = np.arange(5, dtype=np.float32)
        stats = data_loader.compute_delta_loss_stats([output], std_floor=0.1)
        self.assertTrue(stats["active"][0])
        self.assertFalse(stats["active"][1])
        self.assertAlmostEqual(float(stats["scale"][1]), 0.1, places=6)


class LossLogicTest(unittest.TestCase):
    def test_grouped_loss_uses_requested_joint_gripper_weights(self):
        gt = np.zeros((2, 14), dtype=np.float32)
        pred = np.zeros_like(gt)
        pred[:, 0] = 1.0
        pred[:, 6] = 1.0
        stats = {
            "scale": np.ones(14, dtype=np.float32),
            "active": np.ones(14, dtype=bool),
        }
        loss = algorithm.delta_grouped_vla_loss(
            {}, [{}], [gt], forward_fn=lambda _p, _x: [pred],
            delta_stats=stats, joint_weight=0.8, gripper_weight=0.2,
            huber_delta=1.0)
        # Joint group averages one affected dimension over 12; gripper over 2.
        expected = 0.8 * (0.5 / 12.0) + 0.2 * (0.5 / 2.0)
        self.assertAlmostEqual(loss, expected, places=7)


class LoraCompositionTest(unittest.TestCase):
    def test_v25_searches_exactly_three_groups(self):
        groups = algorithm.get_v2_search_group_ids("vision_action_focused")
        self.assertEqual(groups, [0, 1, 3])
        self.assertEqual(len(groups) * 15, 45)

    def test_v3_dimension_is_ten_times_module_count(self):
        self.assertEqual(len(algorithm.LAYER_GROUPS_V3), 10)
        self.assertEqual(len(algorithm.LAYER_GROUPS_V3) * 15, 150)

    def test_v3_uses_generation_batches_and_independent_final_batch(self):
        class FakeSampler:
            dataset = range(100)
            batch_size = 1
            batches_per_evaluation = 1

            def __init__(self):
                self.calls = []
                self.final_calls = 0

            def get_batch(self, evaluation_index):
                self.calls.append(evaluation_index)
                return ([evaluation_index], [0.0], {
                    "generation": evaluation_index,
                    "num_samples": 1,
                })

            def get_final_batch(self):
                self.final_calls += 1
                return ([99], [0.0], {
                    "generation": -1,
                    "num_samples": 1,
                    "final": True,
                })

        modules = ["m0", "m1"]
        cache = {
            name: {"params/x/q_einsum/lora_a/value": np.ones((1, 1))}
            for name in modules
        }
        sampler = FakeSampler()
        with mock.patch.object(
                algorithm, "load_base_params_from_checkpoint", return_value={}), \
             mock.patch.object(
                algorithm, "load_lora_modules", return_value=(cache, modules)):
            weights, _, _ = algorithm.vla_lorahub_learning_v3(
                modules,
                example_inputs=[-1],
                example_outputs=[0.0],
                base_checkpoint_dir="unused",
                max_inference_step=2,
                get_loss=lambda _p, inputs, _o, _f: float(inputs[0]),
                get_regular=lambda _w: 0.0,
                seed=7,
                logit_bound=1.0,
                cma_scale=0.5,
                cma_popsize=2,
                batch_sampler=sampler,
            )

        self.assertEqual(weights.shape, (10, 2))
        self.assertEqual(sampler.calls, [0, 1])
        self.assertEqual(sampler.final_calls, 1)
        self.assertEqual(
            [item["batch_generation"] for item in algorithm.SCORE_HISTORY["v3"]],
            [0, 1, -1],
        )
        self.assertEqual(
            [item["kind"] for item in algorithm.SCORE_HISTORY["v3"]],
            ["search", "search", "validation"],
        )

    def test_v3_writes_periodic_progress_and_resumes_checkpoint(self):
        modules = ["m0", "m1"]
        cache = {
            name: {"params/x/q_einsum/lora_a/value": np.ones((1, 1))}
            for name in modules
        }

        class FakeSampler:
            dataset = range(100)
            batch_size = 1
            batches_per_evaluation = 1

            def get_batch(self, evaluation_index):
                return ([evaluation_index], [0.0], {
                    "generation": evaluation_index,
                    "num_samples": 1,
                })

            def get_final_batch(self):
                return ([99], [0.0], {
                    "generation": -1,
                    "num_samples": 1,
                    "final": True,
                })

        with tempfile.TemporaryDirectory() as progress_dir:
            calls = {"count": 0}

            def interrupting_loss(_params, _inputs, _outputs, _forward):
                calls["count"] += 1
                if calls["count"] == 3:
                    raise RuntimeError("simulated interruption")
                return 0.0

            with mock.patch.object(
                    algorithm, "load_base_params_from_checkpoint", return_value={}), \
                 mock.patch.object(
                    algorithm, "load_lora_modules", return_value=(cache, modules)):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    algorithm.vla_lorahub_learning_v3(
                        modules, [-1], [0.0], "unused", max_inference_step=4,
                        get_loss=interrupting_loss, get_regular=lambda _w: 0.0,
                        seed=7, logit_bound=1.0, cma_scale=0.5,
                        cma_popsize=2, batch_sampler=FakeSampler(),
                        progress_dir=progress_dir, weight_update_interval=1,
                        checkpoint_interval=2,
                    )

                weight_path = os.path.join(progress_dir, "lorahub_weights.json")
                checkpoint_path = os.path.join(
                    progress_dir, "optimization_checkpoints", "step_000000002.pkl")
                self.assertTrue(os.path.exists(weight_path))
                self.assertTrue(os.path.exists(checkpoint_path))
                with open(weight_path) as f:
                    self.assertEqual(json.load(f)["step"], 2)

                weights, _, _ = algorithm.vla_lorahub_learning_v3(
                    modules, [-1], [0.0], "unused", max_inference_step=4,
                    get_loss=lambda _p, _i, _o, _f: 0.0,
                    get_regular=lambda _w: 0.0, seed=7, logit_bound=1.0,
                    cma_scale=0.5, cma_popsize=2, batch_sampler=FakeSampler(),
                    progress_dir=progress_dir, weight_update_interval=1,
                    checkpoint_interval=2, resume=True,
                )

            self.assertEqual(weights.shape, (10, 2))
            with open(weight_path) as f:
                progress = json.load(f)
            self.assertEqual(progress["step"], 4)
            self.assertEqual(progress["status"], "in_progress")

    def setUp(self):
        self.modules = ["m0", "m1"]
        self.a_key = "params/x/lora_a/value"
        self.b_key = "params/x/lora_b/value"
        self.cache = {
            "m0": {
                self.a_key: np.array([[1.0], [2.0]], dtype=np.float32),
                self.b_key: np.array([[3.0, 4.0]], dtype=np.float32),
            },
            "m1": {
                self.a_key: np.array([[2.0], [-1.0]], dtype=np.float32),
                self.b_key: np.array([[1.0, 5.0]], dtype=np.float32),
            },
        }

    def test_delta_concat_matches_sum_of_adapter_deltas(self):
        weights = np.array([0.25, 0.75], dtype=np.float32)
        mixed = algorithm.compose_lora_weights(
            self.cache, weights, self.modules, composition_mode="delta_concat")
        a = mixed["x/lora_a/value"]
        b = mixed["x/lora_b/value"]
        actual = a @ b
        expected = sum(
            weights[i] * self.cache[name][self.a_key] @ self.cache[name][self.b_key]
            for i, name in enumerate(self.modules))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(a.shape[-1], 2)
        self.assertEqual(b.shape[-2], 2)

    def test_delta_sketch_preserves_rank_and_is_exact_for_one_hot(self):
        mixed = algorithm.compose_lora_weights(
            self.cache, np.array([1.0, 0.0], dtype=np.float32), self.modules,
            composition_mode="delta_sketch")
        a = mixed["x/lora_a/value"]
        b = mixed["x/lora_b/value"]
        np.testing.assert_allclose(
            a @ b, self.cache["m0"][self.a_key] @ self.cache["m0"][self.b_key])
        self.assertEqual(a.shape, self.cache["m0"][self.a_key].shape)
        self.assertEqual(b.shape, self.cache["m0"][self.b_key].shape)

    def test_exact_delta_export_merges_base_and_keeps_native_rank_zeros(self):
        exact = algorithm.compose_lora_weights(
            self.cache, np.array([0.25, 0.75], dtype=np.float32), self.modules,
            composition_mode="delta_concat")
        exact_flat = {compose.norm_key(key): value for key, value in exact.items()}
        tree = {"params": {"x": {"w": np.ones((2, 2), dtype=np.float32)}}}
        zero_native = {
            "x/lora_a": np.zeros_like(self.cache["m0"][self.a_key]),
            "x/lora_b": np.zeros_like(self.cache["m0"][self.b_key]),
        }
        result = compose.merge_exact_delta_into_base(
            tree, exact_flat, zero_native)
        expected_delta = sum(
            weight * self.cache[name][self.a_key] @ self.cache[name][self.b_key]
            for weight, name in zip((0.25, 0.75), self.modules)
        )
        np.testing.assert_allclose(
            result["params"]["x"]["w"], 1 + expected_delta,
            rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            result["params"]["x"]["lora_a"], zero_native["x/lora_a"])
        self.assertEqual(result["params"]["x"]["lora_a"].shape, (2, 1))

    def test_build_params_does_not_copy_base_arrays(self):
        base_array = np.ones((2, 2), dtype=np.float32)
        result = algorithm.build_composed_model_params(
            {"base": base_array}, {"x/lora_a/value": np.ones((1, 1))})
        self.assertIs(result["base"], base_array)

    def test_build_params_can_mark_only_expert_lora_as_mutable(self):
        composed = {
            "x/q_einsum/lora_a/value": np.ones((1, 1)),
            "x/q_einsum_1/lora_a/value": np.ones((1, 1)),
        }
        expert_key = "x/q_einsum_1/lora_a/value"
        result = algorithm.build_composed_model_params(
            {}, composed, update_lora_keys={expert_key})
        self.assertEqual(result["__lora_keys__"], set(composed))
        self.assertEqual(result["__lora_update_keys__"], {expert_key})

    def test_action_focused_group_marker_selects_only_expert_groups(self):
        keys = {
            "x/q_einsum/lora_a/value",
            "x/q_einsum_1/lora_a/value",
            "x/mlp/gating_einsum_lora_a/value",
            "x/mlp_1/gating_einsum_lora_a/value",
        }
        selected = {
            key for key in keys
            if algorithm._get_group_id_for_key(key) in {1, 3}
        }
        self.assertEqual(selected, {
            "x/q_einsum_1/lora_a/value",
            "x/mlp_1/gating_einsum_lora_a/value",
        })

    def test_partial_group_composition_matches_full_composition(self):
        cache = {
            "m0": {
                "params/x/q_einsum/lora_a/value": np.array([[1.0]]),
                "params/x/q_einsum/lora_b/value": np.array([[2.0]]),
                "params/x/q_einsum_1/lora_a/value": np.array([[3.0]]),
                "params/x/q_einsum_1/lora_b/value": np.array([[4.0]]),
            },
            "m1": {
                "params/x/q_einsum/lora_a/value": np.array([[5.0]]),
                "params/x/q_einsum/lora_b/value": np.array([[6.0]]),
                "params/x/q_einsum_1/lora_a/value": np.array([[7.0]]),
                "params/x/q_einsum_1/lora_b/value": np.array([[8.0]]),
            },
        }
        weights = np.array([
            [0.25, 0.75], [0.6, 0.4], [0.25, 0.75], [0.6, 0.4],
        ], dtype=np.float32)
        full = algorithm.compose_lora_weights_per_group(
            cache, weights, ["m0", "m1"], "action_delta_sketch")
        main = algorithm.compose_lora_weights_per_group(
            cache, weights, ["m0", "m1"], "action_delta_sketch", {0, 2})
        expert = algorithm.compose_lora_weights_per_group(
            cache, weights, ["m0", "m1"], "action_delta_sketch", {1, 3})
        split = {**main, **expert}
        self.assertEqual(set(full), set(split))
        for key in full:
            np.testing.assert_array_equal(full[key], split[key])


class PartialStateUpdateTest(unittest.TestCase):
    def test_jitted_sampler_observes_dynamic_lora_state(self):
        class TinySampler(nnx.Module):
            def __init__(self):
                self.lora_scale = nnx.Param(jnp.array(1.0))

            def sample_actions(self, _rng, observation, *, num_steps):
                return observation * self.lora_scale * num_steps

        graphdef, state = nnx.split(TinySampler())
        sample_jit = pi0_forward._make_sample_actions_jit(graphdef)
        rng = jnp.array([0, 0], dtype=jnp.uint32)
        first = sample_jit(state, rng, jnp.array([2.0]), num_steps=2)
        state.replace_by_pure_dict({"lora_scale": jnp.array(3.0)})
        third = sample_jit(state, rng, jnp.array([2.0]), num_steps=2)
        np.testing.assert_allclose(first, [4.0])
        np.testing.assert_allclose(third, [12.0])

    def test_partial_nnx_update_preserves_non_lora_value_object(self):
        class TinyModule(nnx.Module):
            def __init__(self):
                self.base = nnx.Param(jnp.ones((2, 2)))
                self.lora_a = nnx.Param(jnp.ones((2, 1)))

        _, state = nnx.split(TinyModule())
        before = pi0_forward._capture_non_lora_value_ids(state)
        state.replace_by_pure_dict({"lora_a": jnp.zeros((2, 1))})
        after = pi0_forward._capture_non_lora_value_ids(state)
        self.assertEqual(before, after)
        np.testing.assert_array_equal(state["lora_a"].value, np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()

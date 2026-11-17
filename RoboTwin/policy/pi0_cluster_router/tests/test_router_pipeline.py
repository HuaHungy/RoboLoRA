from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parents[1]
PI0 = HERE.parent / "pi0"
sys.path[:0] = [str(HERE), str(HERE.parent), str(PI0 / "src")]

from common import CLUSTERS, HERE as PACKAGE, build_config, read_settings
from prepare import bank_key, pair_count


class RouterPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = read_settings()

    def test_twenty_leaves_are_ten_pairs(self):
        module = np.load(
            PACKAGE.parents[2] / "LoRAHub-JAX/lora_modules/place_can_basket_pi0_lora.npz",
            allow_pickle=True)
        try:
            keys = [key for key in module.files if key != "__metadata__"]
            self.assertEqual(len(keys), 20)
            self.assertEqual(pair_count(keys), 10)
            self.assertEqual(len({bank_key(key) for key in keys}), 20)
        finally:
            module.close()

    def test_exact_weighted_delta_matches_expert_choice(self):
        from openpi.models import lora

        config = lora.LoRAConfig(
            rank=1, alpha=1, init_b_fn=nn.initializers.zeros,
            expert_bank_size=2)
        layer = lora.Einsum((2, 2), init_fn=nn.initializers.zeros, lora_config=config)
        variables = layer.init(
            jax.random.key(0), "BD,DF->BF", jnp.ones((2, 2)),
            jnp.asarray([[1.0, 0.0], [0.0, 1.0]]))
        params = variables["params"]
        params["lora_a"] = jnp.zeros_like(params["lora_a"])
        params["lora_b"] = jnp.zeros_like(params["lora_b"])
        params["lora_bank_a"] = jnp.asarray([[[1.0], [0.0]], [[0.0], [1.0]]])
        params["lora_bank_b"] = jnp.asarray([[[2.0, 0.0]], [[0.0, 3.0]]])
        output = layer.apply(
            variables, "BD,DF->BF", jnp.ones((2, 2)),
            jnp.asarray([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(output, [[2.0, 0.0], [0.0, 3.0]])

    def test_expert_bank_preserves_nonleading_batch_axis(self):
        """KV einsum writes B at axis 1; batch>1 must retain that layout."""
        from openpi.models import lora

        config = lora.LoRAConfig(
            rank=1, alpha=1, init_b_fn=nn.initializers.zeros,
            expert_bank_size=2, expert_top_k=1)
        layer = lora.Einsum(
            (2, 1, 3, 2), init_fn=nn.initializers.zeros,
            lora_config=config)
        x = jnp.ones((4, 5, 3), dtype=jnp.float32)
        routes = jnp.asarray([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        ])
        variables = layer.init(
            jax.random.key(0), "BSD,2KDH->2BSKH", x, routes)
        params = variables["params"]
        params["lora_a"] = jnp.zeros_like(params["lora_a"])
        params["lora_b"] = jnp.zeros_like(params["lora_b"])
        params["lora_bank_a"] = params["lora_bank_a"].at[0].set(1.0)
        params["lora_bank_a"] = params["lora_bank_a"].at[1].set(2.0)
        params["lora_bank_b"] = jnp.ones_like(params["lora_bank_b"])

        output = layer.apply(
            variables, "BSD,2KDH->2BSKH", x, routes)

        self.assertEqual(output.shape, (2, 4, 5, 1, 2))
        expected = np.asarray([3.0, 6.0, 3.0, 6.0])
        np.testing.assert_allclose(np.asarray(output[0, :, 0, 0, 0]), expected)

    def test_model_contract_and_trainable_scope(self):
        from openpi.shared.nnx_utils import PathRegex

        cfg = build_config(self.settings, HERE / "runs/_test")
        model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        state = nnx.state(model)
        trainable = jax.tree.leaves(state.filter(cfg.trainable_filter))
        bank = jax.tree.leaves(state.filter(
            nnx.All(nnx.Param, PathRegex(".*lora_bank.*"))))
        self.assertEqual(len(trainable), 34)  # 20 A/B + 10 eta + 4 router MLP leaves
        self.assertEqual(sum(x.size for x in trainable), 50_530_298)
        self.assertEqual(len(bank), 20)
        self.assertEqual(sum(x.size for x in bank), 7 * 49_987_584)
        self.assertEqual(cfg.model.lora_router_top_k, 3)
        self.assertEqual(tuple(cfg.policy_metadata["cluster_order"]), CLUSTERS)

    def test_topk_forward_and_dense_gradient(self):
        from openpi.models.pi0 import topk_router_weights

        logits = jnp.arange(140, dtype=jnp.float32).reshape(2, 10, 7)
        weights = topk_router_weights(logits, 3)
        np.testing.assert_allclose(np.asarray(weights.sum(-1)), 1.0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray((weights > 0).sum(-1)), 3)
        gradient = jax.grad(lambda value: jnp.sum(
            topk_router_weights(value, 3) * logits))(logits)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())

    def test_deploy_uses_routed_config_and_joint_normalization(self):
        from pi0_cluster_router.deploy_policy import RoutedEEPolicy

        fake_policy = object()
        with mock.patch("openpi.policies.policy_config.create_trained_policy", return_value=fake_policy) as create:
            deployed = RoutedEEPolicy(self.settings, HERE / "runs/_test", "/tmp/checkpoint")
        cfg = create.call_args.args[0]
        self.assertIs(deployed.policy, fake_policy)
        self.assertEqual(cfg.model.lora_expert_bank_size, len(CLUSTERS))
        self.assertEqual(cfg.model.lora_router_top_k, 3)
        self.assertEqual(create.call_args.kwargs["robotwin_repo_id"], "robotwin_five_pi0_ee")
        self.assertEqual(create.call_args.kwargs["sample_kwargs"], {"num_steps": 10})

    def test_native_evaluator_policy_api_is_exported(self):
        import pi0_cluster_router

        self.assertTrue(callable(pi0_cluster_router.get_model))
        self.assertTrue(callable(pi0_cluster_router.eval))
        self.assertTrue(callable(pi0_cluster_router.reset_model))

    def test_no_video_chunk_does_not_render_duplicate_observations(self):
        from pi0_cluster_router.deploy_policy import eval as evaluate_policy

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = {
            "endpose": {
                "left_endpose": [0, 0, 0, 1, 0, 0, 0], "left_gripper": 0.0,
                "right_endpose": [0, 0, 0, 1, 0, 0, 0], "right_gripper": 0.0,
            },
            "observation": {
                name: {"rgb": image} for name in ("head_camera", "left_camera", "right_camera")
            },
        }
        calls = {"actions": 0, "observations": 0}
        env = SimpleNamespace(
            eval_success=False, take_action_cnt=0, step_lim=400, eval_video_path=None,
            get_instruction=lambda: "adjust the bottle",
        )
        def take_action(action, action_type):
            self.assertEqual(action_type, "ee")
            calls["actions"] += 1
            env.take_action_cnt += 1
        def get_obs():
            calls["observations"] += 1
            return observation
        env.take_action = take_action
        env.get_obs = get_obs
        model = SimpleNamespace(
            previous_state=None, execute_steps=10, inference_calls=0,
            total_inference_seconds=0.0, total_action_seconds=0.0,
            policy=SimpleNamespace(infer=lambda inputs: {"actions": np.zeros((50, 16), dtype=np.float32)}),
        )
        evaluate_policy(env, model, observation)
        self.assertEqual(calls, {"actions": 10, "observations": 0})


if __name__ == "__main__":
    unittest.main()

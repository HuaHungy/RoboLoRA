import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from action_clustering.mlp_temporal import (episode_steps, chains_and_pairs, fit_gate, temporal_metrics,
                                           guarded_weights, sample_pairs, distance_contribution, calibrate_alpha)
from action_clustering.gripper_review import detect_events, select_events, bracket_event, event_scores


class MLPTemporalTests(unittest.TestCase):
    def test_body_frame_features_and_overlapping_physical_time(self):
        rng = np.random.default_rng(4)
        pose = np.zeros((81, 2, 7)); pose[:, :, :3] = rng.normal(size=(81, 2, 3))
        pose[:, :, 3:] = Rotation.random(162, random_state=rng).as_quat().reshape(81, 2, 4)
        grip = rng.random((81, 2))
        original = episode_steps({'ee_pose': pose, 'gripper': grip})
        transform = Rotation.from_rotvec([.2, -.5, .1]); moved = pose.copy()
        moved[:, :, :3] = transform.apply(pose[:, :, :3].reshape(-1, 3)).reshape(81, 2, 3) + [2, 5, 7]
        moved[:, :, 3:] = (transform * Rotation.from_quat(pose[:, :, 3:].reshape(-1, 4))).as_quat().reshape(81, 2, 4)
        moved[::2, :, 3:] *= -1
        np.testing.assert_allclose(original, episode_steps({'ee_pose': moved, 'gripper': grip}), atol=1e-6)
        first = episode_steps({'ee_pose': pose[:51], 'gripper': grip[:51]})
        second = episode_steps({'ee_pose': pose[25:76], 'gripper': grip[25:76]})
        np.testing.assert_array_equal(first[25:], second[:-25])

    def test_no_pairs_across_gaps_episodes_or_splits(self):
        rows = [{'episode_id': ep, 'split': split, 'start': start} for ep, split, start in
                [('a', 'train', 0), ('a', 'train', 25), ('a', 'train', 75), ('b', 'validation', 0), ('b', 'validation', 25)]]
        chains, pairs = chains_and_pairs(rows, 25)
        np.testing.assert_array_equal(pairs, [[0, 1], [3, 4]])
        self.assertEqual([c.tolist() for c in chains], [[0, 1], [2], [3, 4]])
        rows[1]['split'] = 'validation'
        with self.assertRaises(ValueError):
            chains_and_pairs(rows, 25)

    def test_gate_statistics_are_train_only(self):
        rng = np.random.default_rng(7); steps = rng.normal(size=(6, 50, 16)).astype(np.float32)
        pairs = np.array([[0, 1], [1, 2], [3, 4], [4, 5]]); train = np.array([True]*3+[False]*3)
        score, weights, settings = fit_gate(steps, pairs, train, 25)
        steps[3:] *= 100000
        new_score, new_weights, new_settings = fit_gate(steps, pairs, train, 25)
        self.assertEqual(settings, new_settings)
        np.testing.assert_array_equal(weights[:2], new_weights[:2]); np.testing.assert_array_equal(score[:2], new_score[:2])

    def test_calibration_hits_train_target_without_validation_leakage(self):
        rng = np.random.default_rng(5)
        motion = rng.normal(size=(100, 64)); motion /= np.linalg.norm(motion, axis=1, keepdims=True)
        visual = rng.normal(size=(100, 8)); visual /= np.linalg.norm(visual, axis=1, keepdims=True)
        train_pairs = sample_pairs(np.arange(60), count=2000)
        self.assertTrue(np.all(train_pairs < 60)); self.assertTrue(np.all(train_pairs[:, 0] != train_pairs[:, 1]))
        alpha, audit = calibrate_alpha(motion, visual, train_pairs)
        self.assertAlmostEqual(audit['achieved_train']['visual_ratio_of_mean_distances'], .05, places=6)
        motion[60:] *= 100; visual[60:] *= 100
        other, _ = calibrate_alpha(motion, visual, train_pairs)
        self.assertEqual(alpha, other)

    def test_visual_contribution_increases_when_motion_distances_shrink(self):
        rng = np.random.default_rng(7)
        motion = rng.normal(size=(50, 64)); motion /= np.linalg.norm(motion, axis=1, keepdims=True)
        visual = rng.normal(size=(50, 8)); visual /= np.linalg.norm(visual, axis=1, keepdims=True)
        pairs = sample_pairs(np.arange(50))
        concentrated = .1 * motion + np.eye(1, 64)[0]
        concentrated /= np.linalg.norm(concentrated, axis=1, keepdims=True)
        before = distance_contribution(motion, visual, pairs, .05)
        after = distance_contribution(concentrated, visual, pairs, .05)
        self.assertGreater(after['visual_mean_pair_fraction'], before['visual_mean_pair_fraction'])
        self.assertLess(calibrate_alpha(concentrated, visual, pairs)[0], calibrate_alpha(motion, visual, pairs)[0])

    def test_guard_blocks_cumulative_small_gripper_changes(self):
        action = np.zeros((3, 50, 16), dtype=np.float32); action[1, :30, 14] = .01
        weights, blocked = guarded_weights(action, np.array([[0, 1], [1, 2], [0, 2]]), np.ones(3))
        np.testing.assert_array_equal(blocked, [True, True, False]); np.testing.assert_array_equal(weights, [0, 0, 1])

    def test_metrics_use_only_validation_pairs(self):
        labels = np.array([0, 1, 0, 0, 1]); pairs = np.array([[0, 1], [2, 3], [3, 4]])
        result = temporal_metrics(labels, pairs, np.array([0., 0., 2.]), np.array([False, False, True, True, True]),
                                  [np.array([0, 1]), np.array([2, 3, 4])])
        self.assertEqual(result['all_switch_rate'], .5); self.assertEqual(result['quiet_switch_rate'], 0.)


class GripperReviewTests(unittest.TestCase):
    def test_gradual_open_close_are_separate_events(self):
        grip = np.zeros((61, 2)); grip[10:21, 0] = np.linspace(0, 1, 11); grip[21:30, 0] = 1
        grip[30:41, 0] = np.linspace(1, 0, 11)
        events = detect_events(grip)
        self.assertEqual([e['direction'] for e in events], ['open', 'close'])
        self.assertEqual([(e['start'], e['end']) for e in events], [(10, 20), (30, 40)])
        self.assertEqual(detect_events(np.full((50, 2), .5)), [])

    def test_sampling_is_deterministic_stratified_and_label_independent(self):
        events = [{'arm': arm, 'direction': direction, 'episode_id': str(i), 'event_id': f'{arm}{direction}{i}'}
                  for arm in (0, 1) for direction in ('open', 'close') for i in range(10)]
        a, b = select_events(events, 3), select_events(events, 3)
        self.assertEqual(a, b); self.assertEqual(len(a), 12)
        for arm in (0, 1):
            for direction in ('open', 'close'):
                chosen = [e for e in a if e['arm'] == arm and e['direction'] == direction]
                self.assertEqual(len({e['episode_id'] for e in chosen}), 3)

    def test_edge_events_do_not_fabricate_bracketing_windows(self):
        rows = [{'start': t, 'end': t+50, 'episode_id': 'a'} for t in (0, 25, 50, 75)]
        early = {'start': 0, 'end': 10, 'episode_id': 'a', 'arm': 0, 'direction': 'open'}
        self.assertIsNone(bracket_event(early, rows, range(4))['center_pair'])
        event = {**early, 'start': 55, 'end': 65}
        bracket = bracket_event(event, rows, range(4))
        self.assertEqual(bracket['center_pair'], [1, 2]); self.assertEqual(bracket['clean_flank_pair'], [0, 3])
        scored = event_scores([early, event], rows, np.array([0, 1, 1, 2]))
        self.assertEqual(scored['all']['eligible_events'], 1)
        self.assertEqual(scored['all']['same_cluster_fraction'], 1.)
        self.assertIn('not a skill-merge error rate', scored['all']['interpretation'])


if __name__ == '__main__':
    unittest.main()

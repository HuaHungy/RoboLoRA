import unittest

import numpy as np

from action_clustering.visual_aux import episode_frames, fuse, unit_rows, visual_codes


class VisualAuxTests(unittest.TestCase):
    def test_distance_weight_is_explicit_and_zero_keeps_baseline(self):
        rng = np.random.default_rng(12)
        motion = unit_rows(rng.normal(size=(20, 64)))
        visual = unit_rows(rng.normal(size=(20, 8)))
        np.testing.assert_array_equal(fuse(motion, visual, 0), motion)
        mixed = fuse(motion, visual, .05)
        dm = np.square(motion[:, None] - motion[None, :]).sum(axis=-1)
        dv = np.square(visual[:, None] - visual[None, :]).sum(axis=-1)
        actual = np.square(mixed[:, None] - mixed[None, :]).sum(axis=-1)
        np.testing.assert_allclose(actual, .95 * dm + .05 * dv, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(mixed, axis=1), 1, atol=1e-6)

    def test_visual_pca_does_not_fit_validation(self):
        rng = np.random.default_rng(3)
        first, last = rng.normal(size=(40, 16)), rng.normal(size=(40, 16))
        train = np.arange(30)
        before, params_before, _ = visual_codes(first, last, train)
        first[30:] = rng.normal(size=(10, 16)) * 1000
        last[30:] = rng.normal(size=(10, 16)) * 1000
        after, params_after, _ = visual_codes(first, last, train)
        for key in params_before:
            np.testing.assert_array_equal(params_before[key], params_after[key])
        np.testing.assert_array_equal(before[train], after[train])
        self.assertEqual(after.shape, (40, 8))

    def test_image_frames_do_not_require_missing_terminal_observation(self):
        frames = episode_frames([{'start': 0, 'end': 50}, {'start': 25, 'end': 75}])
        np.testing.assert_array_equal(frames, [0, 25, 49, 74])


if __name__ == '__main__':
    unittest.main()

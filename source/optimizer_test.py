import unittest
import numpy as np

from .optimizer import optimize, arctan_unwarp, _linear_fit, _optimizable


class TestOptimizer(unittest.TestCase):

    def make_test_data(self):
        xs = np.arange(0, 1000*np.pi, 0.1).reshape(-1, 1)
        ys_linear = xs * 3
        ys_nonlinear = np.sin(xs)
        linear_set = np.concatenate([xs, ys_linear], axis=-1)
        nonlinear_set = np.concatenate([xs, ys_nonlinear], axis=-1)
        lines_lst = [linear_set]*3
        nonlines_lst = [nonlinear_set]*3
        combined_lst = nonlines_lst + lines_lst
        return lines_lst, nonlines_lst, combined_lst

    def test_linear_fit(self):
        lines_lst, nonlines_lst, combined_lst = self.make_test_data()
        # check perfect lines
        self.assertAlmostEqual(_linear_fit(lines_lst), len(lines_lst), delta=5e-3)
        # check no lines
        self.assertAlmostEqual(_linear_fit(nonlines_lst), 0, delta=5e-3)
        # check combined
        self.assertAlmostEqual(_linear_fit(combined_lst), len(combined_lst)*0.5, delta=5e-3)

    def test_optimizable(self):
        _, _, combined_lst = self.make_test_data()
        a_vals = np.arange(1e-8, 3, 0.1)
        image_height = 1080
        image_width = 1920
        for a in a_vals:
            _optimizable(combined_lst, a, image_height, image_width)

    def test_arctan_unwarp(self):
        # check that with near-zero `a` we get almost the same points back
        points_array = np.array([[1, 1], [2, 2], [3, 3]])
        transformed_points = arctan_unwarp(points_array, 1e-32, 1920, 1080)
        max_delta = np.max(np.abs(points_array-transformed_points))
        self.assertAlmostEqual(max_delta, 0., delta=0.05)

    def test_optimize(self):
        lines_lst, nolines_lst, combined_lst = self.make_test_data()
        image_height = 1080
        image_width = 1920
        result = optimize(lines_lst, image_width, image_height, max_iterations=200)
        print(result)
        self.assertAlmostEqual(result, 0., delta=0.01)

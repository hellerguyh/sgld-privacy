import os
import sys
import unittest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(os.path.normpath(src_path))

from utils_dp import clopper_pearson_interval, getStats

class TestUtilsDP(unittest.TestCase):
    #@unittest.skip("skipping")
    def test_clopper_pearson(self):
        lb, up = clopper_pearson_interval(35, 200, alpha = 0.05)
        self.assertAlmostEqual(lb, 0.1250, 4)
        self.assertAlmostEqual(up, 0.2349, 4)

    def test_getStats(self):
        pred =  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        label = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
        GT_FP = 2
        GT_FN = 1
        FN_rate, FP_rate, FN, FP, pos, neg = getStats(pred, label)
        self.assertEqual(FN, GT_FN)
        self.assertEqual(FP, GT_FP)
        self.assertEqual(pos, 5)
        self.assertEqual(neg, 5)

        pred =  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        label = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        GT_FP = 1
        GT_FN = 1
        FN_rate, FP_rate, FN, FP, pos, neg = getStats(pred, label)
        self.assertEqual(FN, GT_FN)
        self.assertEqual(FP, GT_FP)
        self.assertEqual(pos, 6)
        self.assertEqual(neg, 4)


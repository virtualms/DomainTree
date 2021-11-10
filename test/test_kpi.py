# import sys
# sys.path.append("..\\domain_tree\\")
import unittest
from domain_tree.tree import DomainTree, DomainNode, NodeNotFoundException
from domain_tree import kpi


class TestKPI(unittest.TestCase):

    def setUp(self):
        self.depth = 3
        self.d0 = {"x0": (0, 1), "x1": (0, 1)}
        self.d1 = {"x0": (1, 2), "x1": (1, 2)}
        self.original_m = DomainTree(self.d0, depth_max=self.depth)
        self.approx_m = DomainTree(self.d0, depth_max=self.depth)
        self.blank_m = DomainTree(self.d0, min_split=1, depth_max=self.depth)

    def tearDown(self) -> None:
        pass

    def test_check_partition_num(self):
        original_partitions, approx_partitions, delta_p = kpi.check_partition_num(self.original_m, self.original_m)
        self.assertEqual(original_partitions, approx_partitions)
        self.assertEqual(delta_p, 0)

        original_partitions, approx_partitions, delta_p = kpi.check_partition_num(self.original_m, self.blank_m)
        self.assertNotEqual(original_partitions, approx_partitions)
        self.assertEqual(delta_p, -len(self.original_m.leaves) + 1)

    @unittest.skip("check_r2 is fake, waiting for the real model")
    def test_check_r2(self):
        pass

    def test_matching_bounds(self):
        a = (0, 1)
        b = (0, 1)
        self.assertTrue(kpi.matching_bounds(a, b, 0))

        a = (0, 1)
        b = (0, 1.01)
        self.assertFalse(kpi.matching_bounds(a, b, 0))

        a = (0, 1)
        b = (-0.5, 0.5)
        self.assertTrue(kpi.matching_bounds(a, b, 0.5))
        self.assertTrue(kpi.matching_bounds(b, a, 0.5))
        self.assertFalse(kpi.matching_bounds(a, b, 0.4999))
        self.assertFalse(kpi.matching_bounds(b, a, 0.4999))

        a = (0.3, 0.6)
        b = (0.2, 0.7)
        self.assertTrue(kpi.matching_bounds(a, b, 0.1))
        self.assertTrue(kpi.matching_bounds(b, a, 0.1))
        self.assertFalse(kpi.matching_bounds(a, b, 0))
        self.assertFalse(kpi.matching_bounds(b, a, 0))


        #float precision error
        a = (0.4, 0.6)
        b = (0.3, 0.7)
        self.assertTrue(kpi.matching_bounds(a, b, 0.1))

        a = (-0.3, -0.6)
        b = (-0.2, -0.7)
        self.assertTrue(kpi.matching_bounds(a, b, 0.1))
        self.assertTrue(kpi.matching_bounds(b, a, 0.1))
        self.assertFalse(kpi.matching_bounds(a, b, 0))
        self.assertFalse(kpi.matching_bounds(b, a, 0))



    def test_matching_intervals(self):
        #TODO very naive
        self.assertFalse(kpi.matching_intervals(DomainNode(domains=self.d0, name="node"),
                                                DomainNode(domains=self.d1, name="node"), 0))
        self.assertTrue(kpi.matching_intervals(DomainNode(domains=self.d0, name="node"),
                                               DomainNode(domains=self.d1, name="node"), 1))
        self.assertTrue(kpi.matching_intervals(DomainNode(domains=self.d0, name="node"),
                                               DomainNode(domains=self.d0, name="node"), 0))

    def test_check_matching_partitions(self):
        # conf=0
        part_num = len(self.original_m.leaves)
        self.assertEqual(kpi.check_matching_partitions(self.original_m, self.original_m, n=1, d=0), part_num)
        self.assertEqual(kpi.check_matching_partitions(self.original_m, self.blank_m, n=1, d=0), 0)

        # conf=1
        self.assertEqual(kpi.check_matching_partitions(self.original_m, self.approx_m, n=1, d=4), part_num)
        self.assertEqual(kpi.check_matching_partitions(self.original_m, self.blank_m, n=1, d=4), 1)

        # conf=0.5
        orig = DomainTree(self.d0, min_split=0.25, depth_max=3)
        appr = DomainTree(self.d0, min_split=0.25, depth_max=3)
        self.assertIn(kpi.check_matching_partitions(orig, appr, n=2, d=4), [part_num - 2, part_num - 1, part_num])


if __name__ == "__main__":
    unittest.main()

import unittest
from domain_tree.tree import DomainTree, DomainNode, NodeNotFoundException
from domain_tree.domain import RealDomain, RealInterval


class TestDomainTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        # self.d0 = {"x0": (0, 1)}
        self.d0 = RealDomain({"x0": RealInterval((0, 1), (True, False))})

    def tearDown(self) -> None:
        pass

    def test_RealInterval(self):
        i0 = RealInterval((0, 1), (False, False))
        i1 = RealInterval((0, 1), (True, False))
        i2 = RealInterval((0, 1), (False, True))
        i3 = RealInterval((0, 1), (True, True))

        sx, dx = i0.perfect_split(0.5)
        self.assertEquals(sx, RealInterval((0, 0.5), (False, False)))
        self.assertEquals(dx, RealInterval((0.5, 1), (True, False)))

        self.assertTrue(i0 == i0)
        self.assertFalse(i0 == i1)
        self.assertFalse(i0 == i2)
        self.assertFalse(i0 == i3)

        sx, dx = i3.perfect_split(0.5)
        self.assertEquals(sx, RealInterval((0, 0.5), (True, False)))
        self.assertEquals(dx, RealInterval((0.5, 1), (True, True)))

        self.assertFalse(i0.contains(0))
        self.assertFalse(i0.contains(1))
        self.assertTrue(i0.contains(0.999))
        self.assertTrue(i3.contains(0))
        self.assertTrue(i3.contains(1))
        self.assertFalse(i0.contains(2))


    def test_RealDomain(self):
        i0 = RealInterval((0, 1), (False, False))
        i1 = RealInterval((0, 1), (True, False))
        i2 = RealInterval((0, 1), (False, True))
        i3 = RealInterval((0, 1), (True, True))

        d0 = {"x0": i0, "x1": i1}
        d1 = {"x0": i2, "x1": i3}
        r0 = RealDomain(d0)
        self.assertTrue(r0.contains({"x0": 0.5, "x1": 0.5}))
        self.assertTrue(r0.contains({"x0": 0.999999999, "x1": 0.9999999}))
        self.assertFalse(r0.contains({"x0": 0, "x1": 0.5}))

        self.assertTrue(r0 == r0)
        self.assertFalse(r0 == RealDomain(d1))


if __name__ == "__main__":
    unittest.main()

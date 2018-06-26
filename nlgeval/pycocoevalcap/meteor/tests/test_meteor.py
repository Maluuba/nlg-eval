import unittest

from nlgeval.pycocoevalcap.meteor.meteor import Meteor


class TestMeteor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Meteor()

    def test_compute_score(self):
        s = self.m.compute_score({0: ["test"]}, {0: ["test"]})
        self.assertEqual(s, (1.0, [1.0]))

        s = self.m.compute_score({0: ["テスト"]}, {0: ["テスト"]})
        self.assertEqual(s, (1.0, [1.0]))

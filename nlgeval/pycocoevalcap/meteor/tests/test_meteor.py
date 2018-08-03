# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import unittest

from nlgeval.pycocoevalcap.meteor.meteor import Meteor


class TestMeteor(unittest.TestCase):
    def test_compute_score(self):
        m = Meteor()

        s = m.compute_score({0: ["test"]}, {0: ["test"]})
        self.assertEqual(s, (1.0, [1.0]))

        s = m.compute_score({0: ["テスト"]}, {0: ["テスト"]})
        self.assertEqual(s, (1.0, [1.0]))

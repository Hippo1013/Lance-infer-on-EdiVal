import unittest
import numpy as np
from lance_mice.attention import group_layout


class AttentionLayoutTests(unittest.TestCase):
    def test_full_denominator_image_streams_and_instruction_identity(self):
        trace = [dict(kind='vit', image_index=0, tokens=[4,8]),
                 dict(kind='vae', image_index=0, tokens=[8,12]),
                 dict(kind='text', role='label', turn=1, tokens=[12,14]),
                 dict(kind='text', role='history', turn=1, tokens=[14,16]),
                 dict(kind='vit', image_index=1, tokens=[16,20]),
                 dict(kind='vae', image_index=1, tokens=[20,24]),
                 dict(kind='text', role='current', turn=2, tokens=[24,26])]
        names, labels, counts = group_layout(trace, 29, 2, 8)
        self.assertEqual(names, ['context_other','I0_vit','I0_vae','T1','I1_vit','I1_vae','T2','generation_markers','target_image'])
        self.assertEqual(counts.tolist(), [9,4,4,2,4,4,2,2,8])
        self.assertEqual(sum(counts), 39)
        self.assertEqual(labels[0], 0)
        self.assertTrue(np.all(labels[-8:] == names.index('target_image')))

    def test_overlap_and_out_of_bounds_fail_closed(self):
        for spans in ([[0,4],[3,7]], [[0,4],[4,12]]):
            with self.assertRaises(ValueError):
                group_layout([dict(kind='vit',image_index=i,tokens=p) for i,p in enumerate(spans)],8,2,4)

    def test_empty_label_boundary_does_not_change_token_groups(self):
        trace = [dict(kind='vit', image_index=0, tokens=[4,8]),
                 dict(kind='vae', image_index=0, tokens=[8,12]),
                 dict(kind='text', role='current', turn=1, tokens=[12,16])]
        bare = trace[:2] + [dict(kind='text', role='label', text='', turn=1, tokens=[12,12])] + trace[2:]
        expected = group_layout(trace, 20, 2, 8)
        actual = group_layout(bare, 20, 2, 8)
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_only_empty_label_may_have_zero_tokens(self):
        for segment in (dict(kind='vit', image_index=0, tokens=[4,4]),
                        dict(kind='text', role='current', text='', turn=1, tokens=[4,4]),
                        dict(kind='text', role='label', text='Edit:', turn=1, tokens=[4,4]),
                        dict(kind='text', role='label', text='', turn=1, tokens=[21,21])):
            with self.assertRaises(ValueError):
                group_layout([segment], 20, 2, 8)

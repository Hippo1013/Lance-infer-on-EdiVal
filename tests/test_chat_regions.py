import unittest
import numpy as np
from lance_mice.protocol import history_segments, CHAT_PROTOCOL_VERSION, PROTOCOL_VERSION, prefix_length, render_user
from lance_mice.attention_regions import region_layout
from lance_mice.settings import Settings

class ChatRegionTests(unittest.TestCase):
    def test_role_history_and_cfg_boundary(self):
        for n in [1,2,3]:
            ins=['first','second','third'][:n]
            seg=history_segments(ins,version=CHAT_PROTOCOL_VERSION)
            text=render_user(ins,version=CHAT_PROTOCOL_VERSION)
            self.assertEqual(text.count('<|im_start|>assistant'),n-1)
            self.assertEqual(text.count('<|im_start|>user'),n-1)
            self.assertEqual([s.image_index for s in seg if s.kind=='image'],list(range(n)))
            self.assertEqual([s.text for s in seg if s.role in {'history','current'}],ins)
            self.assertEqual(seg[prefix_length(seg)+1].role,'current')
            self.assertEqual(seg[-1].text,ins[-1])
    def test_prefix_is_immutable(self):
        a=history_segments(['first','second'],version=CHAT_PROTOCOL_VERSION)
        b=history_segments(['first','second','third'],version=CHAT_PROTOCOL_VERSION)
        self.assertEqual(a[:prefix_length(a)],b[:prefix_length(a)])
    def test_identity_and_separation(self):
        self.assertNotEqual(Settings().identity(),Settings(history_protocol=CHAT_PROTOCOL_VERSION).identity())
        self.assertEqual(Settings().identity()['protocol'],PROTOCOL_VERSION)
        self.assertNotIn('history_protocol',Settings().identity()['settings'])
    def test_spatial_regions_and_instruction_keys(self):
        for h,w in [(8,8),(16,12),(6,14)]:
            n=h*w
            trace=[dict(kind='vit',image_index=0,tokens=[1,n+3],spatial_tokens=[2,n+2],grid_hw=[h,w]),
                   dict(kind='vae',image_index=0,tokens=[n+3,2*n+5],spatial_tokens=[n+4,2*n+4],grid_hw=[h,w]),
                   dict(kind='text',role='current',turn=1,tokens=[2*n+5,2*n+7],text='test',token_ids=[4,5],offsets=[[0,2],[2,4]])]
            d=region_layout(trace,2*n+7,2,64)
            self.assertEqual(d['region_counts'].sum(),2*n)
            self.assertEqual(d['counts'].sum(),len(d['labels']))
            self.assertEqual(d['fine'][0],-1)
            self.assertEqual(d['fine'][n+2],-1)
            self.assertTrue(np.all(d['fine'][2:n+2]>=0))
            self.assertEqual(d['text_keys'].tolist(),[2*n+5,2*n+6])
            self.assertEqual(d['token_ids'].tolist(),[4,5])
    def test_invalid_spatial_grid_rejected(self):
        with self.assertRaises(ValueError):
            region_layout([dict(kind='vit',image_index=0,tokens=[0,66],spatial_tokens=[1,65],grid_hw=[7,8])],66,2,64)

if __name__=='__main__':unittest.main()

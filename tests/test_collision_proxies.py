import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from dual_arm_pose_teleop.model_loader import slab_boxes


class ProxyTests(unittest.TestCase):
    def test_crossing_triangles_are_enclosed(self):
        triangles=np.array([[[0,0,0],[1,0,1],[0,1,1]],
                            [[0,0,0],[0,1,1],[-1,0,1]]],dtype=float)
        boxes=slab_boxes(triangles,.1,.002,0)
        rng=np.random.default_rng(5)
        bary=rng.dirichlet([1,1,1],500)
        for tri in triangles:
            points=bary@tri
            inside=np.zeros(len(points),dtype=bool)
            for position,size in boxes:
                inside |= np.all(np.abs(points-position) <= size+1e-10,axis=1)
            self.assertTrue(inside.all())

    def test_no_bridge_between_low_wide_and_high_narrow_geometry(self):
        triangles=np.array([[[-1,-1,0],[1,-1,.1],[1,1,.1]],
                            [[-.1,-.1,.1],[.1,-.1,1],[.1,.1,1]],
                            [[.1,.1,.1],[-.1,.1,1],[-.1,-.1,1]]])
        boxes=slab_boxes(triangles,.05,.002,0)
        self.assertFalse(any(np.all(np.abs(np.array([.8,0,.8])-p)<=s) for p,s in boxes))
        self.assertTrue(any(np.all(np.abs(np.array([0,0,.8])-p)<=s) for p,s in boxes))

    def test_bad_settings_rejected(self):
        tri=np.zeros((1,3,3))
        for height in (0,-.1,float('nan')):
            with self.assertRaises(ValueError):slab_boxes(tri,height)


if __name__=='__main__':unittest.main()

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from dual_arm_pose_teleop.profile_generator import generate_profile


MODEL = """<mujoco model="candidate">
  <worldbody>
    <body name="base">
      <body name="left_mount" pos="0 .2 0">
        <body name="left_shoulder"><joint name="left_j1" range="-2 2"/><geom group="3" size=".03"/>
          <body name="left_upper"><joint name="left_j2" range="-2 2"/><geom group="3" size=".03"/>
            <body name="left_elbow"><joint name="left_j3" range="-2 2"/><geom group="3" size=".03"/>
              <body name="left_forearm"><joint name="left_j4" range="-2 2"/><geom group="3" size=".03"/>
                <body name="left_wrist"><joint name="left_j5" range="-2 2"/><geom group="3" size=".03"/>
                  <body name="left_hand"><geom group="3" size=".02"/>
                    <body name="left_hand_camera"><geom group="2" contype="0" conaffinity="0" size=".01"/></body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_mount" pos="0 -.2 0">
        <body name="right_shoulder"><joint name="right_j1" range="-2 2"/><geom group="3" size=".03"/>
          <body name="right_upper"><joint name="right_j2" range="-2 2"/><geom group="3" size=".03"/>
            <body name="right_elbow"><joint name="right_j3" range="-2 2"/><geom group="3" size=".03"/>
              <body name="right_forearm"><joint name="right_j4" range="-2 2"/><geom group="3" size=".03"/>
                <body name="right_wrist"><joint name="right_j5" range="-2 2"/><geom group="3" size=".03"/>
                  <body name="right_hand"><geom group="3" size=".02"/></body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="head"><joint name="head_pan" range="-1 1"/><geom group="2" contype="0" conaffinity="0" size=".1"/></body>
    </body>
  </worldbody>
</mujoco>"""


class ProfileGeneratorTests(unittest.TestCase):
    def model(self, directory, xml=MODEL):
        root=pathlib.Path(directory)
        path=root/"assets"/"demo"/"demo.xml"
        path.parent.mkdir(parents=True)
        path.write_text(xml,encoding="utf-8")
        return root,path

    def test_detects_chains_semantic_bodies_and_collision_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.model(directory)
            result=generate_profile(path,root,"demo")
            c=result.values
            self.assertEqual(c["model_xml"],"assets/demo/demo.xml")
            self.assertEqual(c["frames"],{
                "base":"base", "left_shoulder":"left_shoulder",
                "left_elbow":"left_elbow", "left_wrist":"left_hand",
                "right_shoulder":"right_shoulder",
                "right_elbow":"right_elbow", "right_wrist":"right_hand"})
            self.assertEqual(c["arms"]["left"]["joint_names"],[f"left_j{i}" for i in range(1,6)])
            self.assertEqual(c["arms"]["right"]["joint_names"],[f"right_j{i}" for i in range(1,6)])
            self.assertEqual(c["ik"]["collision"]["geom_groups"],[3])
            self.assertEqual(len(c["initial_qpos"]),11)
            self.assertEqual(c["runtime"]["command_output"],"arm_joints")

    def test_ambiguous_names_fail_instead_of_guessing_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.model(directory,MODEL.replace("left_","a_").replace("right_","b_"))
            with self.assertRaisesRegex(ValueError,"Could not infer left"):
                generate_profile(path,root,"demo")

    def test_explicit_joint_and_frame_overrides_support_nonstandard_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.model(directory,MODEL.replace("left_","a_").replace("right_","b_"))
            result=generate_profile(
                path,root,"demo", left_joints=[f"a_j{i}" for i in range(1,6)],
                right_joints=[f"b_j{i}" for i in range(1,6)], base_body="base",
                frames={"left_shoulder":"a_shoulder","left_elbow":"a_elbow","left_wrist":"a_hand",
                        "right_shoulder":"b_shoulder","right_elbow":"b_elbow","right_wrist":"b_hand"},
                forward_axis="-y")
            self.assertEqual(result.values["editor"]["forward_axis"],"-y")
            self.assertEqual(result.values["frames"]["left_wrist"],"a_hand")

    def test_rejects_branched_or_overlapping_explicit_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root,path=self.model(directory)
            with self.assertRaisesRegex(ValueError,"serial kinematic chain"):
                generate_profile(path,root,"demo",left_joints=["left_j1","right_j1","left_j3"])
            with self.assertRaisesRegex(ValueError,"overlap"):
                generate_profile(path,root,"demo",left_joints=["left_j1","left_j2","left_j3"],
                                 right_joints=["left_j1","left_j2","left_j3"])


if __name__ == "__main__":
    unittest.main()

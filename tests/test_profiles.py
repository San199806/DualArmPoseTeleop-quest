import json
import pathlib
import unittest
import xml.etree.ElementTree as ET


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class RobotProfileTests(unittest.TestCase):
    def test_builtin_profiles_reference_valid_model_names(self):
        for profile_path in sorted((PROJECT_ROOT / "robots").glob("*.json")):
            with self.subTest(profile=profile_path.name):
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                model_path = PROJECT_ROOT / profile["model_xml"]
                self.assertTrue(model_path.is_file())
                root = ET.parse(model_path).getroot()
                compiler = root.find("compiler")
                mesh_dir = model_path.parent / (
                    compiler.get("meshdir", ".") if compiler is not None else "."
                )
                for mesh in root.findall(".//asset/mesh"):
                    mesh_file = mesh.get("file")
                    if mesh_file:
                        self.assertTrue(
                            (mesh_dir / mesh_file).resolve().is_file(),
                            f"missing mesh {mesh_file}",
                        )
                worldbody = root.find("worldbody")
                self.assertIsNotNone(worldbody)
                body_names = {
                    body.get("name") for body in worldbody.findall(".//body")
                }
                joint_names = {
                    joint.get("name") for joint in worldbody.findall(".//joint")
                }
                self.assertTrue(set(profile["frames"].values()) <= body_names)
                configured_arm_joints = [
                    name
                    for side in ("left", "right")
                    for name in profile["arms"][side]["joint_names"]
                ]
                self.assertEqual(len(configured_arm_joints), len(set(configured_arm_joints)))
                self.assertTrue(set(configured_arm_joints) <= joint_names)

                expected_nq = 0
                for joint in worldbody.findall(".//joint"):
                    joint_type = joint.get("type", "hinge")
                    expected_nq += {"free": 7, "ball": 4}.get(joint_type, 1)
                self.assertEqual(len(profile["initial_qpos"]), expected_nq)


if __name__ == "__main__":
    unittest.main()

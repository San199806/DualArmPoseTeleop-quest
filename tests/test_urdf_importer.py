import json
from pathlib import Path
import tempfile
import unittest

import mujoco

from dual_arm_pose_teleop.profile_generator import generate_profile
from dual_arm_pose_teleop.urdf_importer import convert_urdf


INERTIAL = '''<inertial><mass value="1"/><inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".01"/></inertial>'''


def dual_arm_urdf(mesh=""):
    links = [f'<link name="base">{INERTIAL}<visual><geometry><box size=".2 .2 .2"/></geometry></visual><collision><geometry><box size=".2 .2 .2"/></geometry></collision></link>']
    joints = []
    for side, y in (("left", ".2"), ("right", "-.2")):
        parent = "base"
        for index, semantic in enumerate(("shoulder", "elbow", "wrist"), 1):
            name = f"{side}_{semantic}"
            geometry = mesh if semantic == "wrist" and mesh else '<box size=".08 .08 .2"/>'
            links.append(f'<link name="{name}">{INERTIAL}<visual><geometry>{geometry}</geometry></visual><collision><geometry><sphere radius=".04"/></geometry></collision></link>')
            xyz = f"0 {y} 0" if index == 1 else ".2 0 0"
            joints.append(f'<joint name="{side}_j{index}" type="revolute"><parent link="{parent}"/><child link="{name}"/><origin xyz="{xyz}"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="1"/></joint>')
            parent = name
    return '<robot name="test">'+''.join(links+joints)+'</robot>'


class UrdfImporterTests(unittest.TestCase):
    def test_converts_primitives_and_generates_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"dual.urdf"
            source.write_text(dual_arm_urdf(), encoding="utf-8")
            destination = root/"assets"/"demo"
            destination.mkdir(parents=True)  # tempfile.mkdtemp-style staging
            imported = convert_urdf(source, destination, "demo")
            model = mujoco.MjModel.from_xml_path(str(imported.xml_path))
            self.assertEqual(model.nq, 6)
            self.assertEqual({int(value) for value in model.geom_group}, {2, 3})
            profile = generate_profile(imported.xml_path, root, "demo")
            self.assertEqual(profile.values["frames"]["left_wrist"], "left_wrist")
            self.assertEqual(profile.values["arms"]["right"]["joint_names"],
                             ["right_j1", "right_j2", "right_j3"])

    def test_resolves_package_mesh_and_copies_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root/"packages"/"demo_description"
            (package/"meshes").mkdir(parents=True)
            (package/"meshes"/"part.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n", encoding="ascii")
            source = root/"dual.urdf"
            source.write_text(dual_arm_urdf(
                '<mesh filename="package://demo_description/meshes/part.obj" scale=".01 .01 .01"/>'),
                encoding="utf-8")
            imported = convert_urdf(source, root/"output", "demo",
                                    package_roots=[root/"packages"])
            self.assertEqual(imported.report["meshes"], 1)
            self.assertEqual(len(list((root/"output"/"meshes").glob("*.obj"))), 1)

    def test_rejects_mimic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"bad.urdf"
            source.write_text('''<robot name="bad"><link name="a"/><link name="b"/>
              <joint name="j" type="revolute"><parent link="a"/><child link="b"/>
              <limit lower="-1" upper="1"/><mimic joint="other"/></joint></robot>''', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Mimic"):
                convert_urdf(source, root/"output", "bad")

    def test_rejects_missing_inertia_on_movable_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"bad.urdf"
            source.write_text(f'''<robot name="bad"><link name="a">{INERTIAL}</link><link name="b"/>
              <joint name="j" type="revolute"><parent link="a"/><child link="b"/>
              <limit lower="-1" upper="1"/></joint></robot>''', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no inertial"):
                convert_urdf(source, root/"output", "bad")

    def test_rejects_xacro(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"robot.urdf.xacro"
            source.write_text('<robot name="x"/>', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Xacro"):
                convert_urdf(source, Path(directory)/"output", "x")


if __name__ == "__main__":
    unittest.main()

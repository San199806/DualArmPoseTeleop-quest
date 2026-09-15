"""Offline initial-pose preview. No SDK, Redis or hardware output."""
import argparse
from pathlib import Path
import sys

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dual_arm_pose_teleop.configuration import load_configuration
from dual_arm_pose_teleop.model_loader import load_robot_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    c = load_configuration(ROOT/'config/teleop.json').values
    model = load_robot_model(ROOT/c['model_xml'])
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 960
    data = mujoco.MjData(model)
    data.qpos[:] = c['initial_qpos']
    mujoco.mj_forward(model,data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.1,0,.0]
    camera.distance = 2.6
    camera.azimuth = 200
    camera.elevation = -12
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    with mujoco.Renderer(model,height=960,width=960) as renderer:
        renderer.update_scene(data,camera=camera,scene_option=options)
        Image.fromarray(renderer.render()).save(args.output)
    print(args.output)

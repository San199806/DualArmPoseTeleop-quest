"""Read-only initial-pose geometry audit; no SDK, Redis or robot connection."""
import argparse
import json
import pathlib
import sys

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from dual_arm_pose_teleop.collision import CollisionGuard
from dual_arm_pose_teleop.model_loader import load_robot_model
from dual_arm_pose_teleop.configuration import load_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',default=None,help='Robot profile name in robots/')
    parser.add_argument('--config',help='Custom profile JSON')
    args = parser.parse_args()
    path = pathlib.Path(args.config).resolve() if args.config else ROOT/'config'/'teleop.json'
    loaded = load_configuration(path,robot=args.robot if not args.config else None)
    config = loaded.values
    model_path = pathlib.Path(config['model_xml'])
    if not model_path.is_absolute():
        model_path = loaded.project_root/model_path
    settings = dict(config['ik'].get('collision',{}))
    settings['enabled'] = True
    model = load_robot_model(model_path,settings)
    guard = CollisionGuard(model,config['arms'],settings)
    report = guard.check(config['initial_qpos'])
    print(f'[Collision audit] model={model_path}, pairs={len(guard.pairs)}, safe={report.safe}')
    print(report.message)
    for a,b,d,_ in guard.distances(guard.data):
        if d < guard.minimum:
            print(f'[Unsafe] {guard.pair_name(a,b)}: {d*1000:.2f} mm')
    print('Convex mesh audit only; not a real-robot safety certification.')
    return 0 if report.safe else 1


if __name__ == '__main__':
    sys.exit(main())

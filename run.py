import argparse
import json
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dual_arm_pose_teleop.configuration import load_configuration


def parse_args():
    parser = argparse.ArgumentParser(description="Pose-driven dual-arm teleoperation")
    parser.add_argument(
        "--robot",
        default=None,
        help="robot profile name in robots/ (ignored when --config is supplied)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="custom robot profile JSON",
    )
    parser.add_argument("--redis-host", default=None)
    parser.add_argument(
        "--skeleton-url",
        default=None,
        help="poll an external daemon instead of listening on port 8000",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="print effective settings without SDK/Redis")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--record", help="record sensor frames/buttons to a new JSONL file")
    capture.add_argument("--replay", help="timed offline replay; disables SDK and Redis output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--simulation",
        action="store_true",
        default=None,
        help="ignore real_robot_joint_state and calibrate from initial_qpos",
    )
    mode.add_argument("--real", dest="simulation", action="store_false", help="use timestamped real joint feedback")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config or str(PROJECT_ROOT / "config" / "teleop.json")
    if args.print_config:
        loaded = load_configuration(config_path, robot=args.robot if not args.config else None)
        if args.simulation is not None: loaded.values['runtime']['simulation'] = args.simulation
        if args.replay: loaded.values['runtime']['simulation'] = True
        if args.redis_host: loaded.values['runtime']['redis_host'] = args.redis_host
        print(json.dumps({'sources':loaded.sources,'robot_profile':str(loaded.profile_path),
                          'effective':loaded.values},ensure_ascii=False,indent=2))
        raise SystemExit(0)
    from dual_arm_pose_teleop.app import TeleopApplication
    TeleopApplication(
        config_path,
        redis_host=args.redis_host,
        headless=args.headless,
        auto_start=args.auto_start,
        simulation=args.simulation,
        record_path=args.record,
        replay_path=args.replay,
        robot=args.robot if not args.config else None,
        skeleton_url=args.skeleton_url,
    ).run()

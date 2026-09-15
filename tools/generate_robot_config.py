"""Generate and validate a robot JSON candidate from a MuJoCo XML model."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from dual_arm_pose_teleop.profile_generator import generate_profile


def comma_list(value):
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated joint list")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("--robot", required=True, help="profile/file name")
    parser.add_argument("--output", type=Path, help="default: robots/<robot>.json")
    parser.add_argument("--left-joints", type=comma_list, metavar="J1,J2,...")
    parser.add_argument("--right-joints", type=comma_list, metavar="J1,J2,...")
    parser.add_argument("--base-body")
    for side in ("left", "right"):
        for semantic in ("shoulder", "elbow", "wrist"):
            parser.add_argument(f"--{side}-{semantic}-body")
    parser.add_argument("--forward-axis", choices=("+x", "-x", "+y", "-y"), default="+x")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    parser.add_argument("--stdout", action="store_true", help="print candidate without writing")
    args = parser.parse_args()
    model = args.model_xml.resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    frames = {f"{side}_{semantic}": getattr(args, f"{side}_{semantic}_body")
              for side in ("left", "right") for semantic in ("shoulder", "elbow", "wrist")
              if getattr(args, f"{side}_{semantic}_body")}
    try:
        generated = generate_profile(
            model, ROOT, args.robot, left_joints=args.left_joints,
            right_joints=args.right_joints, base_body=args.base_body,
            frames=frames, forward_axis=args.forward_axis)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(generated.values, ensure_ascii=False, indent=2)+"\n"
    print(f"[Model] nq={generated.report['nq']} nv={generated.report['nv']} "
          f"left={generated.report['arm_dofs']['left']} DOF right={generated.report['arm_dofs']['right']} DOF")
    print("[Detected frames] "+json.dumps(generated.values["frames"], ensure_ascii=False))
    for warning in generated.warnings:
        print("[REVIEW] "+warning)
    if args.stdout:
        print(payload, end="")
        return
    output = (args.output or ROOT/"robots"/(args.robot+".json")).resolve()
    if output.exists() and not args.force:
        parser.error(f"output already exists: {output}; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=output.stem+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"[Saved] {output}")
    print(f"[Next] python edit_initial_qpos.py --robot {args.robot}")
    print(f"[Next] python check_collisions.py --robot {args.robot}")


if __name__ == "__main__":
    main()

"""Import a URDF and generate a MuJoCo XML plus teleoperation profile."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from dual_arm_pose_teleop.profile_generator import generate_profile
from dual_arm_pose_teleop.urdf_importer import convert_urdf


def comma_list(value):
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated joint list")
    return result


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def publish_directory(staging, destination, force):
    backup = None
    if destination.exists():
        if not force:
            raise FileExistsError(f"asset directory already exists: {destination}; use --force to replace it")
        backup = destination.with_name(destination.name+".import-backup")
        if backup.exists():
            raise FileExistsError(f"stale import backup exists: {backup}; inspect it before retrying")
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None:
            os.replace(backup, destination)
        raise
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="expanded .urdf file (not .xacro)")
    parser.add_argument("--robot", required=True, help="profile and asset directory name")
    parser.add_argument("--package-root", type=Path, action="append", default=[],
                        help="directory containing ROS packages; may be repeated")
    parser.add_argument("--left-joints", type=comma_list, metavar="J1,J2,...")
    parser.add_argument("--right-joints", type=comma_list, metavar="J1,J2,...")
    parser.add_argument("--base-body")
    for side in ("left", "right"):
        for semantic in ("shoulder", "elbow", "wrist"):
            parser.add_argument(f"--{side}-{semantic}-body")
    parser.add_argument("--forward-axis", choices=("+x", "-x", "+y", "-y"), default="+x")
    parser.add_argument("--floating-base", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace existing generated asset/profile")
    args = parser.parse_args()
    urdf = args.urdf.resolve()
    if not urdf.is_file():
        parser.error(f"URDF does not exist: {urdf}")
    if not all(character.isalnum() or character in "_-" for character in args.robot):
        parser.error("robot must contain only letters, numbers, underscores or hyphens")
    destination = ROOT/"assets"/args.robot
    profile_path = ROOT/"robots"/(args.robot+".json")
    if profile_path.exists() and not args.force:
        parser.error(f"profile already exists: {profile_path}; use --force to replace it")
    staging = Path(tempfile.mkdtemp(prefix=f".{args.robot}-import-", dir=ROOT/"assets"))
    backup = None
    try:
        imported = convert_urdf(
            urdf, staging, args.robot, package_roots=args.package_root,
            floating_base=args.floating_base)
        frames = {f"{side}_{semantic}": getattr(args, f"{side}_{semantic}_body")
                  for side in ("left", "right") for semantic in ("shoulder", "elbow", "wrist")
                  if getattr(args, f"{side}_{semantic}_body")}
        generated = generate_profile(
            imported.xml_path, ROOT, args.robot,
            left_joints=args.left_joints, right_joints=args.right_joints,
            base_body=args.base_body, frames=frames, forward_axis=args.forward_axis)
        generated.values["model_xml"] = f"assets/{args.robot}/{args.robot}.xml"
        manifest = dict(imported.report)
        manifest["warnings"] = imported.warnings+generated.warnings
        (staging/"import_report.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        payload = json.dumps(generated.values, ensure_ascii=False, indent=2)+"\n"
        backup = publish_directory(staging, destination, args.force)
        try:
            atomic_text(profile_path, payload)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination)
            if backup is not None:
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except (ValueError, OSError) as error:
        parser.error(str(error))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            print(f"[WARNING] preserved previous assets at {backup}", file=sys.stderr)
    print(f"[Imported] {destination/(args.robot+'.xml')}")
    print(f"[Profile] {profile_path}")
    print(f"[Model] nq={generated.report['nq']} nv={generated.report['nv']} "
          f"left={generated.report['arm_dofs']['left']} DOF right={generated.report['arm_dofs']['right']} DOF")
    print("[Detected frames] "+json.dumps(generated.values["frames"], ensure_ascii=False))
    for warning in imported.warnings+generated.warnings:
        print("[REVIEW] "+warning)
    print(f"[Next] python edit_initial_qpos.py --robot {args.robot}")
    print(f"[Next] python check_collisions.py --robot {args.robot}")
    print(f"[Run]  python run.py --robot {args.robot}")


if __name__ == "__main__":
    main()

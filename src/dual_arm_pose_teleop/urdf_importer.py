"""Convert a conventional URDF into a self-contained MuJoCo model directory."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .model_loader import load_robot_model


NATIVE_MESHES = {".obj", ".stl", ".msh"}


@dataclass
class ImportedUrdf:
    xml_path: Path
    source_sha256: str
    report: dict
    warnings: list[str]


def _numbers(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def _vector(value, size, label, default):
    if value is None:
        result = np.asarray(default, dtype=float)
    else:
        result = np.fromstring(value, sep=" ", dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def _origin(node):
    origin = node.find("origin")
    if origin is None:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    xyz = _vector(origin.get("xyz"), 3, "origin xyz", [0, 0, 0])
    roll, pitch, yaw = _vector(origin.get("rpy"), 3, "origin rpy", [0, 0, 0])
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    quat = np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])
    return xyz, quat


def _rotation_matrix(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "mesh"


def _copy_or_convert_mesh(source, mesh_dir, warnings):
    """Return a portable MuJoCo mesh filename inside *mesh_dir*."""
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    if source.suffix.lower() in NATIVE_MESHES:
        filename = f"{_safe_name(source.stem)}_{source_hash}{source.suffix.lower()}"
        target = mesh_dir/filename
        if not target.exists():
            shutil.copy2(source, target)
        return filename, source_hash
    try:
        import trimesh
    except ImportError as error:
        raise ValueError(
            f"Mesh {source} needs conversion to OBJ; install importer support with "
            "python -m pip install -e \".[urdf-import]\"") from error
    try:
        scene = trimesh.load(source, force="scene", process=False)
        parts = list(scene.dump())
        if not parts:
            raise ValueError("mesh scene is empty")
        combined = trimesh.util.concatenate(parts)
        payload = trimesh.exchange.obj.export_obj(
            combined, include_color=False, include_texture=False)
    except Exception as error:
        raise ValueError(f"Could not convert mesh {source} to OBJ: {error}") from error
    filename = f"{_safe_name(source.stem)}_{source_hash}.obj"
    target = mesh_dir/filename
    if not target.exists():
        target.write_text(payload, encoding="utf-8")
    warnings.append(f"converted {source.suffix} mesh to OBJ without external textures: {source.name}")
    return filename, source_hash


def _resolve_mesh(uri, urdf_path, package_roots):
    if not uri:
        raise ValueError("URDF mesh has no filename")
    normalized = uri.replace("\\", "/")
    candidates = []
    if normalized.startswith("package://"):
        remainder = normalized[len("package://"):]
        if "/" not in remainder:
            raise ValueError(f"Invalid package mesh URI: {uri}")
        package, relative = remainder.split("/", 1)
        for root in package_roots:
            root = Path(root).resolve()
            candidates.append(root/relative if root.name == package else root/package/relative)
        for ancestor in (urdf_path.parent, *urdf_path.parents):
            if ancestor.name == package:
                candidates.append(ancestor/relative)
            candidates.append(ancestor/package/relative)
    elif normalized.startswith("file://"):
        candidates.append(Path(normalized[len("file://"):]))
    elif "://" in normalized:
        raise ValueError(f"Remote mesh URI is unsupported: {uri}")
    else:
        path = Path(normalized)
        candidates.append(path if path.is_absolute() else urdf_path.parent/path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path) for path in candidates[:6])
    raise ValueError(f"Cannot resolve mesh {uri!r}; searched: {attempted}")


def _material_colors(robot):
    result = {}
    for material in robot.findall("material"):
        name = material.get("name")
        color = material.find("color")
        if name and color is not None:
            result[name] = _vector(color.get("rgba"), 4, f"material {name} rgba", [0.7]*3+[1])
    return result


def _validate_tree(links, joints):
    children = {}
    child_names = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')!r} needs parent and child links")
        parent_name, child_name = parent.get("link"), child.get("link")
        if parent_name not in links or child_name not in links:
            raise ValueError(f"Joint {joint.get('name')!r} references an unknown link")
        if child_name in child_names:
            raise ValueError(f"Link {child_name!r} has more than one parent")
        child_names.add(child_name)
        children.setdefault(parent_name, []).append(joint)
    roots = set(links)-child_names
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link; found {sorted(roots)}")
    visited = set()

    def visit(name):
        if name in visited:
            raise ValueError(f"URDF contains a cycle or repeated link at {name!r}")
        visited.add(name)
        for joint in children.get(name, []):
            visit(joint.find("child").get("link"))

    root = roots.pop()
    visit(root)
    if visited != set(links):
        raise ValueError(f"URDF has disconnected links: {sorted(set(links)-visited)}")
    return root, children


def convert_urdf(urdf_path, destination, robot_name, *, package_roots=(),
                 floating_base=False):
    """Convert *urdf_path* and copy referenced assets into *destination*.

    The destination is expected to be a fresh staging directory. The caller is
    responsible for publishing it after profile generation and validation.
    """
    urdf_path = Path(urdf_path).resolve()
    destination = Path(destination).resolve()
    if urdf_path.suffix.lower() == ".xacro":
        raise ValueError("Xacro is not expanded automatically; run xacro first and provide the resulting .urdf")
    try:
        robot = ET.parse(urdf_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid URDF XML: {error}") from error
    if robot.tag != "robot":
        raise ValueError(f"Expected <robot> root, found <{robot.tag}>")
    if any("xacro" in node.tag.lower() for node in robot.iter()):
        raise ValueError("Unexpanded xacro elements found; run xacro before importing")
    links_list = robot.findall("link")
    links = {node.get("name"): node for node in links_list if node.get("name")}
    if len(links) != len(links_list):
        raise ValueError("Every URDF link must have a unique non-empty name")
    joints = robot.findall("joint")
    joint_names = [node.get("name") for node in joints]
    if any(not name for name in joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Every URDF joint must have a unique non-empty name")
    if any(node.find("mimic") is not None for node in joints):
        raise ValueError("Mimic joints require robot-specific coupling; expand/remove the gripper mimic joints before importing")
    root_link, children = _validate_tree(links, joints)
    if destination.exists():
        if any(destination.iterdir()):
            raise ValueError(f"URDF staging directory is not empty: {destination}")
    else:
        destination.mkdir(parents=True)
    mesh_dir = destination/"meshes"
    source_dir = destination/"source"
    mesh_dir.mkdir()
    source_dir.mkdir()
    shutil.copy2(urdf_path, source_dir/urdf_path.name)

    mjcf = ET.Element("mujoco", model=robot_name)
    ET.SubElement(mjcf, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(mjcf, "option", timestep="0.01", gravity="0 0 -9.81")
    visual = ET.SubElement(mjcf, "visual")
    ET.SubElement(visual, "global", azimuth="150", elevation="-18")
    ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".8 .8 .8", specular=".1 .1 .1")
    assets = ET.SubElement(mjcf, "asset")
    world = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(world, "light", pos="1 0 3", dir="-.3 0 -1")
    global_colors = _material_colors(robot)
    mesh_cache = {}
    warnings = []

    def add_mesh(uri, scale):
        source = _resolve_mesh(uri, urdf_path, package_roots)
        scale_vector = _vector(scale, 3, f"mesh scale for {uri}", [1, 1, 1])
        key = source, tuple(scale_vector)
        if key in mesh_cache:
            return mesh_cache[key]
        filename, digest = _copy_or_convert_mesh(source, mesh_dir, warnings)
        asset_name = f"mesh_{len(mesh_cache)}_{digest}"
        ET.SubElement(assets, "mesh", name=asset_name, file=filename,
                      scale=_numbers(scale_vector))
        mesh_cache[key] = asset_name
        return asset_name

    def add_geometry(link_name, source, parent, kind, index):
        geometry = source.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError(f"{link_name} {kind} must contain exactly one geometry")
        xyz, quat = _origin(source)
        attrs = {
            "name": f"{link_name}_{kind}_{index}", "pos": _numbers(xyz),
            "quat": _numbers(quat), "group": "2" if kind == "visual" else "3",
            "contype": "0" if kind == "visual" else "1",
            "conaffinity": "0" if kind == "visual" else "1",
        }
        shape = geometry[0]
        if shape.tag == "mesh":
            attrs.update(type="mesh", mesh=add_mesh(shape.get("filename"), shape.get("scale")))
        elif shape.tag == "box":
            size = _vector(shape.get("size"), 3, f"{link_name} box size", [0, 0, 0])
            if np.any(size <= 0):
                raise ValueError(f"{link_name} box dimensions must be positive")
            attrs.update(type="box", size=_numbers(size/2))
        elif shape.tag == "sphere":
            radius = float(shape.get("radius", "nan"))
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"{link_name} sphere radius must be positive")
            attrs.update(type="sphere", size=f"{radius:.12g}")
        elif shape.tag == "cylinder":
            radius, length = float(shape.get("radius", "nan")), float(shape.get("length", "nan"))
            if not math.isfinite(radius+length) or radius <= 0 or length <= 0:
                raise ValueError(f"{link_name} cylinder dimensions must be positive")
            attrs.update(type="cylinder", size=_numbers([radius, length/2]))
        else:
            raise ValueError(f"Unsupported URDF geometry <{shape.tag}> on link {link_name}")
        if kind == "visual":
            color = source.find("material/color")
            material = source.find("material")
            rgba = (_vector(color.get("rgba"), 4, f"{link_name} visual rgba", [.7, .7, .7, 1])
                    if color is not None else global_colors.get(material.get("name") if material is not None else ""))
            attrs["rgba"] = _numbers(rgba if rgba is not None else [.7, .7, .7, 1])
        ET.SubElement(parent, "geom", attrs)

    def add_link(link_name, parent, incoming=None):
        body_name = link_name
        if link_name == "world":
            body_name = "world_link"
            if body_name in links:
                raise ValueError("URDF link 'world' conflicts with MuJoCo, and 'world_link' is already used")
            warnings.append("URDF link 'world' was renamed to 'world_link' because MuJoCo reserves that body name")
        body = ET.SubElement(parent, "body", name=body_name)
        if incoming is not None:
            xyz, quat = _origin(incoming)
            body.set("pos", _numbers(xyz))
            body.set("quat", _numbers(quat))
            kind = incoming.get("type")
            if kind != "fixed":
                if kind not in ("revolute", "continuous", "prismatic"):
                    raise ValueError(f"Unsupported joint type {kind!r}: {incoming.get('name')}")
                attrs = {
                    "name": incoming.get("name"),
                    "type": "slide" if kind == "prismatic" else "hinge",
                    "axis": _numbers(_vector(
                        incoming.find("axis").get("xyz") if incoming.find("axis") is not None else None,
                        3, f"joint {incoming.get('name')} axis", [1, 0, 0])),
                }
                if kind == "continuous":
                    attrs["limited"] = "false"
                else:
                    limit = incoming.find("limit")
                    if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                        raise ValueError(f"Joint {incoming.get('name')} needs lower and upper limits")
                    attrs["range"] = f"{float(limit.get('lower')):.12g} {float(limit.get('upper')):.12g}"
                dynamics = incoming.find("dynamics")
                if dynamics is not None:
                    if dynamics.get("damping") is not None:
                        attrs["damping"] = f"{float(dynamics.get('damping')):.12g}"
                    if dynamics.get("friction") is not None:
                        attrs["frictionloss"] = f"{float(dynamics.get('friction')):.12g}"
                ET.SubElement(body, "joint", attrs)
        elif floating_base:
            ET.SubElement(body, "freejoint", name=f"{link_name}_free")
        link = links[link_name]
        inertial = link.find("inertial")
        if inertial is not None:
            mass_node, inertia = inertial.find("mass"), inertial.find("inertia")
            if mass_node is None or inertia is None:
                raise ValueError(f"Link {link_name} has an incomplete inertial element")
            mass = float(mass_node.get("value", "nan"))
            fixed_frame = incoming is None or incoming.get("type") == "fixed"
            if not math.isfinite(mass) or mass <= 0:
                # SolidWorks flange frames are often exported with a zero inertial.
                if fixed_frame and mass == 0:
                    warnings.append(
                        f"Link {link_name} is fixed with zero mass; its inertial element was omitted")
                else:
                    raise ValueError(f"Link {link_name} mass must be positive")
            else:
                xyz, quat = _origin(inertial)
                values = [float(inertia.get(key, "nan")) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
                if not np.isfinite(values).all():
                    raise ValueError(f"Link {link_name} inertia tensor is incomplete")
                xx, xy, xz, yy, yz, zz = values
                tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                rotation = _rotation_matrix(quat)
                tensor = rotation@tensor@rotation.T
                ET.SubElement(body, "inertial", pos=_numbers(xyz), mass=f"{mass:.12g}",
                              fullinertia=_numbers(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
        elif incoming is not None and incoming.get("type") != "fixed":
            raise ValueError(f"Movable link {link_name!r} has no inertial data")
        for kind in ("visual", "collision"):
            for index, geom in enumerate(link.findall(kind)):
                add_geometry(link_name, geom, body, kind, index)
        for child_joint in children.get(link_name, []):
            add_link(child_joint.find("child").get("link"), body, child_joint)

    add_link(root_link, world)
    ET.indent(mjcf)
    xml_path = destination/f"{robot_name}.xml"
    ET.ElementTree(mjcf).write(xml_path, encoding="utf-8", xml_declaration=True)
    model = load_robot_model(xml_path)
    if not any(model.geom_group[geom] == 3 for geom in range(model.ngeom)):
        warnings.append("URDF contains no collision geometry; runtime collision checking needs a reviewed model")
    source_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    report = {
        "source_urdf": str(urdf_path), "source_sha256": source_hash,
        "links": len(links), "urdf_joints": len(joints), "nq": int(model.nq),
        "nv": int(model.nv), "meshes": len(mesh_cache),
        "floating_base": bool(floating_base),
    }
    return ImportedUrdf(xml_path, source_hash, report, warnings)

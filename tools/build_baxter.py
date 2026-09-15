"""Reproducible offline conversion of the pinned official Baxter URDF.

Development-only dependencies: trimesh, pycollada, scipy, mujoco.
Runtime uses the generated MJCF/OBJ and needs neither ROS nor a converter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

COMMIT = '6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e'
REPOSITORY = 'https://github.com/RethinkRobotics/baxter_common'


def numbers(values):
    return ' '.join(f'{v:.12g}' for v in values)


def origin(element):
    node = element.find('origin')
    if node is None:
        return np.zeros(3), Rotation.identity()
    return (np.fromstring(node.get('xyz', '0 0 0'), sep=' '),
            Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')))


def convert(upstream, destination, srdf):
    source = upstream / 'baxter_description'
    destination.mkdir(parents=True, exist_ok=True)
    meshdir = destination / 'meshes'
    meshdir.mkdir(exist_ok=True)
    # Preserve the official source, its package paths, license and mesh files.
    shutil.copytree(source, destination / 'source' / 'baxter_description', dirs_exist_ok=True)
    for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.md'):
        if (upstream / name).is_file():
            shutil.copy2(upstream / name, destination / name)
    urdf = ET.parse(source / 'urdf' / 'baxter.urdf').getroot()
    root = ET.Element('mujoco', model='baxter')
    ET.SubElement(root, 'compiler', angle='radian', meshdir='meshes', autolimits='true')
    ET.SubElement(root, 'option', timestep='0.01', gravity='0 0 -9.81')
    ET.SubElement(root, 'statistic', center='0.1 0 0.15', extent='2.5')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', azimuth='150', elevation='-18')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.8 .8 .8', specular='.1 .1 .1')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient',
                  rgb1='.25 .3 .38', rgb2='.05 .06 .08', width='512', height='2048')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='1 0 3', dir='-0.3 0 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='-1 0 2', dir='0.3 0 -1', diffuse='.5 .5 .5')
    cache = {}

    def meshes(filename, split):
        key = filename, split
        if key in cache:
            return cache[key]
        prefix = 'package://baxter_description/'
        if not filename.startswith(prefix):
            raise ValueError('Unexpected mesh URI: ' + filename)
        relative = filename[len(prefix):]
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError('Mesh path escapes source')
        loaded = trimesh.load(path, force='scene', process=False)
        parts = list(loaded.dump())  # Includes the COLLADA scene transforms.
        if not split:
            parts = [trimesh.util.concatenate(parts)]
        result = []
        for index, part in enumerate(parts):
            name = hashlib.sha256((relative+str(split)).encode()).hexdigest()[:12] + f'_{index}'
            # OBJ keeps geometry; explicit MJCF color avoids external textures.
            (meshdir / (name+'.obj')).write_text(
                trimesh.exchange.obj.export_obj(part, include_color=False, include_texture=False),
                encoding='utf-8')
            attrs = dict(name=name, file=name+'.obj')
            if split:
                attrs['inertia'] = 'shell'
            ET.SubElement(asset, 'mesh', **attrs)
            color = None
            if hasattr(part.visual, 'material'):
                material = part.visual.material
                diffuse = getattr(material, 'diffuse', None)
                if diffuse is None:
                    diffuse = getattr(material, 'baseColorFactor', None)
                if diffuse is not None:
                    color = np.asarray(diffuse, dtype=float)
                    if color.max() > 1:
                        color = color/255
            result.append((name, color))
        cache[key] = result
        return result

    links = {link.get('name'): link for link in urdf.findall('link')}
    children = {}
    child_names = set()
    for joint in urdf.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        children.setdefault(parent, []).append(joint)
        child_names.add(child)
    roots = set(links)-child_names
    if len(roots) != 1:
        raise ValueError('Expected one URDF root')

    def add_link(name, parent, joint=None):
        body = ET.SubElement(parent, 'body', name=name)
        if joint is not None:
            position, rotation = origin(joint)
            body.set('pos', numbers(position))
            body.set('quat', numbers(rotation.as_quat(scalar_first=True)))
            kind = joint.get('type')
            if kind != 'fixed':
                if kind not in ('revolute', 'continuous', 'prismatic'):
                    raise ValueError('Unsupported joint type: '+kind)
                attrs = dict(name=joint.get('name'), type='slide' if kind == 'prismatic' else 'hinge',
                             axis=joint.find('axis').get('xyz'))
                if kind != 'continuous':
                    limit = joint.find('limit')
                    attrs['range'] = limit.get('lower')+' '+limit.get('upper')
                dynamics = joint.find('dynamics')
                if dynamics is not None:
                    attrs['damping'] = dynamics.get('damping', '0')
                ET.SubElement(body, 'joint', **attrs)
        link = links[name]
        inertia = link.find('inertial')
        if inertia is not None:
            position, rotation = origin(inertia)
            i = inertia.find('inertia')
            xx, xy, xz, yy, yz, zz = [float(i.get(k)) for k in ('ixx','ixy','ixz','iyy','iyz','izz')]
            matrix = rotation.as_matrix() @ np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]) @ rotation.as_matrix().T
            ET.SubElement(body, 'inertial', pos=numbers(position), mass=inertia.find('mass').get('value'),
                          fullinertia=numbers(matrix[[0,1,2,0,0,1],[0,1,2,1,2,2]]))
        for kind in ('visual', 'collision'):
            for index, geom in enumerate(link.findall(kind)):
                position, rotation = origin(geom)
                attrs = dict(name=f'{name}_{kind}_{index}', pos=numbers(position),
                             quat=numbers(rotation.as_quat(scalar_first=True)),
                             group='2' if kind == 'visual' else '3',
                             contype='0' if kind == 'visual' else '1',
                             conaffinity='0' if kind == 'visual' else '1')
                color = geom.find('material/color')
                rgba = color.get('rgba') if color is not None else None
                geometry = geom.find('geometry')[0]
                if geometry.tag == 'mesh':
                    if geometry.get('scale', '1 1 1') != '1 1 1':
                        raise ValueError('Nonunit mesh scaling must be handled explicitly')
                    for part, (mesh, diffuse) in enumerate(meshes(geometry.get('filename'),kind == 'visual')):
                        chunk = dict(attrs, name=attrs['name']+f'_{part}', type='mesh', mesh=mesh)
                        if kind == 'visual':
                            chunk['rgba'] = rgba or (numbers(diffuse) if diffuse is not None else '.65 .08 .06 1')
                        ET.SubElement(body, 'geom', **chunk)
                else:
                    attrs['type'] = geometry.tag
                    if geometry.tag == 'box':
                        attrs['size'] = numbers(np.fromstring(geometry.get('size'),sep=' ')/2)
                    elif geometry.tag == 'sphere':
                        attrs['size'] = geometry.get('radius')
                    elif geometry.tag == 'cylinder':
                        attrs['size'] = numbers([float(geometry.get('radius')),float(geometry.get('length'))/2])
                    else:
                        raise ValueError('Unsupported geometry: '+geometry.tag)
                    if kind == 'visual':
                        attrs['rgba'] = rgba or '.6 .6 .6 1'
                    ET.SubElement(body, 'geom', **attrs)
        for child in children.get(name, []):
            add_link(child.find('child').get('link'), body, child)

    add_link(roots.pop(), world)
    # Carry robot-specific assembly semantics in MJCF, not teleop JSON/code.
    # Retain all cross-arm checks and forearm/wrist-to-base checks even when
    # the upstream sampling-generated SRDF marks some of them as "Never".
    semantics = ET.parse(srdf).getroot()
    collisions = {n for n,l in links.items() if l.find('collision') is not None}
    contact = ET.SubElement(root, 'contact')
    inherited = []
    for node in semantics.iter('disable_collisions'):
        a,b = node.get('link1'),node.get('link2')
        same_arm = any(a.startswith(s+'_') and b.startswith(s+'_') for s in ('left','right'))
        shoulder_mount = ('torso' in (a,b) and any(
            n.endswith(('upper_shoulder','lower_shoulder','upper_elbow')) for n in (a,b)))
        if a not in collisions or b not in collisions or not (same_arm or shoulder_mount):
            continue
        ET.SubElement(contact,'exclude',body1=a,body2=b)
        inherited.append(dict(body1=a,body2=b,reason=node.get('reason')))
    semantic_dir = destination/'source'/'baxter_moveit_config'
    semantic_dir.mkdir(exist_ok=True)
    shutil.copy2(srdf,semantic_dir/'baxter_base.srdf.xacro')
    (destination/'assembly_exclusions.json').write_text(json.dumps(inherited,indent=2)+'\n',encoding='utf-8')
    ET.indent(root)
    model_path = destination / 'baxter.xml'
    ET.ElementTree(root).write(model_path, encoding='utf-8', xml_declaration=True)
    previous = Path.cwd()
    try:
        os.chdir(destination)
        model = mujoco.MjModel.from_xml_path('baxter.xml')
    finally:
        os.chdir(previous)
    manifest = dict(repository=REPOSITORY,commit=COMMIT,source_urdf='source/baxter_description/urdf/baxter.urdf',
                    source_sha256=hashlib.sha256((source/'urdf'/'baxter.urdf').read_bytes()).hexdigest(),
                    nq=model.nq,joint_order=[model.joint(j).name for j in range(model.njnt)],
                    coordinate_frame='URDF: +X forward, +Y left, +Z up',
                    conversion='Official URDF link/joint/inertial/collision transforms retained; COLLADA transformed to OBJ; visual group 2, collision group 3.',
                    collision_semantics_repository='https://github.com/moveit/moveit_robots',
                    collision_semantics_commit='79c488b4afa548e19256f7adfee4fe1e7801f91f',
                    collision_semantics_sha256=hashlib.sha256(srdf.read_bytes()).hexdigest(),
                    assembly_exclusions=len(inherited))
    (destination/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('srdf', type=Path, help='Pinned upstream baxter_base.srdf.xacro')
    args = parser.parse_args()
    convert(args.upstream, args.destination, args.srdf)

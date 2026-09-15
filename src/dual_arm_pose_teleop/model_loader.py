"""Build conservative collision proxies without changing display meshes/joints.

Vertical slab boxes enclose every clipped mesh triangle. Unlike a single convex
hull they do not bridge empty space between low wide platforms and tall narrow
columns. This is conservative, not an exact CAD solid or safety certification.
"""
import pathlib
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def slab_boxes(triangles, height=.02, padding=.002, merge_tolerance=.001):
    triangles = np.asarray(triangles,dtype=float)
    if (triangles.ndim != 3 or triangles.shape[1:] != (3,3) or not len(triangles)
            or not np.isfinite(triangles).all()):
        raise ValueError('Invalid collision mesh triangles')
    if not np.isfinite([height,padding,merge_tolerance]).all() or height <= 0 or padding < 0 or merge_tolerance < 0:
        raise ValueError('Invalid slab proxy settings')
    low_z,high_z = triangles[:,:,2].min(),triangles[:,:,2].max()
    count = max(1,int(np.ceil((high_z-low_z)/height)))
    if count > 1000:
        raise ValueError('Too many collision slabs; check model units')
    cuts = np.linspace(low_z,high_z,count+1)
    zmin,zmax = triangles[:,:,2].min(axis=1),triangles[:,:,2].max(axis=1)
    boxes = []
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        tri = triangles[(zmin <= hi)&(zmax >= lo)]
        if not len(tri):
            continue
        points = tri.reshape(-1,3)
        samples = [points[(points[:,2]>=lo)&(points[:,2]<=hi)]]
        # Clip edges at both slab boundaries; retaining only original vertices
        # would miss long triangles crossing a slab without a vertex inside it.
        for i,j in ((0,1),(1,2),(2,0)):
            a,b = tri[:,i],tri[:,j]
            dz = b[:,2]-a[:,2]
            for plane in (lo,hi):
                mask = (np.abs(dz)>1e-15)&((a[:,2]-plane)*(b[:,2]-plane)<=0)
                t = (plane-a[mask,2])/dz[mask]
                samples.append(a[mask]+t[:,None]*(b[mask]-a[mask]))
        points = np.concatenate(samples)
        if not len(points):
            continue
        low,high = points.min(axis=0),points.max(axis=0)
        low[2],high[2] = lo,hi
        if boxes and abs(boxes[-1][1][2]-lo)<1e-8 and max(
                np.max(np.abs(boxes[-1][0][:2]-low[:2])),
                np.max(np.abs(boxes[-1][1][:2]-high[:2]))) <= merge_tolerance:
            boxes[-1] = (np.minimum(boxes[-1][0],low),np.maximum(boxes[-1][1],high))
        else:
            boxes.append((low,high))
    return [((lo+hi)/2,(hi-lo)/2+padding) for lo,hi in boxes]


def proxy_prefix(source_name):
    return '__collision_proxy_'+source_name+'_'


def _load_windows_unicode_xml(path):
    """Load through MuJoCo's in-memory filesystem, without changing cwd.

    Rewrite file references to unique ASCII names so different directories
    may safely contain assets with the same basename.
    """
    assets, names = {}, {}

    def collect(filename, directories=None, stack=()):
        filename = pathlib.Path(filename).resolve()
        if filename in stack:
            raise ValueError('Cyclic model include: '+str(filename))
        if filename in names:
            return names[filename]
        name = f'asset_{len(names)}'+filename.suffix
        names[filename] = name
        if filename.suffix.lower() not in ('.xml','.urdf'):
            assets[name] = filename.read_bytes()
            return name
        tree = ET.fromstring(filename.read_text(encoding='utf-8'))
        dirs = dict(directories or {'mesh':filename.parent,'texture':filename.parent})
        compiler = tree.find('compiler')
        if compiler is not None:
            for tag,key in (('mesh','meshdir'),('texture','texturedir')):
                value = compiler.get(key,compiler.get('assetdir'))
                if value is not None:
                    dirs[tag] = filename.parent/pathlib.Path(value)
            for key in ('meshdir','texturedir','assetdir'):
                compiler.attrib.pop(key,None)
        for node in tree.iter():
            if 'file' not in node.attrib:
                continue
            base = dirs.get(node.tag,filename.parent)
            dependency = base/pathlib.Path(node.get('file'))
            node.set('file',collect(dependency,dirs,(*stack,filename)))
        assets[name] = ET.tostring(tree,encoding='utf-8')
        return name

    main = collect(path)
    return mujoco.MjModel.from_xml_string(assets[main].decode('utf-8'),assets=assets)


def load_robot_model(model_xml, collision_settings=None):
    path = pathlib.Path(model_xml).absolute()
    settings = collision_settings or {}
    # Some Windows MuJoCo wheels cannot fopen a UTF-8 absolute path. A relative
    # ASCII path from the project directory avoids changing process-wide cwd.
    try:
        relative = os.path.relpath(path)
    except ValueError:  # Different Windows drives have no relative path.
        relative = str(path)
    filename = relative if os.name == 'nt' and relative.isascii() else str(path)
    baseline = (_load_windows_unicode_xml(path) if os.name == 'nt' and not filename.isascii()
                else mujoco.MjModel.from_xml_path(filename))
    replacements = settings.get('mesh_slab_proxies',[]) if settings.get('enabled',False) else []
    if not replacements:
        return baseline
    tree = ET.fromstring(path.read_text(encoding='utf-8'))
    compiler = tree.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(tree,'compiler')
    # Loading the modified XML from memory must preserve asset resolution.
    for key in ('meshdir','texturedir'):
        directory = pathlib.Path(compiler.get(key,compiler.get('assetdir','.')))
        compiler.set(key,str(directory if directory.is_absolute() else path.parent/directory))
    for node in tree.iter():
        if 'file' in node.attrib and node.tag not in ('mesh','texture'):
            asset = pathlib.Path(node.get('file'))
            if not asset.is_absolute():
                node.set('file',str(path.parent/asset))
    used = set()
    for spec in replacements:
        name = spec['geom']
        if name in used:
            raise ValueError('Duplicate collision proxy source: '+name)
        used.add(name)
        gid = baseline.geom(name).id
        if baseline.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('Slab source must be a mesh: '+name)
        body_name = baseline.body(int(baseline.geom_bodyid[gid])).name
        body = next((b for b in tree.iter('body') if b.get('name')==body_name),None)
        if body is None:
            raise ValueError('Proxy source body must be in the main XML: '+str(body_name))
        mesh = int(baseline.geom_dataid[gid])
        va,vn = int(baseline.mesh_vertadr[mesh]),int(baseline.mesh_vertnum[mesh])
        fa,fn = int(baseline.mesh_faceadr[mesh]),int(baseline.mesh_facenum[mesh])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation,baseline.geom_quat[gid])
        # MuJoCo recenters/reorients mesh vertices; transform back to body frame.
        vertices = baseline.mesh_vert[va:va+vn]@rotation.reshape(3,3).T+baseline.geom_pos[gid]
        triangles = vertices[baseline.mesh_face[fa:fa+fn]]
        boxes = slab_boxes(triangles,float(spec.get('slab_height_m',.02)),
                           float(spec.get('padding_m',.002)),float(spec.get('merge_tolerance_m',.001)))
        for i,(position,size) in enumerate(boxes):
            ET.SubElement(body,'geom',{
                'name':proxy_prefix(name)+str(i), 'type':'box',
                'pos':' '.join(map(str,position)), 'size':' '.join(map(str,size)),
                'quat':'1 0 0 0', 'density':'0', 'contype':'0','conaffinity':'0',
                'rgba':'0 0.8 0.5 0', 'group':'5'})
    result = mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    if result.nq != baseline.nq or result.nv != baseline.nv or result.njnt != baseline.njnt:
        raise ValueError('Collision proxy loading changed robot joint layout')
    for j in range(result.njnt):
        if result.joint(j).name != baseline.joint(j).name:
            raise ValueError('Collision proxy loading changed joint order')
    return result

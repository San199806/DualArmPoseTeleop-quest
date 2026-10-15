"""Compose defaults, robot and daily settings; no SDK/runtime imports."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from .defaults import DEFAULTS

DAILY_KEYS = {
    'simulation': ('runtime','simulation'),
    'mapping_mode': ('mapping','mode'),
    'redis_host': ('runtime','redis_host'),
    'human_upper_arm_length_m': ('sensor','human_upper_arm_length_m'),
    'human_forearm_length_m': ('sensor','human_forearm_length_m'),
    'max_joint_velocity_rad_s': ('runtime','max_joint_velocity_rad_s'),
    'collision_enabled': ('ik','collision','enabled'),
}


@dataclass
class LoadedConfiguration:
    values: dict
    project_root: Path
    profile_path: Path
    sources: list


def _strip_json_comments(text):
    """Drop // and /* */ comments that sit outside JSON strings."""
    out=[]; i=0; n=len(text); in_string=False; escape=False
    while i<n:
        ch=text[i]; nxt=text[i+1] if i+1<n else ''
        if in_string:
            out.append(ch)
            if escape: escape=False
            elif ch=='\\': escape=True
            elif ch=='"': in_string=False
            i+=1; continue
        if ch=='"':
            in_string=True; out.append(ch); i+=1; continue
        if ch=='/' and nxt=='/':
            i+=2
            while i<n and text[i] not in '\r\n': i+=1
            continue
        if ch=='/' and nxt=='*':
            end=text.find('*/', i+2)
            if end<0: raise ValueError('Unclosed block comment')
            i=end+2; continue
        out.append(ch); i+=1
    return ''.join(out)


def _read(path):
    value=json.loads(_strip_json_comments(Path(path).read_text(encoding='utf-8')))
    if not isinstance(value,dict):raise ValueError(f'Configuration must be an object: {path}')
    return value


def _merge(base,override):
    result=deepcopy(base)
    for key,value in override.items():
        if isinstance(result.get(key),dict):
            if not isinstance(value,dict):raise ValueError(f'Expected configuration object: {key}')
            result[key]=_merge(result[key],value)
        else:result[key]=deepcopy(value)
    return result


def _validate(c):
    for key in ('robot','model_xml','frames','arms','initial_qpos'):
        if key not in c:raise ValueError(f'Missing required configuration: {key}')
    if not isinstance(c['robot'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',c['robot']):
        raise ValueError('Invalid robot profile name')
    for key in ('frames','arms'):
        if not isinstance(c[key],dict) or not c[key]:raise ValueError(f'{key} must be a nonempty object')
    if c['mapping']['mode'] not in ('segment_direction','cooperative_wrist'):raise ValueError('Invalid mapping_mode')
    if not isinstance(c['runtime']['simulation'],bool) or not isinstance(c['ik']['collision']['enabled'],bool):
        raise ValueError('simulation and collision_enabled must be true or false')
    for section,key in (('sensor','human_upper_arm_length_m'),('sensor','human_forearm_length_m'),
                        ('runtime','max_joint_velocity_rad_s'),('runtime','frequency_hz')):
        value=c[section][key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'{section}.{key} must be a finite positive number')
    if not isinstance(c['runtime']['redis_host'],str) or not c['runtime']['redis_host'].strip():
        raise ValueError('redis_host must be nonempty')
    q=c['initial_qpos']
    if not isinstance(q,list) or not q or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in q):
        raise ValueError('initial_qpos must contain finite numbers')


def _profile(path):
    raw=_read(path); robot=raw.get('robot')
    if not isinstance(robot,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',robot):raise ValueError('Invalid robot name')
    defaults=deepcopy(DEFAULTS)
    if raw.get('schema_version')!=1:
        defaults['ik']['collision']['enabled']=False  # Preserve old profiles without collision settings.
    defaults['runtime']['calibration_file']=f'calibration/{robot}_default.json'
    root=path.parent.parent
    sources=[str(Path(__file__).with_name('defaults.py')),str(path)]
    return LoadedConfiguration(_merge(defaults,{k:v for k,v in raw.items() if k!='schema_version'}),root,path,sources)


def load_configuration(path,robot=None,_visited=()):
    path=Path(path).resolve()
    if path in _visited or len(_visited)>8:raise ValueError('Configuration reference cycle')
    raw=_read(path)
    if 'robot_profile' in raw:
        if set(raw)!={'robot_profile'}:raise ValueError('Legacy alias must contain only robot_profile')
        loaded=load_configuration(path.parent/raw['robot_profile'],_visited=(*_visited,path))
        loaded.sources.append(str(path))
    elif 'model_xml' in raw:
        loaded=_profile(path)
    else:
        unknown=set(raw)-set(DAILY_KEYS)-{'robot','advanced'}
        if unknown:raise ValueError('Unknown daily setting(s): '+', '.join(sorted(unknown)))
        selected=robot if robot is not None else raw.get('robot','lycus')
        if not isinstance(selected,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',selected):raise ValueError('Invalid robot name')
        loaded=_profile(path.parent.parent/'robots'/f'{selected}.json')
        if loaded.values['robot']!=selected:raise ValueError('Robot filename and profile name disagree')
        for key,keys in DAILY_KEYS.items():
            if key in raw:
                target=loaded.values
                for part in keys[:-1]:target=target[part]
                target[keys[-1]]=deepcopy(raw[key])
        advanced=raw.get('advanced',{})
        if not isinstance(advanced,dict) or set(advanced)-set(DEFAULTS):
            raise ValueError('advanced supports only sensor, mapping, ik, diagnostics and runtime')
        loaded.values=_merge(loaded.values,advanced);loaded.sources.append(str(path))
    _validate(loaded.values)
    return loaded


def print_summary(loaded):
    c=loaded.values
    print(f"[Config] robot={c['robot']} mode={c['mapping']['mode']} simulation={c['runtime']['simulation']} "
          f"collision={c['ik']['collision']['enabled']} max_velocity={c['runtime']['max_joint_velocity_rad_s']} rad/s")
    print(f'[Config] robot profile: {loaded.profile_path}')
    print('[Config] use --print-config for the full effective configuration')


def save_initial_qpos(loaded,values):
    """Save only the source profile's initial_qpos, never expanded defaults."""
    q=[float(v) for v in values]
    if len(q)!=len(loaded.values['initial_qpos']) or not all(math.isfinite(v) for v in q):raise ValueError('Invalid initial_qpos')
    path=loaded.profile_path;raw=_read(path)
    for key in ('robot','model_xml','frames','arms'):
        if raw.get(key)!=loaded.values.get(key):raise ValueError('Robot profile changed; reload before saving')
    raw['initial_qpos']=q;temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix=path.stem+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);json.dump(raw,stream,ensure_ascii=False,indent=2);stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    loaded.values['initial_qpos']=q

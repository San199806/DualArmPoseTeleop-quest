"""Explicit geometric safety checks, independent of contact masks and SDK IO.

Mesh queries use MuJoCo's convex collision representation. Sampled paths are
NOT continuous collision detection; unmodelled obstacles cannot be checked.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from .model_loader import proxy_prefix


@dataclass
class CollisionReport:
    safe: bool
    distance_m: float
    pair: str
    message: str


class CollisionGuard:
    def __init__(self, model, arms, settings=None):
        self.model = model
        settings = settings or {}
        self.enabled = bool(settings.get("enabled", False))
        self.minimum = float(settings.get("minimum_distance_m", .02))
        self.activation = float(settings.get("activation_distance_m", .08))
        self.step = float(settings.get("path_step", .01))
        self.max_samples = int(settings.get("max_path_samples", 100))
        if not (np.isfinite([self.minimum,self.activation,self.step]).all()
                and 0 < self.minimum < self.activation and self.step > 0
                and self.max_samples >= 2):
            raise ValueError("Invalid collision distance/path settings")
        self.data = mujoco.MjData(model)
        self.pairs = []
        self._last_q = None
        self._last_report = None
        if not self.enabled:
            return
        roots = [int(model.jnt_bodyid[model.joint(arms[s]["joint_names"][0]).id])
                 for s in ("left","right")]
        ancestors = []
        for b in range(model.nbody):
            chain = [b]
            while chain[-1] != 0:
                chain.append(int(model.body_parentid[chain[-1]]))
            ancestors.append(chain)
        owner = [next((i for i,r in enumerate(roots) if r in chain), -1)
                 for chain in ancestors]
        # Respect the model's explicit MJCF contact/exclude semantics. Keep
        # this separate from geom contact masks, which do not disable queries.
        excluded = {frozenset((int(s) >> 16, int(s) & 0xffff))
                    for s in model.exclude_signature}
        for pair in settings.get("exclude_body_pairs", []):
            if len(pair) != 2:
                raise ValueError("exclude_body_pairs entries must contain two body names")
            excluded.add(frozenset(model.body(n).id for n in pair))
        ignored = {model.geom(n).id for n in settings.get("exclude_geoms", [])}
        # Explicit opt-in for models with separate URDF visual/collision geoms.
        # Do not infer this from contact masks: those do not govern distance queries.
        groups = settings.get('geom_groups')
        if groups is not None:
            if (not isinstance(groups,list) or not groups or
                    any(type(g) is not int or not 0 <= g <= 5 for g in groups)):
                raise ValueError('geom_groups must be a nonempty list of group IDs 0..5')
            ignored.update(g for g in range(model.ngeom) if int(model.geom_group[g]) not in groups)
            for side in range(2):
                if not any(g not in ignored and owner[int(model.geom_bodyid[g])] == side
                           for g in range(model.ngeom)):
                    raise ValueError('No selected collision geometry for an arm')
        for spec in settings.get('mesh_slab_proxies',[]):
            name = spec['geom']
            if not any((model.geom(g).name or '').startswith(proxy_prefix(name)) for g in range(model.ngeom)):
                raise ValueError('Missing collision proxy geometry; use load_robot_model: '+name)
            ignored.add(model.geom(name).id)
        adjacent = int(settings.get("adjacent_body_hops", 1))
        if adjacent < 0 or adjacent > 2:
            raise ValueError("adjacent_body_hops must be 0, 1 or 2")
        for a in range(model.ngeom):
            for b in range(a+1,model.ngeom):
                ba,bb = int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
                if a in ignored or b in ignored or (owner[ba] < 0 and owner[bb] < 0):
                    continue
                if model.body_weldid[ba] == model.body_weldid[bb]:
                    continue
                if frozenset((ba,bb)) in excluded:
                    continue
                if ba != 0 and bb != 0 and (bb in ancestors[ba][1:adjacent+1] or ba in ancestors[bb][1:adjacent+1]):
                    continue
                self.pairs.append((a,b))
        if not self.pairs:
            raise ValueError("Collision checking enabled but no geometry pairs selected")
        self._pair_array = np.asarray(self.pairs,dtype=int)
        a,b = self._pair_array.T
        self._bounded = (model.geom_rbound[a] > 0) & (model.geom_rbound[b] > 0)
        self._radius = model.geom_rbound[a]+model.geom_rbound[b]+self.activation

    def pair_name(self, a,b):
        def name(g):
            return f"{self.model.body(int(self.model.geom_bodyid[g])).name}/{self.model.geom(g).name or 'geom#'+str(g)}"
        return name(a)+" <-> "+name(b)

    def distances(self, data):
        # Conservative bounding spheres skip only provably distant pairs.
        a,b = self._pair_array.T
        delta = data.geom_xpos[a]-data.geom_xpos[b]
        keep = (~self._bounded) | (np.einsum('ij,ij->i',delta,delta) <= self._radius**2)
        # Distances beyond activation are intentionally reported as capped.
        yield self.pairs[0][0], self.pairs[0][1], self.activation, np.zeros(6)
        for a,b in self._pair_array[keep]:
            points = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.model,data,a,b,self.activation,points))
            if not np.isfinite(distance):
                raise ValueError("Non-finite collision distance")
            yield a,b,distance,points

    def check(self, qpos):
        q = np.asarray(qpos,dtype=float)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            return CollisionReport(False,float("nan"),"invalid qpos","Invalid collision-check pose")
        if not self.enabled:
            return CollisionReport(True,float("inf"),"","Collision checking disabled")
        if self._last_q is not None and np.array_equal(q,self._last_q):
            return self._last_report
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model,self.data)
        nearest = min(self.distances(self.data),key=lambda item:item[2])
        a,b,d,_ = nearest
        safe = d >= self.minimum
        pair = self.pair_name(a,b)
        report = CollisionReport(safe,d,pair,
            f"{pair}: distance={d*1000:.2f} mm, required={self.minimum*1000:.1f} mm")
        self._last_q,self._last_report = q.copy(),report
        return report

    def check_path(self, start, end):
        first = self.check(start)
        if not first.safe or not self.enabled:
            return first
        last = self.check(end)
        if not last.safe:
            return last
        velocity = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model,velocity,1.0,np.asarray(start),np.asarray(end))
        count = max(1,math.ceil(float(np.max(np.abs(velocity),initial=0))/self.step))
        if count+1 > self.max_samples:
            return CollisionReport(False,first.distance_m,first.pair,"Collision path sampling budget exceeded; command rejected")
        nearest = min((first,last),key=lambda r:r.distance_m)
        for i in range(1,count):
            q = np.asarray(start,dtype=float).copy()
            mujoco.mj_integratePos(self.model,q,velocity,i/count)
            report = self.check(q)
            if not report.safe:
                return CollisionReport(False,report.distance_m,report.pair,
                                       f"Path sample {i}/{count}: "+report.message)
            if report.distance_m < nearest.distance_m:
                nearest = report
        return nearest

    def make_limit(self):
        # Lazy import lets the offline editor run without loading Mink.
        from mink.limits.limit import Constraint, Limit
        guard = self

        class DistanceLimit(Limit):
            def compute_qp_inequalities(self, configuration, dt):
                rows, bounds = [], []
                for a,b,d,points in guard.distances(configuration.data):
                    if d >= guard.activation:
                        continue
                    if abs(d) < 1e-9:
                        raise ValueError("Zero collision distance; pause and move to a safe starting pose")
                    normal = (points[3:]-points[:3])/d
                    jac_a,jac_b = np.zeros((3,guard.model.nv)),np.zeros((3,guard.model.nv))
                    mujoco.mj_jac(guard.model,configuration.data,jac_a,None,points[:3],int(guard.model.geom_bodyid[a]))
                    mujoco.mj_jac(guard.model,configuration.data,jac_b,None,points[3:],int(guard.model.geom_bodyid[b]))
                    rows.append(-normal @ (jac_b-jac_a))
                    bounds.append(.5*(d-guard.minimum))
                if not rows:
                    return Constraint()
                return Constraint(G=np.asarray(rows),h=np.asarray(bounds))
        return DistanceLimit()

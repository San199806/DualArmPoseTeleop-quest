import argparse
import json
import pathlib
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import messagebox, ttk

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dual_arm_pose_teleop.pose_symmetry import symmetrize_pose, generate_l_pose
from dual_arm_pose_teleop.configuration import load_configuration, save_initial_qpos
from dual_arm_pose_teleop.model_loader import load_robot_model
def parse_args():
    parser = argparse.ArgumentParser(description="Edit and save robot initial_qpos")
    parser.add_argument("--robot", default=None, help="Robot profile name in robots/")
    parser.add_argument("--config", default=None, help="custom robot profile JSON")
    return parser.parse_args()


class InitialQposEditor:
    def __init__(self, config_path, robot=None):
        self.loaded_config = load_configuration(config_path,robot=robot)
        self.config_path = self.loaded_config.profile_path
        self.config = self.loaded_config.values
        model_path = pathlib.Path(self.config["model_xml"])
        if not model_path.is_absolute():
            model_path = (self.loaded_config.project_root / model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Robot model not found: {model_path}")

        self.model = load_robot_model(model_path)
        self.data = mujoco.MjData(self.model)
        initial = np.asarray(self.config["initial_qpos"], dtype=np.float64)
        if initial.shape != (self.model.nq,):
            raise ValueError(
                f"initial_qpos has {initial.size} values; model requires {self.model.nq}"
            )
        self.original_qpos = initial.copy()
        self.symmetry_undo = None
        self.symmetry_future = None
        self.symmetry_executor = ThreadPoolExecutor(max_workers=1)
        self.refreshing_rows = False
        self.closed = False
        self.data.qpos[:] = initial
        mujoco.mj_forward(self.model, self.data)

        self.root = tk.Tk()
        self.root.title(f"Initial qpos editor - {self.config['robot']}")
        self.root.geometry("900x760")
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.rows = []
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(16, self._tick)

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header,
            text=f"Robot: {self.config['robot']}    config: {self.config_path}",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Drag a slider or type a value and press Enter. Hinge joints use degrees; slide joints use metres.",
        ).pack(anchor="w", pady=(4, 0))
        symmetry_bar = ttk.Frame(header)
        generate_bar = ttk.Frame(header)
        generate_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(generate_bar, text="Forward in base frame:").pack(side="left")
        self.forward_axis = tk.StringVar(value=self.config.get('editor', {}).get('forward_axis', '+y'))
        ttk.Combobox(generate_bar, textvariable=self.forward_axis,
                     values=("+x", "-x", "+y", "-y"), state="readonly", width=5).pack(side="left", padx=8)
        generate_button = ttk.Button(generate_bar, text="Generate symmetric L pose", command=self.generate_l)
        generate_button.pack(side="left")
        ttk.Label(header, text="Editor collision checks OFF: inspect and adjust manually before saving. Runtime protection is unchanged.").pack(anchor="w")
        symmetry_bar.pack(fill="x", pady=(8, 0))
        self.symmetry_buttons = [generate_button]
        for label, side in (("Mirror Left -> Right", "left"), ("Mirror Right -> Left", "right")):
            button = ttk.Button(symmetry_bar, text=label, command=lambda s=side: self.symmetrize(s))
            button.pack(side="left", padx=(0, 8))
            self.symmetry_buttons.append(button)
        self.undo_symmetry_button = ttk.Button(symmetry_bar, text="Undo symmetry", command=self.undo_symmetry, state="disabled")
        self.undo_symmetry_button.pack(side="left")
        ttk.Label(header, text="Mirror shoulder/elbow/wrist geometry; keep the source arm and base fixed. Save manually after preview.").pack(anchor="w", pady=(4, 0))

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.joint_frame = ttk.Frame(canvas, padding=(10, 0, 10, 10))
        window = canvas.create_window((0, 0), window=self.joint_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.joint_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )

        arm_joint_names = {
            name
            for side in ("left", "right")
            for name in self.config["arms"][side]["joint_names"]
        }
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            qpos_index = int(self.model.jnt_qposadr[joint_id])
            is_hinge = joint_type == mujoco.mjtJoint.mjJNT_HINGE
            scale_factor = 180.0 / np.pi if is_hinge else 1.0
            current = float(self.data.qpos[qpos_index] * scale_factor)
            low, high = self._display_range(joint_id, current, scale_factor, is_hinge)
            self._add_joint_row(
                name,
                qpos_index,
                current,
                low,
                high,
                scale_factor,
                "deg" if is_hinge else "m",
                name in arm_joint_names,
            )

        footer = ttk.Frame(self.root, padding=10)
        footer.pack(fill="x")
        self.status = tk.StringVar(value="Not saved")
        ttk.Label(footer, textvariable=self.status, wraplength=550).pack(side="left")
        ttk.Button(footer, text="Reset", command=self.reset).pack(side="right")
        ttk.Button(footer, text="Save to config", command=self.save).pack(
            side="right", padx=(0, 8)
        )

    def _display_range(self, joint_id, current, scale_factor, is_hinge):
        if self.model.jnt_limited[joint_id]:
            low, high = self.model.jnt_range[joint_id] * scale_factor
        elif is_hinge:
            low, high = -180.0, 180.0
        else:
            low, high = current - 1.0, current + 1.0
        if is_hinge and high - low > 720.0:
            low, high = -360.0, 360.0
        if not is_hinge and high - low > 4.0:
            low, high = current - 1.0, current + 1.0
        return min(float(low), current), max(float(high), current)

    def _add_joint_row(
        self, name, qpos_index, current, low, high, scale_factor, unit, is_arm
    ):
        row_index = len(self.rows)
        style = "Arm.TLabel" if is_arm else "TLabel"
        if is_arm and not ttk.Style().lookup("Arm.TLabel", "foreground"):
            ttk.Style().configure("Arm.TLabel", foreground="#0b65c2")
        ttk.Label(self.joint_frame, text=name, width=24, style=style).grid(
            row=row_index, column=0, sticky="w", pady=2
        )
        value = tk.DoubleVar(value=current)
        text_value = tk.StringVar(value=f"{current:.4f}")
        slider = tk.Scale(
            self.joint_frame,
            from_=low,
            to=high,
            orient="horizontal",
            resolution=0.1 if unit == "deg" else 0.001,
            showvalue=False,
            variable=value,
            command=lambda raw, idx=qpos_index, factor=scale_factor, text=text_value: self._from_slider(
                idx, factor, text, raw
            ),
        )
        slider.grid(row=row_index, column=1, sticky="ew", padx=8)
        entry = ttk.Entry(self.joint_frame, textvariable=text_value, width=12)
        entry.grid(row=row_index, column=2, padx=(0, 4))
        entry.bind(
            "<Return>",
            lambda _event, idx=qpos_index, factor=scale_factor, var=value, text=text_value: self._from_entry(
                idx, factor, var, text
            ),
        )
        entry.bind(
            "<FocusOut>",
            lambda _event, idx=qpos_index, factor=scale_factor, var=value, text=text_value: self._from_entry(
                idx, factor, var, text, show_error=False
            ),
        )
        ttk.Label(self.joint_frame, text=unit, width=5).grid(
            row=row_index, column=3, sticky="w"
        )
        self.joint_frame.columnconfigure(1, weight=1)
        self.rows.append((qpos_index, scale_factor, value, text_value))

    def _set_qpos(self, qpos_index, value):
        with self.viewer.lock():
            self.data.qpos[qpos_index] = value
            mujoco.mj_forward(self.model, self.data)
        self.status.set("Modified - not saved")

    def _from_slider(self, qpos_index, scale_factor, text_value, raw):
        if self.refreshing_rows:
            return
        display_value = float(raw)
        text_value.set(f"{display_value:.4f}")
        self._set_qpos(qpos_index, display_value / scale_factor)

    def _from_entry(
        self, qpos_index, scale_factor, slider_value, text_value, show_error=True
    ):
        if self.refreshing_rows:
            return
        try:
            display_value = float(text_value.get())
            if not np.isfinite(display_value):
                raise ValueError("Value must be finite")
        except ValueError:
            if show_error:
                messagebox.showerror("Invalid value", "Please enter a number")
            text_value.set(f"{float(slider_value.get()):.4f}")
            return
        if text_value.get() == f"{self.data.qpos[qpos_index]*scale_factor:.4f}":
            return
        slider_value.set(display_value)
        text_value.set(f"{display_value:.4f}")
        self._set_qpos(qpos_index, display_value / scale_factor)

    def reset(self):
        with self.viewer.lock():
            self.data.qpos[:] = self.original_qpos
            mujoco.mj_forward(self.model, self.data)
        self._refresh_rows()
        self.status.set("Reset to the config values")

    def _refresh_rows(self):
        self.refreshing_rows = True
        for qpos_index, scale_factor, slider_value, text_value in self.rows:
            display_value = float(self.data.qpos[qpos_index] * scale_factor)
            slider_value.set(display_value)
            text_value.set(f"{display_value:.4f}")
        self.root.after_idle(lambda: setattr(self, "refreshing_rows", False))

    def symmetrize(self, source_side):
        if self.symmetry_future is not None:
            return
        if not self._commit_entries():
            return
        with self.viewer.lock():
            self.symmetry_snapshot = self.data.qpos.copy()
        self.status.set(f"Solving {source_side} mirror... (not saved)")
        for button in self.symmetry_buttons:
            button.configure(state="disabled")
        self.symmetry_future = self.symmetry_executor.submit(
            symmetrize_pose, self.model, self.symmetry_snapshot, self.config, source_side
        )

    def generate_l(self):
        if self.symmetry_future is not None or not self._commit_entries():
            return
        with self.viewer.lock():
            self.symmetry_snapshot = self.data.qpos.copy()
        self.status.set("Generating symmetric L pose... (not saved)")
        for button in self.symmetry_buttons:
            button.configure(state="disabled")
        self.symmetry_future = self.symmetry_executor.submit(
            generate_l_pose, self.model, self.symmetry_snapshot, self.config, self.forward_axis.get()
        )

    def _finish_symmetry(self):
        future = self.symmetry_future
        if future is None or not future.done():
            return
        self.symmetry_future = None
        for button in self.symmetry_buttons:
            button.configure(state="normal")
        try:
            result = future.result()
        except Exception as error:
            self.status.set("Symmetry failed; pose unchanged")
            messagebox.showerror("Symmetry failed", str(error))
            return
        with self.viewer.lock():
            if not np.array_equal(self.data.qpos, self.symmetry_snapshot):
                self.status.set("Pose changed during solve; mirror discarded. Try again.")
                return
            if result.success:
                self.symmetry_undo = self.data.qpos.copy()
                self.data.qpos[:] = result.qpos
                mujoco.mj_forward(self.model, self.data)
        print("[Symmetry] " + result.message)
        if result.success:
            self._refresh_rows()
            self.undo_symmetry_button.configure(state="normal")
            self.status.set(result.message + " | Not saved")
        else:
            self.status.set("Pose unreachable within tolerance; pose unchanged")
            messagebox.showwarning("Pose not applied", result.message +
                "\n\nTolerance: 5 mm per point, 2 deg mirrored orientation.\nTry another forward axis or source pose.")

    def undo_symmetry(self):
        if self.symmetry_undo is None:
            return
        with self.viewer.lock():
            self.data.qpos[:] = self.symmetry_undo
            mujoco.mj_forward(self.model, self.data)
        self.symmetry_undo = None
        self.undo_symmetry_button.configure(state="disabled")
        self._refresh_rows()
        self.status.set("Symmetry undone - not saved")

    def save(self):
        if self.symmetry_future is not None:
            messagebox.showinfo("Solving", "Wait for symmetry solving to finish before saving.")
            return
        if not self._commit_entries():
            return
        with self.viewer.lock():
            qpos = self.data.qpos.copy()
        try:
            save_initial_qpos(self.loaded_config,qpos)
        except (ValueError,OSError) as error:
            messagebox.showerror('Save failed',str(error))
            return
        self.original_qpos = qpos
        self.status.set(f"Saved: {self.config_path}")
        messagebox.showinfo("Saved", f"initial_qpos saved to:\n{self.config_path}")

    def _commit_entries(self):
        """Include typed-but-not-entered values before mirror/save, atomically."""
        candidate = self.data.qpos.copy()
        try:
            for index, factor, _slider, text in self.rows:
                if text.get() != f"{candidate[index]*factor:.4f}":
                    candidate[index] = float(text.get())/factor
            if not np.all(np.isfinite(candidate)):
                raise ValueError("Joint values must be finite")
            for joint in range(self.model.njnt):
                if self.model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    continue
                if self.model.jnt_limited[joint]:
                    value = candidate[self.model.jnt_qposadr[joint]]
                    low, high = self.model.jnt_range[joint]
                    if value < low-1e-9 or value > high+1e-9:
                        raise ValueError(f"{self.model.joint(joint).name}: outside joint limits")
        except ValueError as error:
            messagebox.showerror("Invalid joint values", str(error))
            return False
        with self.viewer.lock():
            self.data.qpos[:] = candidate
            mujoco.mj_forward(self.model, self.data)
        self._refresh_rows()
        return True

    def _tick(self):
        if self.closed:
            return
        if not self.viewer.is_running():
            self.close()
            return
        self._finish_symmetry()
        self.viewer.sync()
        self.root.after(16, self._tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.symmetry_executor.shutdown(wait=False, cancel_futures=True)
        if self.viewer.is_running():
            self.viewer.close()
        if self.root.winfo_exists():
            self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    args = parse_args()
    selected_config = args.config or PROJECT_ROOT / 'config' / 'teleop.json'
    InitialQposEditor(selected_config,robot=args.robot if not args.config else None).run()

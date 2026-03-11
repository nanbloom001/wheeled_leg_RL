# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an exported JIT policy while driving the velocity command from a script or keyboard.

This script is intended for velocity-tracking locomotion policies whose observations
include ``generated_commands(..., command_name="base_velocity")``.

Example usages:

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_jit_with_command.py \
        --task Isaac-Velocity-Flat-QweDog-Play-v0 \
        --checkpoint logs/rsl_rl/qwe_dog/exported/policy.pt \
        --lin-vel-x 0.2 --ang-vel-z 0.0

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_jit_with_command.py \
        --task Isaac-Velocity-Flat-QweDog-Play-v0 \
        --checkpoint logs/rsl_rl/qwe_dog/exported/policy.pt \
        --command-sequence 0.0,0.0,0.0,0.0 \
        --command-sequence 3.0,0.2,0.0,0.0 \
        --command-sequence 6.0,0.4,0.0,0.3

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_jit_with_command.py \
        --task Isaac-Velocity-Flat-QweDog-Play-v0 \
        --checkpoint logs/rsl_rl/qwe_dog/exported/policy.pt \
        --keyboard-control
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from isaaclab.app import AppLauncher


@dataclass(frozen=True)
class CommandSegment:
    start_time_s: float
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_z: float
    heading: float | None = None


def parse_command_sequence(spec: str) -> CommandSegment:
    """Parse ``start,lin_x,lin_y,ang_z[,heading]`` into a command segment."""
    values = [value.strip() for value in spec.split(",") if value.strip()]
    if len(values) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "--command-sequence must be 'start_time,lin_vel_x,lin_vel_y,ang_vel_z[,heading]'"
        )

    start_time_s, lin_vel_x, lin_vel_y, ang_vel_z = map(float, values[:4])
    heading = float(values[4]) if len(values) == 5 else None
    return CommandSegment(start_time_s, lin_vel_x, lin_vel_y, ang_vel_z, heading)


parser = argparse.ArgumentParser(description="Play a JIT locomotion policy with scripted or keyboard commands.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the exported JIT policy (.pt).")
parser.add_argument("--task", type=str, required=True, help="Task name, usually a *-Play-v0 velocity task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--command-term", type=str, default="base_velocity", help="Name of the command term to drive.")
parser.add_argument("--lin-vel-x", type=float, default=0.0, help="Constant linear velocity command in x (m/s).")
parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Constant linear velocity command in y (m/s).")
parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Constant yaw velocity command (rad/s).")
parser.add_argument(
    "--heading",
    type=float,
    default=None,
    help="Optional heading target (rad). Only used when the command term supports heading control.",
)
parser.add_argument(
    "--command-sequence",
    dest="command_sequence",
    action="append",
    type=parse_command_sequence,
    default=None,
    help="Repeatable command schedule entry: start_time,lin_vel_x,lin_vel_y,ang_vel_z[,heading]",
)
parser.add_argument(
    "--keyboard-control",
    action="store_true",
    default=False,
    help="Drive the command from keyboard input in the Isaac Sim window instead of a scripted schedule.",
)
parser.add_argument(
    "--key-lin-speed",
    type=float,
    default=0.4,
    help="Linear x speed applied while W/S is pressed in keyboard mode.",
)
parser.add_argument(
    "--key-lat-speed",
    type=float,
    default=0.2,
    help="Linear y speed applied while Q/E is pressed in keyboard mode.",
)
parser.add_argument(
    "--key-yaw-speed",
    type=float,
    default=0.8,
    help="Yaw speed applied while A/D is pressed in keyboard mode.",
)
parser.add_argument(
    "--keyboard-mode",
    type=str,
    default="direct",
    choices=("direct", "incremental"),
    help="Keyboard control mode. direct is recommended. incremental is accepted for backward compatibility but is mapped to direct.",
)
parser.add_argument(
    "--key-lin-step",
    type=float,
    default=0.05,
    help="Increment applied to linear x target per key press in incremental mode.",
)
parser.add_argument(
    "--key-lat-step",
    type=float,
    default=0.05,
    help="Increment applied to linear y target per key press in incremental mode.",
)
parser.add_argument(
    "--key-yaw-step",
    type=float,
    default=0.1,
    help="Increment applied to yaw target per key press in incremental mode.",
)
parser.add_argument(
    "--key-lin-speed-step",
    type=float,
    default=0.05,
    help="Increment applied to the keyboard forward speed when adjusting speed from the keyboard.",
)
parser.add_argument(
    "--key-lat-speed-step",
    type=float,
    default=0.05,
    help="Increment applied to the keyboard strafe speed when adjusting speed from the keyboard.",
)
parser.add_argument(
    "--key-yaw-speed-step",
    type=float,
    default=0.1,
    help="Increment applied to the keyboard yaw speed when adjusting speed from the keyboard.",
)
parser.add_argument(
    "--max-lin-vel-x",
    type=float,
    default=1.0,
    help="Absolute clamp for linear x target in keyboard mode.",
)
parser.add_argument(
    "--max-lin-vel-y",
    type=float,
    default=0.5,
    help="Absolute clamp for linear y target in keyboard mode.",
)
parser.add_argument(
    "--max-ang-vel-z",
    type=float,
    default=1.5,
    help="Absolute clamp for yaw target in keyboard mode.",
)
parser.add_argument(
    "--keyboard-frame",
    type=str,
    default="auto",
    choices=("auto", "standard", "wavego"),
    help="Keyboard semantic frame mapping. standard: forward->+x, left->+y. wavego: forward->+y, right->+x.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--max-duration-s",
    type=float,
    default=None,
    help="Optional wall-clock duration after which playback stops.",
)
parser.add_argument(
    "--print-command-every",
    type=float,
    default=1.0,
    help="Print the active command every N seconds. Set <= 0 to disable.",
)
parser.add_argument(
    "--keep-play-settings",
    action="store_true",
    default=False,
    help="Do not disable observation corruption and common push events automatically.",
)
parser.add_argument(
    "--terrain-static-friction",
    type=float,
    default=None,
    help="Override terrain static friction for playback.",
)
parser.add_argument(
    "--terrain-dynamic-friction",
    type=float,
    default=None,
    help="Override terrain dynamic friction for playback.",
)
parser.add_argument(
    "--status-panel",
    action="store_true",
    default=False,
    help="Render a fixed-refresh terminal status panel instead of scrolling command logs.",
)
parser.add_argument(
    "--status-refresh-hz",
    type=float,
    default=5.0,
    help="Refresh rate for the terminal status panel.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import gymnasium as gym
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401


class KeyboardVelocityController:
    """Map keyboard input from the Isaac Sim app window to velocity commands."""

    def __init__(
        self,
        lin_speed: float,
        lat_speed: float,
        yaw_speed: float,
        heading: float | None,
        mode: str,
        lin_step: float,
        lat_step: float,
        yaw_step: float,
        max_lin_vel_x: float,
        max_lin_vel_y: float,
        max_ang_vel_z: float,
        frame_mapping: str,
        lin_speed_step: float,
        lat_speed_step: float,
        yaw_speed_step: float,
    ):
        if mode != "direct":
            print("[WARNING] keyboard mode 'incremental' is deprecated for this script and will be treated as 'direct'.")
        self._mode = "direct"
        self._lin_speed = lin_speed
        self._lat_speed = lat_speed
        self._yaw_speed = yaw_speed
        self._lin_step = lin_step
        self._lat_step = lat_step
        self._yaw_step = yaw_step
        self._max_lin_vel_x = abs(max_lin_vel_x)
        self._max_lin_vel_y = abs(max_lin_vel_y)
        self._max_ang_vel_z = abs(max_ang_vel_z)
        self._frame_mapping = frame_mapping
        self._lin_speed_step = abs(lin_speed_step)
        self._lat_speed_step = abs(lat_speed_step)
        self._yaw_speed_step = abs(yaw_speed_step)
        self._heading_target = heading
        self._pressed_keys: set[str] = set()
        self._target_forward = 0.0
        self._target_lateral_left = 0.0
        self._target_yaw_left = 0.0

        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard() if app_window is not None else None
        if self._keyboard is None:
            raise RuntimeError("Keyboard control requires an Isaac Sim application window.")

        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self.print_current_speed()

    def close(self):
        """Release the keyboard subscription."""
        if self._subscription is not None and hasattr(self._subscription, "unsubscribe"):
            self._subscription.unsubscribe()
        self._subscription = None

    def _on_keyboard_event(self, event):
        """Track currently pressed keys and handle stateless commands."""
        key_name = self._normalize_key_name(getattr(event, "input", None))
        if not key_name:
            return

        if event.type in {carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT}:
            if key_name == "SPACE":
                self._pressed_keys.clear()
                self._heading_target = None
                self._target_forward = 0.0
                self._target_lateral_left = 0.0
                self._target_yaw_left = 0.0
                return
            if key_name in {"Z", "X"}:
                direction = 1.0 if key_name == "X" else -1.0
                self._adjust_forward_speed(direction)
                return
            if key_name in {"T", "Y"}:
                direction = 1.0 if key_name == "Y" else -1.0
                self._adjust_strafe_speed(direction)
                return
            if key_name in {"C", "V"}:
                direction = 1.0 if key_name == "V" else -1.0
                self._adjust_yaw_speed(direction)
                return
            if key_name == "R":
                self._heading_target = 0.0 if self._heading_target is None else None
                return
            self._pressed_keys.add(key_name)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key_name)

    def current_command(self) -> CommandSegment:
        """Build the current command from the active key state."""
        forward = 0.0
        lateral_left = 0.0
        yaw_left = 0.0

        if "W" in self._pressed_keys or "UP" in self._pressed_keys:
            forward += self._lin_speed
        if "S" in self._pressed_keys or "DOWN" in self._pressed_keys:
            forward -= self._lin_speed
        if "A" in self._pressed_keys or "LEFT" in self._pressed_keys:
            lateral_left += self._lat_speed
        if "D" in self._pressed_keys or "RIGHT" in self._pressed_keys:
            lateral_left -= self._lat_speed
        if "Q" in self._pressed_keys:
            yaw_left += self._yaw_speed
        if "E" in self._pressed_keys:
            yaw_left -= self._yaw_speed

        return self._semantic_to_command(forward=forward, lateral_left=lateral_left, yaw_left=yaw_left)

    @staticmethod
    def _normalize_key_name(event_input) -> str:
        """Normalize carb keyboard identifiers into simple key names like W, LEFT, SPACE."""
        if event_input is None:
            return ""
        raw_name = event_input.name if hasattr(event_input, "name") else str(event_input)
        raw_name = str(raw_name).upper()
        if "." in raw_name:
            raw_name = raw_name.split(".")[-1]
        if raw_name.startswith("KEYBOARDINPUT "):
            raw_name = raw_name.split(" ")[-1]
        return raw_name.strip()

    def _semantic_to_command(self, forward: float, lateral_left: float, yaw_left: float) -> CommandSegment:
        """Convert semantic teleop commands into environment base_velocity components."""
        if self._frame_mapping == "wavego":
            lin_vel_x = -lateral_left
            lin_vel_y = forward
        else:
            lin_vel_x = forward
            lin_vel_y = lateral_left

        return CommandSegment(
            start_time_s=0.0,
            lin_vel_x=lin_vel_x,
            lin_vel_y=lin_vel_y,
            ang_vel_z=yaw_left,
            heading=self._heading_target,
        )

    def _adjust_forward_speed(self, direction: float):
        """Adjust the forward keyboard speed and print the updated value."""
        self._lin_speed = min(max(self._lin_speed + direction * self._lin_speed_step, 0.0), self._max_lin_vel_x)
        self.print_current_speed(prefix="[INFO] keyboard speed updated:")

    def _adjust_strafe_speed(self, direction: float):
        """Adjust the strafe keyboard speed and print the updated value."""
        self._lat_speed = min(max(self._lat_speed + direction * self._lat_speed_step, 0.0), self._max_lin_vel_y)
        self.print_current_speed(prefix="[INFO] keyboard speed updated:")

    def _adjust_yaw_speed(self, direction: float):
        """Adjust the yaw keyboard speed and print the updated value."""
        self._yaw_speed = min(max(self._yaw_speed + direction * self._yaw_speed_step, 0.0), self._max_ang_vel_z)
        self.print_current_speed(prefix="[INFO] keyboard speed updated:")

    def print_current_speed(self, prefix: str = "[INFO] keyboard speed:"):
        """Print the currently configured direct keyboard speeds."""
        print(
            prefix,
            f"forward={self._lin_speed:.3f}",
            f"strafe={self._lat_speed:.3f}",
            f"yaw={self._yaw_speed:.3f}",
        )

    def current_speed_settings(self) -> tuple[float, float, float]:
        """Return the currently configured direct keyboard speeds."""
        return self._lin_speed, self._lat_speed, self._yaw_speed

    @staticmethod
    def print_help(mode: str, frame_mapping: str):
        """Print the active keyboard bindings once at startup."""
        print("[INFO] Keyboard control enabled in the Isaac Sim window:")
        print("[INFO]   mode                  : direct")
        print(f"[INFO]   frame mapping         : {frame_mapping}")
        if mode != "direct":
            print("[INFO]   requested mode        : incremental (treated as direct)")
        print("[INFO]   W / S or Up / Down     : forward / backward while held")
        print("[INFO]   A / D or Left / Right : strafe left / right while held")
        print("[INFO]   Q / E                 : yaw left / right while held")
        print("[INFO]   Space                 : stop and clear pressed keys")
        print("[INFO]   Z / X                 : decrease / increase forward speed")
        print("[INFO]   T / Y                 : decrease / increase strafe speed")
        print("[INFO]   C / V                 : decrease / increase yaw speed")
        print("[INFO]   R                     : toggle heading target 0.0 on/off")


def load_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
    """Load a JIT policy from a local path or an Omniverse-accessible URI."""
    resolved_path = os.path.abspath(policy_path)
    if os.path.isfile(resolved_path):
        return torch.jit.load(resolved_path, map_location=device)

    result, _, file_content = omni.client.read_file(policy_path)
    if result != omni.client.Result.OK or file_content is None:
        raise FileNotFoundError(
            f"Unable to load policy checkpoint from '{policy_path}'. If this is a local file, pass an existing path."
        )

    file = io.BytesIO(memoryview(file_content).tobytes())
    return torch.jit.load(file, map_location=device)


def resolve_keyboard_frame_mapping(task_name: str, keyboard_frame: str) -> str:
    """Resolve keyboard semantic mapping from CLI setting and task name."""
    if keyboard_frame != "auto":
        return keyboard_frame

    if "WAVEGO" in task_name.upper():
        return "wavego"
    return "standard"


def build_command_schedule() -> list[CommandSegment]:
    """Build the active command schedule from CLI arguments."""
    if args_cli.command_sequence:
        return sorted(args_cli.command_sequence, key=lambda item: item.start_time_s)

    return [
        CommandSegment(
            start_time_s=0.0,
            lin_vel_x=args_cli.lin_vel_x,
            lin_vel_y=args_cli.lin_vel_y,
            ang_vel_z=args_cli.ang_vel_z,
            heading=args_cli.heading,
        )
    ]


def select_command(schedule: list[CommandSegment], elapsed_s: float) -> CommandSegment:
    """Select the command segment active at the given elapsed time."""
    active = schedule[0]
    for segment in schedule:
        if elapsed_s >= segment.start_time_s:
            active = segment
        else:
            break
    return active


def maybe_apply_play_overrides(env_cfg):
    """Disable common training-time randomness for scripted playback."""
    if args_cli.keep_play_settings:
        apply_env_parameter_overrides(env_cfg)
        return

    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False
        for obs_name in ("base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "height_scan"):
            obs_term = getattr(env_cfg.observations.policy, obs_name, None)
            if obs_term is not None and hasattr(obs_term, "noise"):
                obs_term.noise = None

    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        if hasattr(env_cfg.actions.joint_pos, "noise"):
            env_cfg.actions.joint_pos.noise = None

    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "physics_material"):
            env_cfg.events.physics_material = None
        if hasattr(env_cfg.events, "add_base_mass"):
            env_cfg.events.add_base_mass = None
        if hasattr(env_cfg.events, "base_com"):
            env_cfg.events.base_com = None
        if hasattr(env_cfg.events, "reset_base") and hasattr(env_cfg.events.reset_base, "params"):
            env_cfg.events.reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            }

    apply_env_parameter_overrides(env_cfg)


def apply_env_parameter_overrides(env_cfg):
    """Apply explicit playback parameter overrides from the CLI."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
        material = getattr(env_cfg.scene.terrain, "physics_material", None)
        if material is not None:
            if args_cli.terrain_static_friction is not None:
                material.static_friction = args_cli.terrain_static_friction
            if args_cli.terrain_dynamic_friction is not None:
                material.dynamic_friction = args_cli.terrain_dynamic_friction


def apply_velocity_command(command_term, segment: CommandSegment):
    """Overwrite the command term so the next observation carries the active command."""
    if command_term.command.shape[1] < 3:
        raise RuntimeError(
            f"Command term '{args_cli.command_term}' has shape {tuple(command_term.command.shape)}; expected at least 3 values."
        )

    command_term.command[:, 0] = segment.lin_vel_x
    command_term.command[:, 1] = segment.lin_vel_y
    command_term.command[:, 2] = segment.ang_vel_z

    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "time_left"):
        command_term.time_left[:] = 1.0e9

    if hasattr(command_term, "heading_target") and segment.heading is not None:
        command_term.heading_target[:] = segment.heading
    if hasattr(command_term, "is_heading_env"):
        use_heading = bool(segment.heading is not None and getattr(command_term.cfg, "heading_command", False))
        command_term.is_heading_env[:] = use_heading


def extract_policy_obs(obs: Mapping[str, torch.Tensor | dict[str, torch.Tensor]] | torch.Tensor) -> torch.Tensor:
    """Extract the tensor expected by exported locomotion JIT policies."""
    if isinstance(obs, Mapping):
        if "policy" not in obs:
            raise KeyError("Expected the environment observations to contain a 'policy' entry.")
        policy_obs = obs["policy"]
        if not isinstance(policy_obs, torch.Tensor):
            raise TypeError("Expected obs['policy'] to be a torch.Tensor.")
        return policy_obs
    return obs


def format_vector(values: torch.Tensor) -> str:
    """Format a 1-D tensor for compact status output."""
    return "[" + ", ".join(f"{value:.3f}" for value in values.tolist()) + "]"


def render_status_panel(
    *,
    elapsed_s: float,
    active_segment: CommandSegment,
    env,
    env_cfg,
    policy_obs: torch.Tensor,
    keyboard_controller: KeyboardVelocityController | None,
):
    """Render a fixed-refresh terminal dashboard."""
    obs0 = policy_obs[0].detach().cpu()
    lines = [
        "IsaacLab Keyboard Teleop Panel",
        "=" * 72,
        f"time={elapsed_s:7.2f}s  task={args_cli.task}  envs={env.num_envs}  step_dt={env.step_dt:.3f}s  device={args_cli.device}",
        f"command term={args_cli.command_term}  frame={resolve_keyboard_frame_mapping(args_cli.task, args_cli.keyboard_frame)}  headless={args_cli.headless}",
        f"active command: vx={active_segment.lin_vel_x:.3f}  vy={active_segment.lin_vel_y:.3f}  wz={active_segment.ang_vel_z:.3f}  heading={'None' if active_segment.heading is None else f'{active_segment.heading:.3f}'}",
    ]

    if keyboard_controller is not None:
        forward_speed, strafe_speed, yaw_speed = keyboard_controller.current_speed_settings()
        lines.append(
            f"keyboard speed: forward={forward_speed:.3f}  strafe={strafe_speed:.3f}  yaw={yaw_speed:.3f}"
        )

    terrain_material = getattr(getattr(getattr(env_cfg, "scene", None), "terrain", None), "physics_material", None)
    if terrain_material is not None:
        lines.append(
            f"terrain friction: static={getattr(terrain_material, 'static_friction', 'n/a')}  dynamic={getattr(terrain_material, 'dynamic_friction', 'n/a')}"
        )

    lines.extend(
        [
            "-" * 72,
            f"obs[0:3]   base_lin_vel        {format_vector(obs0[0:3])}",
            f"obs[3:6]   base_ang_vel        {format_vector(obs0[3:6])}",
            f"obs[6:9]   projected_gravity   {format_vector(obs0[6:9])}",
            f"obs[9:12]  velocity_commands   {format_vector(obs0[9:12])}",
            f"obs[12:24] joint_pos_rel       {format_vector(obs0[12:24])}",
            f"obs[24:36] joint_vel           {format_vector(obs0[24:36])}",
            f"obs[36:48] last_action         {format_vector(obs0[36:48])}",
        ]
    )

    panel_text = "\n".join(lines)
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[J" + panel_text + "\n")
        sys.stdout.flush()
    else:
        print(panel_text)


def main():
    """Main function."""
    if args_cli.keyboard_control and args_cli.headless:
        raise RuntimeError("Keyboard control requires GUI mode. Remove --headless when using --keyboard-control.")
    if args_cli.keyboard_control and not os.environ.get("DISPLAY"):
        print("[WARNING] DISPLAY is not set. Isaac Sim may start without a visible window, and keyboard input will not work.")

    policy = load_policy(args_cli.checkpoint, args_cli.device)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    maybe_apply_play_overrides(env_cfg)

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg).unwrapped)
    if not isinstance(env, ManagerBasedRLEnv):
        raise RuntimeError("This script requires a manager-based environment with a command_manager.")

    command_term = env.command_manager.get_term(args_cli.command_term)
    schedule = build_command_schedule()
    keyboard_controller = None
    if args_cli.keyboard_control:
        frame_mapping = resolve_keyboard_frame_mapping(args_cli.task, args_cli.keyboard_frame)
        keyboard_controller = KeyboardVelocityController(
            lin_speed=args_cli.key_lin_speed,
            lat_speed=args_cli.key_lat_speed,
            yaw_speed=args_cli.key_yaw_speed,
            heading=args_cli.heading,
            mode=args_cli.keyboard_mode,
            lin_step=args_cli.key_lin_step,
            lat_step=args_cli.key_lat_step,
            yaw_step=args_cli.key_yaw_step,
            max_lin_vel_x=args_cli.max_lin_vel_x,
            max_lin_vel_y=args_cli.max_lin_vel_y,
            max_ang_vel_z=args_cli.max_ang_vel_z,
            frame_mapping=frame_mapping,
            lin_speed_step=args_cli.key_lin_speed_step,
            lat_speed_step=args_cli.key_lat_speed_step,
            yaw_speed_step=args_cli.key_yaw_speed_step,
        )
        KeyboardVelocityController.print_help(args_cli.keyboard_mode, frame_mapping)
        print("[INFO] Focus the Isaac Sim window to capture key presses.")

    obs, _ = env.reset()
    initial_command = keyboard_controller.current_command() if keyboard_controller is not None else schedule[0]
    apply_velocity_command(command_term, initial_command)
    obs = env.observation_manager.compute(update_history=False)

    start_wall_time = time.time()
    last_print_time = -float("inf")
    last_panel_time = -float("inf")

    if args_cli.status_panel and sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    with torch.inference_mode():
        while simulation_app.is_running():
            loop_start_time = time.time()

            actions = policy(extract_policy_obs(obs))
            obs, _, _, _, _ = env.step(actions)

            elapsed_s = time.time() - start_wall_time
            active_segment = (
                keyboard_controller.current_command()
                if keyboard_controller is not None
                else select_command(schedule, elapsed_s)
            )
            apply_velocity_command(command_term, active_segment)
            obs = env.observation_manager.compute(update_history=False)

            if args_cli.status_panel:
                panel_interval = 0.2 if args_cli.status_refresh_hz <= 0 else 1.0 / args_cli.status_refresh_hz
                if elapsed_s - last_panel_time >= panel_interval:
                    render_status_panel(
                        elapsed_s=elapsed_s,
                        active_segment=active_segment,
                        env=env,
                        env_cfg=env_cfg,
                        policy_obs=extract_policy_obs(obs),
                        keyboard_controller=keyboard_controller,
                    )
                    last_panel_time = elapsed_s
            elif args_cli.print_command_every > 0 and elapsed_s - last_print_time >= args_cli.print_command_every:
                heading_text = "None" if active_segment.heading is None else f"{active_segment.heading:.3f}"
                print(
                    "[INFO] active command:",
                    f"t={elapsed_s:.2f}s",
                    f"vx={active_segment.lin_vel_x:.3f}",
                    f"vy={active_segment.lin_vel_y:.3f}",
                    f"wz={active_segment.ang_vel_z:.3f}",
                    f"heading={heading_text}",
                )
                last_print_time = elapsed_s

            if args_cli.max_duration_s is not None and elapsed_s >= args_cli.max_duration_s:
                break

            sleep_time = env.step_dt - (time.time() - loop_start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)

    if keyboard_controller is not None:
        keyboard_controller.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
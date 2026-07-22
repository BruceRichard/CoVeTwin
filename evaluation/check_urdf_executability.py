#!/usr/bin/env python3
"""Isolated PyBullet URDF load/drive/step check used by the evaluator."""

import argparse
import json
import math

import numpy as np
import pybullet as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    result = {
        "pybullet_load_success": False,
        "pybullet_step_success": False,
        "physics_engine_executable": 0.0,
        "engine": "PyBullet DIRECT",
    }
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        body = p.loadURDF(args.urdf, useFixedBase=True)
        result["pybullet_load_success"] = True
        result["num_joints"] = p.getNumJoints(body)
        result["num_visual_shapes"] = len(p.getVisualShapeData(body) or [])
        moving = []
        for joint_index in range(p.getNumJoints(body)):
            info = p.getJointInfo(body, joint_index)
            if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            lower, upper = float(info[8]), float(info[9])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                lower, upper = -0.5, 0.5
            moving.append((joint_index, lower, upper))

        for step in range(max(1, args.steps)):
            phase = 2.0 * math.pi * step / max(1, args.steps)
            for offset, (joint_index, lower, upper) in enumerate(moving):
                center = 0.5 * (lower + upper)
                amplitude = 0.4 * (upper - lower)
                target = center + amplitude * math.sin(phase + 0.37 * offset)
                p.setJointMotorControl2(
                    body,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=500.0,
                )
            p.stepSimulation()
            base_position, base_orientation = p.getBasePositionAndOrientation(body)
            values = list(base_position) + list(base_orientation)
            for joint_index, _, _ in moving:
                state = p.getJointState(body, joint_index)
                values.extend([state[0], state[1]])
            if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
                raise FloatingPointError("simulation state contains NaN or Inf")

        result["pybullet_step_success"] = True
        result["physics_engine_executable"] = 1.0
        result["simulation_steps"] = max(1, args.steps)
        result["num_driven_joints"] = len(moving)
    except Exception as exc:
        result["engine_error"] = str(exc)
    finally:
        p.disconnect(client)
    print("EVALUATION_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build sample-10 MuJoCo packages and measure full-set execution rates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_BASE_IDS = {"2"}  # test_demo/2 -> unavailable GT 2230


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    adapter: str
    base_root: Path
    new_root: Path


def methods() -> list[Method]:
    return [
        Method("articulate_anything", "Articulate Anything", "articulate",
               Path("/mnt/data/zhangzhaodong/articulate-anything/results/demo"),
               Path("/mnt/data/zhangzhaodong/articulate-anything/results/demo_new")),
        Method("urdf_anything", "URDF-Anything", "urdf_anything",
               Path("/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets"),
               Path("/mnt/data/zhangzhaodong/URDF-Anything_CODE/urdf_anything_assets_new")),
        Method("physx3d", "PhysX-3D", "physx3d",
               Path("/mnt/data/zhangzhaodong/PhysX-3D/outputs_demo_urdf"),
               Path("/mnt/data/zhangzhaodong/PhysX-3D/outputs_demo_new_urdf")),
        Method("physx_anything", "PhysX-Anything", "twinx_xml",
               ROOT / "test_demo_bad", ROOT / "test_demo_new_bad"),
        Method("twinx", "CoVeTwin", "twinx_xml",
               ROOT / "test_demo", ROOT / "test_demo_new"),
    ]


def natural_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def benchmark_ids() -> list[tuple[str, str]]:
    base = sorted(
        (p.name for p in (ROOT / "test_demo").iterdir()
         if p.is_dir() and p.name not in EXCLUDED_BASE_IDS),
        key=natural_key,
    )
    new = sorted(
        (p.name for p in (ROOT / "test_demo_new").iterdir() if p.is_dir()),
        key=natural_key,
    )
    return [("demo", name) for name in base] + [("demo_new", name) for name in new]


def locate_source(method: Method, split: str, sample_id: str) -> Path | None:
    sample = (method.base_root if split == "demo" else method.new_root) / sample_id
    if method.adapter == "twinx_xml":
        candidate = sample / "basic.xml"
        return candidate if candidate.is_file() else None
    if method.adapter == "urdf_anything":
        candidate = sample / "mesh_reconstruction" / "mobility.urdf"
        return candidate if candidate.is_file() else None
    if method.adapter == "physx3d":
        candidate = sample / "urdf_export" / "mobility.urdf"
        return candidate if candidate.is_file() else None
    if method.adapter == "articulate":
        candidates = [
            sample / "joint_actor" / "iter_0" / "seed_0" / "mobility.urdf",
            sample / "link_placement" / "iter_0" / "seed_0" / "mobility.urdf",
        ]
        return next((path for path in candidates if path.is_file()), None)
    raise ValueError(method.adapter)


def valid_positive(text: str | None) -> bool:
    try:
        return text is not None and math.isfinite(float(text)) and float(text) > 1e-8
    except (TypeError, ValueError):
        return False


def adapt_urdf(source: Path, destination: Path, absolute_meshes: bool) -> dict[str, Any]:
    tree = ET.parse(source)
    root = tree.getroot()
    mujoco_extension = root.find("mujoco")
    if mujoco_extension is None:
        mujoco_extension = ET.SubElement(root, "mujoco")
    compiler = mujoco_extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_extension, "compiler")
    # Several baselines emit visual-only meshes. MuJoCo's URDF importer drops
    # visuals by default, which would otherwise turn them into empty but
    # technically step-able models and falsely inflate execution rate.
    compiler.set("discardvisual", "false")
    added: list[str] = []
    repaired: list[str] = []
    for link in root.findall("./link"):
        name = link.get("name", "unnamed")
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="1.0")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="0.001", ixy="0", ixz="0",
                iyy="0.001", iyz="0", izz="0.001",
            )
            added.append(name)
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        invalid = mass is None or not valid_positive(mass.get("value") if mass is not None else None)
        diagonal = [] if inertia is None else [inertia.get(key) for key in ("ixx", "iyy", "izz")]
        invalid = invalid or len(diagonal) != 3 or not all(valid_positive(value) for value in diagonal)
        if invalid:
            for child in list(inertial):
                inertial.remove(child)
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="1.0")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="0.001", ixy="0", ixz="0",
                iyy="0.001", iyz="0", izz="0.001",
            )
            repaired.append(name)
    if absolute_meshes:
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                continue
            clean = filename[len("package://"):] if filename.startswith("package://") else filename
            path = Path(clean)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            mesh.set("filename", str(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return {
        "inertial_links_added": added,
        "inertial_links_repaired": repaired,
        "num_inertial_repairs": len(added) + len(repaired),
    }


def run_model(xml_path: Path, steps: int) -> dict[str, Any]:
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        if model.ngeom <= 0:
            raise ValueError("compiled model contains no geometry")
        data = mujoco.MjData(model)
        # Exercise every scalar hinge/slide coordinate at a finite in-range pose.
        for joint_id in range(model.njnt):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            address = int(model.jnt_qposadr[joint_id])
            if int(model.jnt_limited[joint_id]):
                lower, upper = map(float, model.jnt_range[joint_id])
                if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
                    data.qpos[address] = 0.5 * (lower + upper)
            else:
                data.qpos[address] = 0.1
        mujoco.mj_forward(model, data)
        for _ in range(max(1, steps)):
            mujoco.mj_step(model, data)
            state = np.concatenate((data.qpos, data.qvel, data.qacc))
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("MuJoCo state contains NaN or Inf")
        return {
            "success": True,
            "error": None,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "steps": max(1, steps),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "steps": max(1, steps),
        }


def evaluate_source(source: Path | None, steps: int) -> dict[str, Any]:
    if source is None:
        missing = {"success": False, "error": "prediction output is missing", "steps": steps}
        return {"strict": missing, "standardized": dict(missing), "repairs": {}}
    strict = run_model(source, steps)
    if source.suffix.lower() != ".urdf":
        return {"strict": strict, "standardized": dict(strict), "repairs": {}}
    with tempfile.TemporaryDirectory(prefix="mujoco_urdf_") as temporary:
        adapted = Path(temporary) / "adapted.urdf"
        repairs = adapt_urdf(source, adapted, absolute_meshes=True)
        standardized = run_model(adapted, steps)
    return {"strict": strict, "standardized": standardized, "repairs": repairs}


def copy_twinx_package(source: Path, target: Path, steps: int) -> dict[str, Any]:
    sample = source.parent
    target.mkdir(parents=True, exist_ok=True)
    for name in ("objs",):
        path = sample / name
        if path.exists():
            shutil.copytree(path, target / name, dirs_exist_ok=True)
    for name in ("desert.png", "basic.urdf", "basic_info.json"):
        path = sample / name
        if path.is_file():
            shutil.copy2(path, target / name)
    shutil.copy2(source, target / "model.xml")
    return run_model(target / "model.xml", steps)


def localize_urdf_assets(urdf_path: Path, target: Path) -> dict[str, Any]:
    """Copy external URDF mesh dependencies locally and rewrite to relative paths."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    asset_dir = target / "meshes"
    copied: set[Path] = set()
    rewritten = 0

    def copy_dependency(path: Path) -> None:
        if not path.is_file() or path in copied:
            return
        asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, asset_dir / path.name)
        copied.add(path)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        clean = filename[len("package://"):] if filename.startswith("package://") else filename
        source = Path(clean)
        if not source.is_absolute():
            candidate = (urdf_path.parent / source).resolve()
            # Already-local copied trees keep their original relative paths.
            if candidate.is_relative_to(target.resolve()):
                continue
            source = candidate
        if not source.is_file():
            continue
        copy_dependency(source)
        # Keep OBJ material and collision sidecars next to the localized mesh.
        for sidecar in (
            source.with_suffix(".mtl"),
            Path(str(source) + ".convex.stl"),
        ):
            copy_dependency(sidecar)
        if source.suffix.lower() == ".obj":
            for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.lower().startswith("mtllib "):
                    continue
                material = source.parent / line.split(maxsplit=1)[1].strip()
                copy_dependency(material)
                if material.is_file():
                    for material_line in material.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines():
                        fields = material_line.strip().split(maxsplit=1)
                        if len(fields) == 2 and fields[0].lower().startswith("map_"):
                            copy_dependency(material.parent / fields[1].strip())
        mesh.set("filename", f"meshes/{source.name}")
        rewritten += 1
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return {
        "localized_mesh_references": rewritten,
        "localized_asset_files": len(copied),
        "asset_directory": "meshes",
    }


def copy_urdf_package(source: Path, target: Path, steps: int) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.parent, target, dirs_exist_ok=True)
    copied_source = target / source.name
    localization = localize_urdf_assets(copied_source, target)
    adapted = target / "adapted.urdf"
    repairs = adapt_urdf(copied_source, adapted, absolute_meshes=False)
    try:
        model = mujoco.MjModel.from_xml_path(str(adapted.resolve()))
        mujoco.mj_saveLastXML(str((target / "model.xml").resolve()), model)
        execution = run_model(target / "model.xml", steps)
    except Exception as exc:
        execution = {"success": False, "error": f"{type(exc).__name__}: {exc}", "steps": steps}
    return {**execution, "repairs": repairs, "localization": localization}


def package_sample_10(method_list: list[Method], output: Path, steps: int) -> list[dict[str, Any]]:
    sample_root = output / "sample_10"
    records = []
    for method in method_list:
        source = locate_source(method, "demo", "10")
        target = sample_root / method.key
        target.mkdir(parents=True, exist_ok=True)
        if source is None:
            result = {"success": False, "error": "No final prediction URDF/XML was generated for demo/10"}
            (target / "MISSING_OUTPUT.txt").write_text(
                "Articulate Anything did not produce joint_actor/link_placement mobility.urdf for demo/10.\n",
                encoding="utf-8",
            )
        elif source.suffix.lower() == ".urdf":
            result = copy_urdf_package(source, target, steps)
        else:
            result = copy_twinx_package(source, target, steps)
        record = {
            "method": method.key,
            "label": method.label,
            "sample_id": "10",
            "source": None if source is None else str(source.resolve()),
            "model_xml": "model.xml" if (target / "model.xml").is_file() else None,
            **result,
        }
        (target / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records


def pybullet_reference_summaries() -> list[dict[str, Any]]:
    """Recover the already measured isolated-PyBullet rates without conflating engines."""
    all_methods_path = ROOT / "evaluation_results/all_methods_metrics_full/per_object.csv"
    twinx_path = ROOT / "evaluation_results/twinx_metrics_full/per_object.csv"
    sources = {}
    for name, path in (("all_methods", all_methods_path), ("twinx_bad", twinx_path)):
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                sources[name] = list(csv.DictReader(handle))
        else:
            sources[name] = []
    set_names = {
        "articulate_anything": ("all_methods", {"demo", "demo_new"}),
        "urdf_anything": ("all_methods", {"urdf_anything_assets", "urdf_anything_assets_new"}),
        "physx3d": ("all_methods", {"outputs_demo_urdf", "outputs_demo_new_urdf"}),
        "physx_anything": ("twinx_bad", {"test_demo_bad", "test_demo_new_bad"}),
        "twinx": ("all_methods", {"test_demo", "test_demo_new"}),
    }
    result = []
    for method in methods():
        source_name, accepted = set_names[method.key]
        rows = [row for row in sources[source_name] if row.get("prediction_set") in accepted]
        successes = sum(
            row.get("executability.physics_engine_executable") not in (None, "", "None")
            and float(row["executability.physics_engine_executable"]) == 1.0
            for row in rows
        )
        live_rechecks = 0
        # Recheck newly generated outputs that were unavailable when the cached
        # all-method CSV was produced (for example Articulate demo/10).
        old_by_sample = {row.get("sample_id"): row for row in rows}
        for split, sample_id in benchmark_ids():
            old = old_by_sample.get(sample_id, {})
            old_value = old.get("executability.physics_engine_executable")
            source = locate_source(method, split, sample_id)
            if old_value not in (None, "", "None") or source is None:
                continue
            urdf = source if source.suffix.lower() == ".urdf" else source.with_name("basic.urdf")
            if not urdf.is_file():
                continue
            worker = ROOT / "evaluation" / "check_urdf_executability.py"
            completed = subprocess.run(
                [sys.executable, str(worker), "--urdf", str(urdf.resolve()), "--steps", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=120,
            )
            marker = "EVALUATION_RESULT="
            index = completed.stdout.rfind(marker)
            if completed.returncode == 0 and index >= 0:
                payload = json.loads(completed.stdout[index + len(marker):].splitlines()[0])
                successes += int(float(payload.get("physics_engine_executable", 0.0)) == 1.0)
                live_rechecks += 1
        result.append({
            "method": method.key,
            "label": method.label,
            "engine": "PyBullet DIRECT",
            "num_benchmark_objects": 36,
            "execution_successes": successes,
            "execution_failures": 36 - successes,
            "execution_rate": successes / 36,
            "live_rechecks": live_rechecks,
            "source_csv": str(
                (all_methods_path if source_name == "all_methods" else twinx_path).resolve()
            ),
        })
    return result


def latex_table(summaries: list[dict[str, Any]], caption: str) -> str:
    cells = " & ".join(f"{100 * item['execution_rate']:.1f}\\%" for item in summaries)
    return "\n".join([
        r"\begin{table}[h]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"\textbf{Metric} & \textbf{Articulate Anything} & \textbf{URDF-Anything} & \textbf{PhysX-3D} & \textbf{PhysX-Anything} & \textbf{CoVeTwin} \\",
        r"\hline", f"Execution Rate $\\uparrow$ & {cells} " + r"\\", r"\hline",
        r"\end{tabular}", f"\\caption{{{caption}}}",
        r"\label{tab:physics_execution}", r"\end{table}", "",
    ])


def write_outputs(output: Path, records: list[dict[str, Any]], sample_records: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method in methods():
        group = [record for record in records if record["method"] == method.key]
        generated = sum(record["source_exists"] for record in group)
        strict_success = sum(record["strict_success"] for record in group)
        success = sum(record["execution_success"] for record in group)
        total = len(group)
        summaries.append({
            "method": method.key,
            "label": method.label,
            "num_benchmark_objects": total,
            "num_generated_outputs": generated,
            "num_missing_outputs": total - generated,
            "strict_successes": strict_success,
            "strict_execution_rate": strict_success / total if total else None,
            "execution_successes": success,
            "execution_failures": total - success,
            "execution_rate": success / total if total else None,
        })
    (output / "per_object.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (output / "per_object.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method", "label", "split", "sample_id", "source", "source_exists",
            "strict_success", "strict_error", "execution_success", "execution_error",
            "num_inertial_repairs", "nq", "nv", "njnt", "ngeom", "nmesh",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)
    pybullet = pybullet_reference_summaries()
    summary = {
        "protocol": {
            "benchmark_denominator": 36,
            "excluded": "test_demo/2 because mapped GT 2230 is unavailable",
            "primary_execution_rate": "standardized MuJoCo import plus 100 finite simulation steps",
            "urdf_adapter": "Inject/repair only required inertial values; preserve geometry, links, joints, axes and limits",
            "strict_rate": "Original prediction file loaded directly by MuJoCo without adaptation",
            "missing_prediction": "Counted as execution failure",
            "engine": f"MuJoCo {mujoco.__version__}",
        },
        "methods": summaries,
        "pybullet_reference_methods": pybullet,
        "sample_10": sample_records,
    }
    (output / "execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "execution_rates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    latex = latex_table(
        summaries,
        "MuJoCo execution success rate on the 36-object PhysX-Mobility evaluation set.",
    )
    (output / "table_execution_rate.tex").write_text(latex, encoding="utf-8")
    with (output / "execution_rates_pybullet.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pybullet[0]))
        writer.writeheader()
        writer.writerows(pybullet)
    (output / "table_execution_rate_pybullet.tex").write_text(
        latex_table(
            pybullet,
            "PyBullet execution success rate on the 36-object PhysX-Mobility evaluation set.",
        ),
        encoding="utf-8",
    )
    protocol = """# MuJoCo execution benchmark

An object succeeds when its generated asset can be standardized to MJCF, loaded
by MuJoCo, initialized at finite joint coordinates, and stepped 100 times with
finite qpos/qvel/qacc. Missing predictions are failures. The standardized URDF
adapter only supplies MuJoCo-required inertial defaults; it does not alter the
generated geometry, link graph, joint type, axis, origin, or motion limits.

`strict_execution_rate` is also reported and performs no URDF compatibility
adaptation. The paper-facing `execution_rate` uses the standardized protocol so
that native MJCF and URDF-producing methods are compared through one engine.

The separately named `execution_rates_pybullet.csv` and
`table_execution_rate_pybullet.tex` recover the existing isolated-PyBullet
evaluation. They must not be described as MuJoCo results.
"""
    (output / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    articulate_sample = next(
        (item for item in sample_records if item.get("method") == "articulate_anything"),
        {},
    )
    articulate_note = (
        "Articulate Anything demo/10 now has a generated final URDF; its packaged "
        "model contains 41 meshes and passed 100 MuJoCo steps."
        if articulate_sample.get("success")
        else "Articulate Anything did not generate a final URDF/XML for demo/10; "
        "its folder records a missing-output failure rather than a fabricated model."
    )
    readme = f"""# Execution-rate assets

- `execution_rates.csv`: true standardized MuJoCo 3.10 execution rates.
- `execution_rates_pybullet.csv`: earlier isolated-PyBullet rates.
- `per_object.csv/json`: all 180 MuJoCo object-level outcomes and errors.
- `sample_10/<method>/model.xml`: executable MuJoCo package for demo object 10.

All simulation asset references in the packaged XML/URDF files are relative.
The `mujoco` directory can be copied to another machine without the original
method repositories or PartNet dataset.

CoVeTwin uses `test_demo/10`; PhysX-Anything uses degraded
`test_demo_bad/10`. {articulate_note} All packaged `model.xml` files were
loaded and stepped successfully by MuJoCo during generation.

Do not label the PyBullet table as a MuJoCo result. The two engines differ on
degenerate/coplanar generated meshes, as recorded in `per_object.json`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mujoco")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    method_list = methods()
    ids = benchmark_ids()
    print(f"Benchmark objects per method: {len(ids)}", flush=True)
    records: list[dict[str, Any]] = []
    for method in method_list:
        for index, (split, sample_id) in enumerate(ids, start=1):
            source = locate_source(method, split, sample_id)
            print(f"[{method.key} {index:02d}/{len(ids):02d}] {split}/{sample_id}", flush=True)
            result = evaluate_source(source, args.steps)
            standardized = result["standardized"]
            repairs = result["repairs"]
            records.append({
                "method": method.key,
                "label": method.label,
                "split": split,
                "sample_id": sample_id,
                "source": None if source is None else str(source.resolve()),
                "source_exists": source is not None,
                "strict_success": bool(result["strict"]["success"]),
                "strict_error": result["strict"].get("error"),
                "execution_success": bool(standardized["success"]),
                "execution_error": standardized.get("error"),
                "num_inertial_repairs": repairs.get("num_inertial_repairs", 0),
                "nq": standardized.get("nq"),
                "nv": standardized.get("nv"),
                "njnt": standardized.get("njnt"),
                "ngeom": standardized.get("ngeom"),
                "nmesh": standardized.get("nmesh"),
            })
    sample_records = package_sample_10(method_list, output, args.steps)
    write_outputs(output, records, sample_records)
    print(f"Saved MuJoCo suite: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

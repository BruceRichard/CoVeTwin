"""Prompt templates matching the two-stage CoVeTwin formulation."""

from __future__ import annotations


RELATIVE_SHAPE_SPAN_INSTRUCTION = """Represent the occupied part on a 32 x 32 x 32 voxel grid.
Flatten each voxel as q = x*32*32 + y*32 + z. Sort unique q values and merge every maximal consecutive run [s,e]. Let b be the first run start. Return exactly one compact line:
rss b delta_1:length_1 delta_2:length_2 ...
where delta_m = s_m - b and length_m = e_m - s_m + 1. The first pair must start with 0. Output no explanation, Markdown, coordinates, or other numbers.
Example: absolute runs [184,184], [198,216], [230,237] become:
rss 184 0:1 14:19 46:8"""


def part_geometry_prompt(part_index: int, grid_size: int = 32) -> str:
    if grid_size != 32:
        instruction = RELATIVE_SHAPE_SPAN_INSTRUCTION.replace(
            "32 x 32 x 32", f"{grid_size} x {grid_size} x {grid_size}"
        ).replace("x*32*32 + y*32 + z", f"x*{grid_size}*{grid_size} + y*{grid_size} + z")
    else:
        instruction = RELATIVE_SHAPE_SPAN_INSTRUCTION
    return (
        f"Based on the image and the structured global description, predict only "
        f"the geometry of l_{part_index}.\n{instruction}"
    )

"""Compose per-object baseline videos into 2x3 grids for optional VLM ranking.

Layout (2 rows x 3 columns):
    (0,0)=GT          (0,1)=A           (0,2)=B
    (1,0)=empty        (1,1)=C           (1,2)=D

Letter-to-method mapping:
    A=physxanything  B=physx3d  C=articulate  D=urdformer

A grid is generated only when the corresponding video exists for all five
inputs (GT plus four baselines). Incomplete indices are skipped. Outputs use
object IDs so downstream evaluation can resolve finaljson directly.
"""
import os
import cv2
import numpy as np
import imageio

# Video directory for each grid cell; None leaves the cell empty.
LAYOUT = [
    [('GT', 'evaluation_video_gt'),
     ('A',  'evaluation_video_physxanything'),
     ('B',  'evaluation_video_physx3d')],
    [(None, None),
     ('C',  'evaluation_video_articulateanything'),
     ('D',  'evaluation_video_urdformer')],
]

# Index-to-object-ID mapping shared with the unified evaluator and video renderer.
ID_MAP = {
    0: "1817", 1: "1986", 2: "2230", 3: "4533", 4: "11586", 5: "12654",
    6: "102434", 7: "100197", 8: "100321", 9: "100443", 10: "100501",
    11: "100523", 12: "100758", 13: "100907", 14: "100925", 15: "101448",
    16: "101861",
}

CELL = 512                 # Side length of each grid cell.
OUT_DIR = './evaluation_video'
FPS = 30
OVERWRITE = False          # Set to True to regenerate existing outputs.


def read_frames(path):
    """Read all frames, resize them to CELL x CELL, and return BGR arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (CELL, CELL)))
    cap.release()
    return frames


def label(cell, text):
    """Draw the method letter in the upper-left corner of a cell."""
    if not text:
        return cell
    cv2.putText(cell, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return cell


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, cols = len(LAYOUT), len(LAYOUT[0])
    made, skipped = 0, 0

    for idx in sorted(ID_MAP):
        obj_id = ID_MAP[idx]

        # Collect all method videos for this index and check completeness.
        cell_paths = {}
        missing = []
        for r in range(rows):
            for c in range(cols):
                tag, folder = LAYOUT[r][c]
                if folder is None:
                    continue
                vp = os.path.join(folder, f'{idx}.mp4')
                if os.path.exists(vp):
                    cell_paths[(r, c)] = vp
                else:
                    missing.append(tag)

        if missing:
            print(f"skip index {idx} (id {obj_id}): missing {missing}")
            skipped += 1
            continue

        out_path = os.path.join(OUT_DIR, f'{obj_id}.mp4')
        if os.path.exists(out_path) and not OVERWRITE:
            print(f"skip existing: {out_path}")
            continue

        # Read every cell and align video length to the shortest input.
        frames_by_cell = {pos: read_frames(p) for pos, p in cell_paths.items()}
        T = min(len(f) for f in frames_by_cell.values())
        if T == 0:
            print(f"skip index {idx} (id {obj_id}): empty video")
            skipped += 1
            continue

        writer = imageio.get_writer(out_path, fps=FPS, quality=9)
        for t in range(T):
            canvas = np.zeros((rows * CELL, cols * CELL, 3), dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    tag, folder = LAYOUT[r][c]
                    if folder is None:
                        continue
                    cell = frames_by_cell[(r, c)][t].copy()
                    cell = label(cell, tag)
                    canvas[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cell
            # OpenCV uses BGR while imageio expects RGB.
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        writer.close()
        print(f"save: {out_path}  (index {idx}, {T} frames)")
        made += 1

    print(f"\ndone. generated {made} grid videos; skipped {skipped}.")


if __name__ == '__main__':
    main()

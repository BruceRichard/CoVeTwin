"""Merge sharded CoVeTwin conversation JSON files into one record list."""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="training_set_*_v2codec*.json")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_abs = os.path.abspath(args.output)
    paths = [
        path for path in sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if os.path.abspath(path) != output_abs
    ]
    if not paths:
        raise FileNotFoundError(f"No files match {os.path.join(args.input_dir, args.pattern)}")

    total = 0
    used_paths = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_output = args.output + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as out_fp:
        out_fp.write("[\n")
        first = True
        for path in paths:
            if os.path.getsize(path) == 0:
                print(f"skip empty file: {path}")
                continue
            used_paths.append(path)
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, list):
                raise ValueError(f"{path} is not a JSON list")
            for item in data:
                if not first:
                    out_fp.write(",\n")
                json.dump(item, out_fp, ensure_ascii=False)
                first = False
                total += 1
        out_fp.write("\n]\n")

    os.replace(tmp_output, args.output)
    print(f"merged {len(used_paths)} files, {total} samples -> {args.output}")


if __name__ == "__main__":
    main()

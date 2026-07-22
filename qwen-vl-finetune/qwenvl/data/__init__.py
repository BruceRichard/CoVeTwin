import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

PHYSXNET = {
    "annotation_path": os.environ.get("PHYSXNET_ANNOTATION_PATH", "PATH_TO_PHYSXNET_ANNOTATION"),
    "data_path": os.environ.get("PHYSXNET_IMAGE_ROOT", "PATH_TO_PHYSXNET_IMAGES"),
}

PHYSXMOBILITY = {
    "annotation_path": str(
        _REPO_ROOT
        / "dataset/im_data_obj_sort_new_32_finetune_final/training_set_all_mobility.json"
    ),
    "data_path": str(_REPO_ROOT / "dataset_toolkits/renders_all"),
}

PHYSXMOBILITY_V2 = {
    "annotation_path": str(
        _REPO_ROOT
        / "dataset/im_data_obj_sort_new_32_finetune_mobility_v2codec/training_set_all_mobility_v2codec.json"
    ),
    "data_path": str(_REPO_ROOT / "dataset_toolkits/renders_all"),
}

PHYSXMOBILITY_V2_COT = {
    "annotation_path": str(
        _REPO_ROOT
        / "dataset/im_data_obj_sort_new_32_finetune_mobility_v2codec_cot/training_set_all_mobility_v2codec_cot.json"
    ),
    "data_path": str(_REPO_ROOT / "dataset_toolkits/renders_all"),
}

# Environment variables keep the CoVeTwin entry portable.
COVETWIN = {
    "annotation_path": os.environ.get(
        "COVETWIN_ANNOTATION_PATH",
        str(_REPO_ROOT / "dataset" / "covetwin_training" / "conversations.json"),
    ),
    "data_path": os.environ.get(
        "COVETWIN_IMAGE_ROOT",
        str(_REPO_ROOT / "dataset_toolkits" / "renders_all"),
    ),
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,

    "physxnet":PHYSXNET,
    "physxmobility":PHYSXMOBILITY,
    "physxmobility_v2":PHYSXMOBILITY_V2,
    "physxmobility_v2_cot":PHYSXMOBILITY_V2_COT,
    "covetwin": COVETWIN,

}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["physxmobility"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)

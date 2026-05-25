"""Generic MVTec-style metadata generator."""

import argparse
import json
from pathlib import Path


DATASET_CLASSES = {
    "mpdd": ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"],
    "btad": ["01", "02", "03"],
}

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
NORMAL_DIRS = {"good", "ok"}


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


class MVTecStyleSolver:
    def __init__(self, root, dataset_name, class_names=None):
        self.root = Path(root)
        self.dataset_name = dataset_name.lower()
        self.class_names = class_names or DATASET_CLASSES.get(self.dataset_name)
        self.root = self._resolve_root(self.root)
        self.meta_path = self.root / "meta.json"

        if self.class_names is None:
            self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())

    def _resolve_root(self, root):
        if self.class_names is None:
            return root

        if any((root / cls_name).exists() for cls_name in self.class_names):
            return root

        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if any((child / cls_name).exists() for cls_name in self.class_names):
                return child

        return root

    def run(self):
        info = {"train": {}, "test": {}}
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue

            for phase in ["train", "test"]:
                phase_dir = cls_dir / phase
                cls_info = []

                if not phase_dir.exists():
                    info[phase][cls_name] = cls_info
                    continue

                species_dirs = sorted(path for path in phase_dir.iterdir() if path.is_dir())
                for species_dir in species_dirs:
                    specie_name = species_dir.name
                    is_abnormal = specie_name.lower() not in NORMAL_DIRS
                    img_paths = sorted(path for path in species_dir.iterdir() if is_image(path))

                    mask_paths = []
                    mask_dir = cls_dir / "ground_truth" / specie_name
                    if is_abnormal and mask_dir.exists():
                        mask_paths = sorted(path for path in mask_dir.iterdir() if is_image(path))

                    for idx, img_path in enumerate(img_paths):
                        mask_path = ""
                        if is_abnormal and idx < len(mask_paths):
                            mask_path = str(mask_paths[idx].relative_to(self.root))

                        cls_info.append(
                            {
                                "img_path": str(img_path.relative_to(self.root)),
                                "mask_path": mask_path,
                                "cls_name": cls_name,
                                "specie_name": specie_name,
                                "anomaly": 1 if is_abnormal else 0,
                            }
                        )

                        if phase == "test":
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1

                info[phase][cls_name] = cls_info

        with self.meta_path.open("w") as f:
            f.write(json.dumps(info, indent=4) + "\n")

        print("normal_samples", normal_samples, "anomaly_samples", anomaly_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate MVTec-style meta.json")
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CLASSES), help="dataset name")
    args = parser.parse_args()

    MVTecStyleSolver(root=args.root, dataset_name=args.dataset).run()

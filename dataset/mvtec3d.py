"""MVTec 3D-AD dataset solver for anomaly detection."""

import argparse
import json
import os
from pathlib import Path


class MVTec3DSolver:
    CLSNAMES = [
        "bagel", "cable_gland", "carrot", "cookie", "dowel",
        "foam", "peach", "potato", "rope", "tire",
    ]
    IMG_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

    def __init__(self, root="./dataset/MVTec-3D", class_names=None):
        self.root = Path(root)
        self.meta_path = self.root / "meta.json"
        if class_names is None:
            class_names = [
                name for name in self.CLSNAMES
                if (self.root / name).is_dir()
            ]
            if not class_names:
                class_names = sorted(
                    path.name for path in self.root.iterdir()
                    if path.is_dir() and path.name != "lost+found"
                )
        self.class_names = class_names

    def run(self):
        info = {"train": {}, "test": {}}
        normal_samples = 0
        anomaly_samples = 0

        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            for phase in ["train", "test"]:
                phase_dir = cls_dir / phase
                cls_info = []
                if not phase_dir.exists():
                    info[phase][cls_name] = cls_info
                    continue

                for specie_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
                    specie = specie_dir.name
                    is_abnormal = specie != "good"
                    image_dir = specie_dir / "rgb" if (specie_dir / "rgb").is_dir() else specie_dir
                    image_paths = self._list_images(image_dir)

                    mask_paths = []
                    if is_abnormal:
                        mask_dir = self._find_mask_dir(cls_dir, specie_dir, specie)
                        if mask_dir is None:
                            raise FileNotFoundError(
                                f"Mask directory not found for {cls_name}/{phase}/{specie}. "
                                "Expected one of gt, ground_truth, mask, masks, or class-level ground_truth/<defect>."
                            )
                        mask_paths = self._list_images(mask_dir)

                    for idx, image_path in enumerate(image_paths):
                        mask_path = ""
                        if is_abnormal:
                            mask_path = self._match_mask(image_path, mask_paths, idx)
                        cls_info.append(
                            {
                                "img_path": self._relative(image_path),
                                "mask_path": self._relative(mask_path) if mask_path else "",
                                "cls_name": cls_name,
                                "specie_name": specie,
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

    def _list_images(self, directory):
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMG_EXTENSIONS
        )

    def _find_mask_dir(self, cls_dir, specie_dir, specie):
        candidates = [
            specie_dir / "gt",
            specie_dir / "ground_truth",
            specie_dir / "mask",
            specie_dir / "masks",
            cls_dir / "ground_truth" / specie,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def _match_mask(self, image_path, mask_paths, idx):
        if not mask_paths:
            raise FileNotFoundError(f"No mask images found for {image_path}")
        by_stem = {path.stem: path for path in mask_paths}
        if image_path.stem in by_stem:
            return by_stem[image_path.stem]
        if idx < len(mask_paths):
            return mask_paths[idx]
        raise FileNotFoundError(f"No matching mask found for {image_path}")

    def _relative(self, path):
        return os.path.relpath(path, self.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate MVTec 3D-AD meta.json")
    parser.add_argument("--root", type=str, default="./dataset/MVTec-3D", help="path to MVTec 3D-AD dataset root")
    parser.add_argument("--class_name", type=str, default=None, help="optional single class, for example cable_gland")
    args = parser.parse_args()

    runner = MVTec3DSolver(root=args.root, class_names=[args.class_name] if args.class_name else None)
    runner.run()

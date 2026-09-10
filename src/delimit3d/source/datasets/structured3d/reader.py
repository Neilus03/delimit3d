from __future__ import annotations

import os
import zipfile

import cv2
import numpy as np


class Structured3DReader:
    """
    ZIP-backed reader for Structured3D assets.

    This intentionally mirrors the Pointcept preprocessing conventions:
    - reads files from one-or-many ZIPs without extracting to disk
    - provides directory-style listing helpers
    - decodes PNG/JPEG bytes via OpenCV
    """

    def __init__(self, files: str | list[str]):
        if isinstance(files, str):
            files = [files]
        self.readers = [zipfile.ZipFile(f, "r") for f in files]
        self.names_mapper: dict[str, int] = {}
        for idx, reader in enumerate(self.readers):
            for name in reader.namelist():
                self.names_mapper[name] = idx

    def filelist(self) -> list[str]:
        return list(self.names_mapper.keys())

    def listdir(self, dir_name: str) -> list[str]:
        dir_name = dir_name.lstrip(os.path.sep).rstrip(os.path.sep)
        file_list = list(
            np.unique(
                [
                    f.replace(dir_name + os.path.sep, "", 1).split(os.path.sep)[0]
                    for f in self.filelist()
                    if f.startswith(dir_name + os.path.sep)
                ]
            )
        )
        if "" in file_list:
            file_list.remove("")
        return file_list

    def read(self, file_name: str) -> bytes:
        split = self.names_mapper[file_name]
        return self.readers[split].read(file_name)

    def read_camera(self, camera_path: str):
        z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        cam_extr = np.fromstring(self.read(camera_path), dtype=np.float32, sep=" ")
        cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
        if cam_extr.shape[0] > 3:
            cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
            cam_n = np.cross(cam_front, cam_up)
            cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
            cam_r = np.matmul(z2y_top_m, cam_r)
            cam_f = cam_extr[9:11]
        else:
            cam_r = np.eye(3, dtype=np.float32)
            cam_f = None
        return cam_r, cam_t, cam_f

    def read_depth(self, depth_path: str) -> np.ndarray:
        depth_data = np.frombuffer(self.read(depth_path), np.uint8)
        depth = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        return depth  # 2D uint16 (mm)

    def read_color(self, color_path: str) -> np.ndarray:
        color_data = np.frombuffer(self.read(color_path), np.uint8)
        color = cv2.imdecode(color_data, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]  # BGR->RGB
        return color

    def read_instance(self, instance_path: str) -> np.ndarray:
        inst_data = np.frombuffer(self.read(instance_path), np.uint8)
        instance = cv2.imdecode(inst_data, cv2.IMREAD_UNCHANGED)
        return instance

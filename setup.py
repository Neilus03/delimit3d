from setuptools import find_packages, setup

setup(
    name="delimit3d",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": {
        "delimit3d-adapt": "delimit3d.training.runner:main",
        "delimit3d-init-scratch": "delimit3d.cli.initialize_scratch:main",
        "delimit3d-init-public": "delimit3d.cli.initialize_public:main",
        "delimit3d-select-checkpoint": "delimit3d.cli.select_checkpoint:main",
        "delimit3d-prepare-structured3d-raw": "delimit3d.cli.prepare_structured3d_raw:main",
        "delimit3d-prepare-structured3d-sources": "delimit3d.cli.prepare_structured3d_sources:main",
    }},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
        "PyYAML>=6.0",
        "Pillow>=9.0",
        "opencv-python>=4.8",
        "imageio>=2.20",
        "plyfile>=0.8",
    ],
    extras_require={"sources": ["open3d>=0.17"]},
)

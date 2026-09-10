from setuptools import find_packages, setup

setup(
    name="delimit3d",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": [
        "delimit3d-adapt=delimit3d.training.runner:main",
        "delimit3d-init-scratch=delimit3d.cli.initialize_scratch:main",
        "delimit3d-init-public=delimit3d.cli.initialize_public:main",
        "delimit3d-select-checkpoint=delimit3d.cli.select_checkpoint:main",
        "delimit3d-prepare-structured3d-raw=delimit3d.cli.prepare_structured3d_raw:main",
        "delimit3d-prepare-structured3d-sources=delimit3d.cli.prepare_structured3d_sources:main",
    ]},
)

import os

from setuptools import find_packages, setup


package_name = "arena_evaluation"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where=".", include=[f"{package_name}*"]),
    package_dir={"": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), [
            "config/data_recorder_config.yaml",
        ]),
    ],
    install_requires=["setuptools", "pyyaml", "pandas>=2,<3", "bddl>=3.5.0"],
    zip_safe=True,
    maintainer="Name",
    maintainer_email="your@email.com",
    description="Arena metric engine for VLN + dynamic social navigation.",
    license="MIT",
    package_data={
        package_name: [
            "bddl/definitions/*.bddl",
            "bddl/tasks/**/*.yaml",
            "bddl/tasks/**/*.bddl",
        ],
    },
    entry_points={
        "console_scripts": [
            "arena-eval = arena_evaluation.cli:main",
            "arena-episode-recorder = arena_evaluation.ros.episode_recorder_node:main",
            "arena-bddl-evaluator = arena_evaluation.ros.bddl_evaluator_node:main",
            "record = arena_evaluation.data_recorder_node:main",
            "metrics = arena_evaluation.get_metrics:main",
            "social_metrics = arena_evaluation.social_metrics:main",
            "vln_task_metrics = arena_evaluation.vln_task_metrics:main",
        ]
    },
)

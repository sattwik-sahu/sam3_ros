from glob import glob

from setuptools import find_packages, setup

package_name = "sam3_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="moonlab",
    maintainer_email="sattwik21@iiserb.ac.in",
    description=(
        "ROS 2 server and streaming nodes for SAM 3 text-prompted segmentation."
    ),
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "sam3_server = sam3_ros.sam3_server_node:main",
            "sam3_stream = sam3_ros.sam3_stream_node:main",
            "sam3_download_weights = sam3_ros.download_weights:main",
        ],
    },
)

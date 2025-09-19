#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="fluorosam",
    version="0.0.1",
    description="TODO.",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/flurosam",
    install_requires=["lightning", "pyyaml"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "segment = fluorosam.segment:main",
            "gui = fluorosam.segment_gui",
        ]
    },
)

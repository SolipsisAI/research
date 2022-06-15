from pathlib import Path

from setuptools import setup, find_packages

ROOT_DIR = Path().absolute()
REQUIREMENTS_DIR = ROOT_DIR.joinpath("requirements")


def read_requirements_from_dir(requirements_dir):
    requirement_files = list(map(open, Path(requirements_dir).glob("*.txt")))
    requirement_groups = list(
        map(
            lambda g: list(map(lambda i: i.strip(), g)),
            map(lambda f: f.readlines(), requirement_files),
        )
    )
    requirements = [
        item for requirement_group in requirement_groups for item in requirement_group
    ]
    return requirements


setup(
    name="src",
    version="0.1.0",
    packages=find_packages(include=["src.*"]),
    entry_points={
        "console_scripts": [
            "solipsis-trainer=src.main:main",
            "solipsis-chat=src.chat:main",
        ],
    },
    install_requires=read_requirements_from_dir(REQUIREMENTS_DIR),
)

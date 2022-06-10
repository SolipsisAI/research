from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(include=["src.*"]),
    entry_points={
        "console_scripts": ["solipsis-trainer=src.main:main"],
    },
)

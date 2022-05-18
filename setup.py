from setuptools import setup, find_packages


setup(
    name="src",
    version="0.1.0",
    packages=find_packages(include=["src.*"]),
    install_requires=[
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "trainer=src.run:main",
            "chat-test=src.chat:main",
        ],
    },
)

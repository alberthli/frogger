# flake8: noqa
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grasp_refinement",
    version="0.0.1",
    author="Albert H. Li",
    author_email="alberthli@caltech.edu",
    description="FRoGGeR: Fast Robust Grasp Generation via the Min-Weight Metric.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alberthli/grasp_refinement",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    license="MIT",
)

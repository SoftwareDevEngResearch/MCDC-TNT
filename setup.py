from setuptools import setup, find_packages

setup(
    name="mcdc-tnt",
    version="0.1",
    packages=find_packages(include=["mcdc-tnt", "mcdc-tnt.*"]),
    include_package_data=True
)

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'numba',
    'matplotlib',
    'pyyaml',
]

tests_require = [
    'pytest',
    'pytest-cov',
]


setup(
    name="mcdc_tnt",
    version="0.1",
    description='Testbed for accelerator schemes in Python',
    author='Jackson P. Morgan',
    author_email='morgjack@oregonstate.edu',
    packages=find_packages(include=["mcdc_tnt", "mcdc_tnt.*"]),
    include_package_data=True,
)

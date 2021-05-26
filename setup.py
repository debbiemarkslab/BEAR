from os import path
from setuptools import setup, find_packages
import sys

# This setup file, and the .gitignore file of this repository were made using
# scientific python cookiecutter (see https://nsls-ii.github.io/scientific-python-cookiecutter/)

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 7)
if sys.version_info < min_version:
    error = """
bear-model does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='bear-model',
    description="A package for making BEAR generative biological sequence models.",
    long_description=readme,
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'bear_model': [
            'models/data/shuffled_virus_kmers_lag_5.tsv',
            'models/',
            'models/train_bear_net.py',
            'models/train_bear_ref.py',
            'models/config_files/*.cfg',
            'tests/check_summarize.py',
            'tests/exdata'
        ]
    },
    install_requires=requirements,
    license="MIT",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    keywords=('biological-sequences genomes tensorflow machine-learning'),
)

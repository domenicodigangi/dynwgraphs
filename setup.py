#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'numpy>=1.19', 'pandas>=1.2', 'scipy>=1.6' 'torch>=1.8' ]

test_requirements = ['pytest>=3', ]

setup(
    author="Domenico Di Gangi",
    author_email='digangidomenico@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package to model dynamical weighted, possibly sparse, graphs a.k.a. known as complex networks",
    entry_points={
        'console_scripts': [
            'dynwgraphs=dynwgraphs.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dynwgraphs',
    name='dynwgraphs',
    packages=find_packages(include=['dynwgraphs', 'dynwgraphs.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/digangidomenico/dynwgraphs',
    version='0.1.0',
    zip_safe=False,
)

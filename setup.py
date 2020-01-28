import glob

from setuptools import setup, find_packages

setup(
    name='atlas',
    description='Neural-Backed Generators',
    author='Rohan Bavishi',
    author_email='rbavishi@berkeley.edu',
    version='1.0.0',
    packages=find_packages(include=['atlas*']),
    package_data={'': ['*.py']},
    data_files=[('config', glob.glob('config/*', recursive=True))],
    include_package_data=True,
    entry_points={'console_scripts': ['atlas=atlas.main:run']},
    zip_safe=False, install_requires=['astunparse', 'tensorflow==1.15.2', 'ray', 'numpy', 'tqdm',
                                      'pandas==0.23.4', 'scipy==1.1.0', 'pytest', 'cloudpickle']
)

import os
from glob import glob
from setuptools import setup

package_name = 'team10_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Einar Augestad',
    maintainer_email='einaraug@stud.ntnu.no',
    description='Launch files that start the full team 10 system',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
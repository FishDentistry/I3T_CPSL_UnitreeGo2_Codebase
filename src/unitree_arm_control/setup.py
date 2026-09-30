import os
from glob import glob

from setuptools import setup


package_name = 'unitree_arm_control'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description=(
        'Safe ROS 2 command and feedback wrapper for the Unitree D1 arm.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'd1_arm_controller = unitree_arm_control.controller:main',
            'd1_trajectory_test = unitree_arm_control.trajectory_test:main',
        ],
    },
)

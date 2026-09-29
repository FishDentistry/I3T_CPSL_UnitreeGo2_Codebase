from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'intel_realsense_functions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cpsl',
    maintainer_email='cfronk12@gmail.com',
    description='RealSense frame publishing and grounded object localization.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'getCameraFrames = intel_realsense_functions.getCameraFrames:main',
            'groundingDinoNode = '
            'intel_realsense_functions.groundingDinoNode:main',
        ],
    },
)

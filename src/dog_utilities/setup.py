from setuptools import setup
import os
from glob import glob

package_name = 'dog_utilities'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,'scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
        ('share/' + package_name + '/dae', glob('dae/*.dae')),
        ('share/' + package_name + '/meshes/d1', glob('meshes/d1/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unitree',
    maintainer_email='unitree@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'transformUpdater = dog_utilities.DogTransformUpdater:main',
            'lidarScanRelay = dog_utilities.LidarScanRelay:main',
            'cmdVelTranslator = dog_utilities.CmdVelTranslator:main'
        ],
    },
)

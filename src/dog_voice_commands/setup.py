import os
from glob import glob

from setuptools import setup

package_name = 'dog_voice_commands'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description=(
        'Offline, owner-verified voice commands for the Unitree Go2.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_command_node = dog_voice_commands.voice_command_node:main',
            'enroll_speaker = dog_voice_commands.tools:enroll',
            'verify_speaker = dog_voice_commands.tools:verify',
            'list_audio_devices = dog_voice_commands.tools:list_devices',
        ],
    },
)

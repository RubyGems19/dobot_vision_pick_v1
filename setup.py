from setuptools import find_packages, setup

package_name = 'dobot_vision_pick'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ruby',
    maintainer_email='ruby@todo.todo',
    description='Homography-based picking for Dobot CR5 with ROS2',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_pick_node = dobot_vision_pick.vision_pick_node:main',
        ],
    },
)

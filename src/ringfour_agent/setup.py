from setuptools import find_packages, setup

package_name = 'ringfour_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'ringfour-sdk==0.1.0.dev0', 'pyyaml>=6'],
    zip_safe=True,
    maintainer='Cristi Dragomir',
    maintainer_email='cristidragomir97@gmail.com',
    description='ringfour agent: robot-agnostic JSON-RPC bridge node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent = ringfour_agent.main:main',
            'ringfour-agent = ringfour_agent.main:main',
            'ringfour-onboard = ringfour_agent.onboard:main',
            # Records a robot: mesh volumes, collision matrix, reach map.
            # A ROS executable because resolving package:// and expanding
            # xacro conditionals is exactly what a ROS environment is for.
            'onboard = ringfour_agent.onboard:main',
        ],
    },
)

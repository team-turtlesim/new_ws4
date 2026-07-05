from setuptools import find_packages, setup

package_name = 'interpret'

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
    maintainer='topst',
    maintainer_email='sooyong.park@telechips.com',
    description='Lane interpretation node: LaneDetection perception -> temporal '
                'filtering (judgment) -> control-ready LaneInfo.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'interpret_node = interpret.interpret_node:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'mrs_crazyflies_project'

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
    maintainer='tanakrit',
    maintainer_email='tanakrit@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazyflies_node = mrs_crazyflies_project.crazyflies_node:main'
        ],
    },
)

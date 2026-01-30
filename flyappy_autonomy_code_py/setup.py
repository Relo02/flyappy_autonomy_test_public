from glob import glob
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil


package_name = 'flyappy_autonomy_code_py'


class InstallWithScriptsCopy(install):
    """Custom install to copy scripts to lib/<package_name>/ for ROS2."""
    def run(self):
        super().run()
        # Copy scripts from bin/ to lib/<package_name>/
        # self.install_scripts is typically: install/<package>/bin
        # We need: install/<package>/lib/<package_name>
        bin_dir = self.install_scripts
        # Get the base install directory (install/<package>/)
        base_dir = os.path.dirname(bin_dir)
        lib_dir = os.path.join(base_dir, 'lib', package_name)
        os.makedirs(lib_dir, exist_ok=True)

        if os.path.exists(bin_dir):
            for script in os.listdir(bin_dir):
                if script.startswith('flyappy_autonomy_code_node') or script.startswith('flyappy_dwa_controller') or script.startswith('gap_visualizer'):
                    src = os.path.join(bin_dir, script)
                    dst = os.path.join(lib_dir, script)
                    shutil.copy2(src, dst)
                    os.chmod(dst, 0o755)


setup(
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        # Install config files
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'rclpy', 'nav2_msgs', 'numpy', 'casadi', 'scikit-learn', 'matplotlib'],
    entry_points={
        'console_scripts': [
            'flyappy_autonomy_code_node = flyappy_autonomy_code.flyappy_autonomy_code_node:main',
            'flyappy_dwa_controller = flyappy_autonomy_code.flyappy_dwa_controller:main',
            'gap_visualizer = flyappy_autonomy_code.gap_visualizer:main',
            'flyappy_foa_controller = flyappy_autonomy_code.flyappy_foa:main',
        ],
    },
    cmdclass={
        'install': InstallWithScriptsCopy,
    },
)

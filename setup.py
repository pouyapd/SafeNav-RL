from setuptools import setup, find_packages

setup(
    name="safenav-rl",
    version="0.1.0",
    author="Pouya Bathaei Pourmand",
    description="Safety-Constrained RL for Assistive Robot Navigation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.26.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "ros2": ["rclpy"],
        "dev": ["pytest", "tensorboard"],
    },
)

from setuptools import setup, find_packages

setup(
    name="luxtac",
    version="0.1.0",
    description="Simulation of the LuxTac optical taxel pad and its sim-to-real alignment study.",
    packages=find_packages(include=["luxtac", "luxtac.*"]),
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "trimesh", "hydra-core>=1.2", "omegaconf", "pyyaml"],
)

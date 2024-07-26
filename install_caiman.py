import os
import subprocess
import sys

def run_command(command):
    """Run a system command and ensure it completes successfully."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}: {command}")
        sys.exit(result.returncode)

def main(env_name="element-miniscope-env"):
    conda_executable = 'conda'
    mamba_executable = 'mamba'

    # Step 1: Create the Conda Environment
    print(f"Creating conda environment: {env_name}")
    run_command(f"{conda_executable} create -n {env_name} -y")

    # Step 2: Activate the Environment
    print(f"Activating conda environment: {env_name}")
    run_command(f"{conda_executable} activate base")
    run_command(f"{conda_executable} activate C:/Users/kusha/miniconda3/envs/{env_name}")

    # Step 3: Install CaImAn and its dependencies
    print("Installing CaImAn and its dependencies")
    run_command(f"{conda_executable} install -c conda-forge mamba -y")
    run_command(f"{mamba_executable} install -c conda-forge python==3.10 -y")
    run_command(f"{mamba_executable} install -c conda-forge caiman -y")
    run_command("pip install keras==2.15.0")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install CaImAn and element-miniscope dependencies.")
    parser.add_argument('-e', '--env', type=str, default="element-miniscope-env", help="Name of the conda environment to create and use. Default is 'element-miniscope-env'.")
    args = parser.parse_args()

    main(args.env)

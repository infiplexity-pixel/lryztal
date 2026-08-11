"""
Author: Ansh Mathur
Github: https://github.com/infiplexity-pixel/lryztal
"""

import subprocess
import sys
import os
import shutil
import re
from pathlib import Path

def get_version_from_pyproject():
    """Extract version from pyproject.toml"""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
        # Look for version = "x.y.z" pattern
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return None

def _run_command(cmd, header, check_returncode=True):
    # Handle both string and list commands
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    print(f"{header} Running: {' '.join(cmd)}")
    
    proc = subprocess.Popen(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)

    # Read stdout line by line while the process is running
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(header, line.strip())
    
    # Get any remaining output and errors
    stdout, stderr = proc.communicate()
    if stdout:
        for line in stdout.splitlines():
            print(header, line)
    if stderr:
        for line in stderr.splitlines():
            print("ERROR:", line)
    
    if check_returncode and proc.returncode != 0:
        print(f"ERROR: Command failed with return code {proc.returncode}")
        sys.exit(proc.returncode)
    
    return proc.returncode

_run_command(["python", "version.py"], "[VERSION]")
# Parse arguments
if len(sys.argv) < 2 or sys.argv[1] == "--build":
    # No commit message provided, try to get version from pyproject.toml
    version = get_version_from_pyproject()
    if version:
        commit_message = f"version {version}"
        print(f"Auto-generated commit message: '{commit_message}'")
    else:
        raise SyntaxError("Must provide a commit message of progress or ensure pyproject.toml has a version field.")
else:
    commit_message = sys.argv[1]

# Check if --build is in arguments (after the commit message or as the only arg)
build_mode = "--build" in sys.argv

# Remove --build from sys.argv if present to avoid confusion
if build_mode and len(sys.argv) > 2:
    # If we have both commit message and --build, remove --build from consideration
    pass

# Change to script directory
os.chdir(Path(__file__).parent.absolute())

# Run version script

# Git operations
_run_command(["git", "add", "*"], "[GIT]")
_run_command(["git", "commit", "-am", commit_message], "[GIT]")
_run_command(["git", "push", "-u", "origin", "main"], "[GIT-PUSH]")

if build_mode:
    # Clean previous builds
    dist_dir = Path("dist")
    if dist_dir.exists():
        print(f"{'Cleaning dist directory':=^100}")
        shutil.rmtree(dist_dir)
    
    # Build the package
    print(f"{'Building package':=^100}")
    _run_command(["python", "-m", "build"], "[BUILD]")
    
    # Find the built wheel file
    wheel_files = list(dist_dir.glob("*.whl"))
    if wheel_files:
        # Get the most recently created wheel
        latest_wheel = max(wheel_files, key=lambda f: f.stat().st_ctime)
        print(f"{f'Installing wheel: {latest_wheel.name}':=^100}")
        _run_command(["pip", "install", "--force-reinstall", str(latest_wheel)], "[PIP]")
        
        # Optional: Clean up build artifacts
        # shutil.rmtree(dist_dir)
        # shutil.rmtree(Path("build"), ignore_errors=True)
        # shutil.rmtree(Path("__pycache__"), ignore_errors=True)
    else:
        print("ERROR: No wheel file found in dist/ directory")
        sys.exit(1)
else:
    # Development installation
    print(f"{'Installing in development mode':=^100}")
    _run_command(["pip", "install", "-e", ".[dev]"], "[PIP]")

_run_command(["pytest", "tests"], "[TESTS]")
print(f"{'Done!':=^100}")
"""
Author: Ansh Mathur
Github: https://github.com/infiplexity-pixel/lryztal
"""

"""
Version management script for Python projects.

Usage:
    python version.py                # Bump patch version (1.2.3 -> 1.2.4)
    python version.py 1.2.3          # Set version exactly
    python version.py rc             # Bump RC version (1.2.3 -> 1.2.3rc1)
    python version.py release        # Remove RC (1.2.3rc1 -> 1.2.3)
    python version.py feature        # Bump minor, commit, push (1.2.3 -> 1.3.0)
    python version.py major          # Bump major, commit, push (1.2.3 -> 2.0.0)
    python version.py current        # Show current version
    python version.py --dry-run <cmd> # Preview changes without modifying
"""

import re
import subprocess
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class Version:
    major: int
    minor: int
    patch: int
    rc: Optional[int] = None

    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version string into Version object."""
        # Match version like 1.2.3 or 1.2.3rc4
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$'
        match = re.match(pattern, version_str.strip())
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor, patch, rc = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            rc=int(rc) if rc is not None else None
        )

    def to_string(self) -> str:
        """Convert Version object to string."""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.rc is not None:
            return f"{base}rc{self.rc}"
        return base

    def bump_patch(self) -> 'Version':
        """Bump patch version (1.2.3 -> 1.2.4)."""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            rc=None
        )

    def bump_minor(self) -> 'Version':
        """Bump minor version (1.2.3 -> 1.3.0)."""
        return Version(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            rc=None
        )

    def bump_major(self) -> 'Version':
        """Bump major version (1.2.3 -> 2.0.0)."""
        return Version(
            major=self.major + 1,
            minor=0,
            patch=0,
            rc=None
        )

    def bump_rc(self) -> 'Version':
        """Bump RC version (1.2.3 -> 1.2.3rc1, 1.2.3rc1 -> 1.2.3rc2)."""
        if self.rc is None:
            return Version(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                rc=1
            )
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            rc=self.rc + 1
        )

    def remove_rc(self) -> 'Version':
        """Remove RC designation (1.2.3rc1 -> 1.2.3)."""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            rc=None
        )


class VersionManager:
    def __init__(self, package_name: str, pyproject_path: Path, init_path: Path):
        self.package_name = package_name
        self.pyproject_path = pyproject_path
        self.init_path = init_path
        self.current_version = self._read_current_version()

    def _read_current_version(self) -> Version:
        """Read current version from pyproject.toml."""
        content = self.pyproject_path.read_text()
        pattern = r'^version\s*=\s*["\']([^"\']+)["\']$'
        
        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                return Version.from_string(match.group(1))
        
        raise ValueError("Could not find version in pyproject.toml")

    def _update_files(self, new_version: Version) -> None:
        """Update version in both pyproject.toml and __init__.py."""
        version_str = new_version.to_string()
        
        # Update pyproject.toml
        content = self.pyproject_path.read_text()
        pattern = r'^(version\s*=\s*["\'])([^"\']+)(["\'])$'
        content = re.sub(pattern, f'\\g<1>{version_str}\\g<3>', content, flags=re.MULTILINE)
        self.pyproject_path.write_text(content)
        
        # Update __init__.py
        content = self.init_path.read_text()
        pattern = r'^(__version__\s*=\s*["\'])([^"\']+)(["\'])$'
        content = re.sub(pattern, f'\\g<1>{version_str}\\g<3>', content, flags=re.MULTILINE)
        self.init_path.write_text(content)

    def set_version(self, new_version: Version, dry_run: bool = False) -> Version:
        """Set version to exact value."""
        if dry_run:
            print(f"Would set version to: {new_version.to_string()}")
            return new_version
        
        self._update_files(new_version)
        return new_version

    def bump_patch(self, dry_run: bool = False) -> Version:
        """Bump patch version."""
        new_version = self.current_version.bump_patch()
        if dry_run:
            print(f"Current: {self.current_version.to_string()}")
            print(f"Next:    {new_version.to_string()}")
            return new_version
        
        self._update_files(new_version)
        return new_version

    def bump_rc(self, dry_run: bool = False) -> Version:
        """Bump RC version."""
        new_version = self.current_version.bump_rc()
        if dry_run:
            print(f"Current: {self.current_version.to_string()}")
            print(f"Next:    {new_version.to_string()}")
            return new_version
        
        self._update_files(new_version)
        return new_version

    def remove_rc(self, dry_run: bool = False) -> Version:
        """Remove RC designation."""
        if self.current_version.rc is None:
            print("Current version is not an RC release")
            return self.current_version
        
        new_version = self.current_version.remove_rc()
        if dry_run:
            print(f"Current: {self.current_version.to_string()}")
            print(f"Next:    {new_version.to_string()}")
            return new_version
        
        self._update_files(new_version)
        return new_version

    def _git_commit_and_push(self, version_str: str, dry_run: bool = False) -> None:
        """Commit and push changes to git."""
        commands = [
            ["git", "add", "."],
            ["git", "commit", "-m", f"Release {version_str}"],
            ["git", "push"],
        ]
        
        if dry_run:
            print(f"Would commit: Release {version_str}")
            print("Would push to origin")
            return
        
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {' '.join(cmd)}:")
                print(e.stderr)
                raise

    def bump_minor(self, dry_run: bool = False) -> Version:
        """Bump minor version, commit, and push."""
        if self.current_version.rc is not None:
            # First remove RC if present
            new_version = self.current_version.remove_rc()
        else:
            new_version = self.current_version.bump_minor()
        
        if dry_run:
            print(f"Current: {self.current_version.to_string()}")
            print(f"Next:    {new_version.to_string()}")
            self._git_commit_and_push(self.current_version.to_string(), dry_run=True)
            return new_version
        
        self._update_files(new_version)
        self._git_commit_and_push(new_version.to_string())
        return new_version

    def bump_major(self, dry_run: bool = False) -> Version:
        """Bump major version, commit, and push."""
        if self.current_version.rc is not None:
            # First remove RC if present
            new_version = self.current_version.remove_rc()
        else:
            new_version = self.current_version.bump_major()
        
        if dry_run:
            print(f"Current: {self.current_version.to_string()}")
            print(f"Next:    {new_version.to_string()}")
            self._git_commit_and_push(self.current_version.to_string(), dry_run=True)
            return new_version
        
        self._update_files(new_version)
        self._git_commit_and_push(new_version.to_string())
        return new_version

    def show_current(self) -> None:
        """Display current version."""
        print(self.current_version.to_string())


def find_package_info() -> Tuple[str, Path, Path]:
    """Find package name and paths to version files."""
    # Find pyproject.toml
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    # Read package name from pyproject.toml
    content = pyproject.read_text()
    name_pattern = r'^name\s*=\s*["\']([^"\']+)["\']$'
    package_name = None
    
    for line in content.split('\n'):
        match = re.match(name_pattern, line.strip())
        if match:
            package_name = match.group(1)
            break
    
    if not package_name:
        raise ValueError("Could not find package name in pyproject.toml")
    
    # Find __init__.py
    init_path = Path("src") / package_name / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"{init_path} not found")
    
    return package_name, pyproject, init_path


def main():
    parser = argparse.ArgumentParser(description="Version management tool")
    parser.add_argument("command", nargs="?", help="Command to execute")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying")
    args = parser.parse_args()
    
    # Find package info
    package_name, pyproject, init_path = find_package_info()
    manager = VersionManager(package_name, pyproject, init_path)
    
    # No command = bump patch
    if args.command is None:
        manager.bump_patch(dry_run=args.dry_run)
        return
    
    # Handle commands
    if args.command == "current":
        manager.show_current()
    elif args.command == "rc":
        manager.bump_rc(dry_run=args.dry_run)
    elif args.command == "release":
        manager.remove_rc(dry_run=args.dry_run)
    elif args.command == "feature":
        manager.bump_minor(dry_run=args.dry_run)
    elif args.command == "major":
        manager.bump_major(dry_run=args.dry_run)
    elif re.match(r'^\d+\.\d+\.\d+(?:rc\d+)?$', args.command):
        # Exact version string
        new_version = Version.from_string(args.command)
        manager.set_version(new_version, dry_run=args.dry_run)
    else:
        print(f"Unknown command: {args.command}")
        print("Available commands:")
        print("  (no args)     - Bump patch version")
        print("  current       - Show current version")
        print("  rc            - Bump RC version")
        print("  release       - Remove RC designation")
        print("  feature       - Bump minor version (commit & push)")
        print("  major         - Bump major version (commit & push)")
        print("  X.Y.Z         - Set exact version")
        sys.exit(1)


if __name__ == "__main__":
    main()
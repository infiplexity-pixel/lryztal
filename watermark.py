#!/usr/bin/env python3
"""
watermark.py - Adds author watermark to all Python files in a directory tree
Author: Ansh Mathur
Github: https://github.com/infiplexity-pixel/lryztal
"""

import os
import sys
import re
import tomllib
from pathlib import Path
from typing import Optional, Dict, Any

# Default watermark template (will be customized)
WATERMARK_TEMPLATE = '''"""
Author: {author}
Github: {github_url}
"""

'''


def find_pyproject_toml(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find pyproject.toml by searching upwards from the current directory.
    
    Args:
        start_dir: Directory to start searching from (default: current directory)
    
    Returns:
        Path to pyproject.toml or None if not found
    """
    if start_dir is None:
        start_dir = Path.cwd()
    
    # Search upwards
    current = start_dir.resolve()
    while current != current.parent:
        toml_path = current / 'pyproject.toml'
        if toml_path.exists():
            return toml_path
        current = current.parent
    
    return None


def get_github_url_from_pyproject(toml_path: Path) -> Optional[str]:
    """
    Extract GitHub URL from pyproject.toml.
    Checks project.urls.Homepage and project.urls.Source.
    
    Args:
        toml_path: Path to pyproject.toml file
    
    Returns:
        GitHub URL string or None if not found
    """
    try:
        with open(toml_path, 'rb') as f:
            data = tomllib.load(f)
        
        # Check project.urls section
        urls = data.get('project', {}).get('urls', {})
        
        # Check for Homepage or Source URLs
        for key in ['Homepage', 'Source', 'Repository', 'Bug Tracker']:
            url = urls.get(key, '')
            if url and 'github.com' in url.lower():
                # Clean up URL - remove trailing slashes
                return url.rstrip('/')
        
        # Check if there's a github field directly in project
        github = data.get('project', {}).get('github', '')
        if github and 'github.com' in github.lower():
            return github.rstrip('/')
        
        # Fallback: try to construct from package name
        name = data.get('project', {}).get('name', '')
        if name and 'github.com' in name.lower():
            return name.rstrip('/')
            
    except (tomllib.TOMLDecodeError, KeyError, IOError) as e:
        print(f"Warning: Could not parse pyproject.toml: {e}", file=sys.stderr)
    
    return None


def get_author_info(author_arg: Optional[str] = None) -> tuple[str, str]:
    """
    Get author name and GitHub URL.
    
    Args:
        author_arg: Author name from command line
    
    Returns:
        Tuple of (author_name, github_url)
    """
    # Get author name
    author = author_arg or "Ansh Mathur"  # Default fallback
    
    # Try to get GitHub URL from pyproject.toml
    toml_path = find_pyproject_toml()
    github_url = None
    
    if toml_path:
        github_url = get_github_url_from_pyproject(toml_path)
        if github_url:
            print(f"✓ Found GitHub URL in {toml_path}: {github_url}")
        else:
            print(f"⚠ No GitHub URL found in {toml_path}")
    
    # If no GitHub URL found, use the default or prompt
    if not github_url:
        github_url = "https://github.com/username/repo"  # Default placeholder
        print(f"⚠ Using default GitHub URL: {github_url}")
        print("  (Add 'urls.Homepage' or 'urls.Source' to pyproject.toml)")
    
    return author, github_url


def has_watermark(content: str, author: str, github_url: str) -> bool:
    """Check if the file already has the watermark."""
    # Create pattern that matches the watermark with any content between
    pattern = r'"""[^"]*Author:\s*' + re.escape(author) + r'[^"]*Github:\s*' + re.escape(github_url) + r'[^"]*"""\s*\n'
    return re.search(pattern, content, re.DOTALL) is not None


def add_watermark(file_path: Path, author: str, github_url: str, dry_run: bool = False) -> bool:
    """
    Add watermark to a single Python file.
    
    Args:
        file_path: Path object pointing to the Python file
        author: Author name
        github_url: GitHub URL
        dry_run: If True, only print what would be done
    
    Returns:
        bool: True if watermark was added, False if already present
    """
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate watermark
        watermark = WATERMARK_TEMPLATE.format(author=author, github_url=github_url)
        
        # Check if watermark already exists
        if has_watermark(content, author, github_url):
            return False
        
        # Check if file starts with shebang
        if content.startswith('#!'):
            lines = content.split('\n')
            shebang = lines[0] + '\n'
            rest = '\n'.join(lines[1:])
            new_content = shebang + watermark + rest
        else:
            new_content = watermark + content
        
        if dry_run:
            print(f"[DRY RUN] Would add watermark to: {file_path}")
            return True
        
        # Write the updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ Added watermark to: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}", file=sys.stderr)
        return False


def process_directory(
    directory: str = '.',
    recursive: bool = True,
    dry_run: bool = False,
    exclude_dirs: Optional[list] = None,
    author: str = "Ansh Mathur",
    github_url: str = "https://github.com/infiplexity-pixel/omninn"
) -> Dict[str, int]:
    """
    Process all Python files in a directory.
    
    Args:
        directory: Root directory to scan
        recursive: If True, scan subdirectories recursively
        dry_run: If True, only print what would be done
        exclude_dirs: List of directory names to exclude
        author: Author name
        github_url: GitHub URL
    
    Returns:
        dict: Statistics about processed files
    """
    if exclude_dirs is None:
        exclude_dirs = ['venv', 'env', '.venv', '__pycache__', '.git', 'node_modules', 'dist', 'build']
    
    directory = Path(directory)
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    # Pattern for Python files
    pattern = '**/*.py' if recursive else '*.py'
    
    # Process files
    for file_path in directory.glob(pattern):
        # Skip if in excluded directory
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        
        # Skip this script itself
        if file_path.name == 'watermark.py':
            continue
        
        if add_watermark(file_path, author, github_url, dry_run):
            stats['processed'] += 1
        else:
            stats['skipped'] += 1
    
    return stats


def remove_watermark(file_path: Path, author: str, github_url: str, dry_run: bool = False) -> bool:
    """Remove watermark from a Python file if it exists."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not has_watermark(content, author, github_url):
            return False
        
        # Remove watermark
        pattern = r'"""[^"]*Author:\s*' + re.escape(author) + r'[^"]*Github:\s*' + re.escape(github_url) + r'[^"]*"""\s*\n'
        new_content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        if dry_run:
            print(f"[DRY RUN] Would remove watermark from: {file_path}")
            return True
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✓ Removed watermark from: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Command line interface for watermark script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add or remove author watermark to Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python watermark.py                              # Use default author, get GitHub from pyproject.toml
  python watermark.py -a "John Doe"               # Specify author
  python watermark.py -r                          # Process recursively
  python watermark.py -d ./src                    # Process src directory
  python watermark.py --remove                    # Remove watermark
  python watermark.py --dry-run                   # Preview changes without modifying files
  python watermark.py -a "John Doe" -g https://github.com/johndoe/project  # Manual GitHub URL

The script will automatically look for pyproject.toml in the current directory
and extract GitHub URL from project.urls.Homepage or project.urls.Source.

Example pyproject.toml:
  [project.urls]
  Homepage = "https://github.com/username/project"
  Source = "https://github.com/username/project"
        '''
    )
    
    parser.add_argument(
        '-a', '--author',
        default=None,
        help='Author name (if not provided, uses default or prompts)'
    )
    
    parser.add_argument(
        '-g', '--github',
        dest='github_url',
        default=None,
        help='GitHub repository URL (overrides pyproject.toml if provided)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        default='.',
        help='Root directory to process (default: current directory)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process subdirectories recursively'
    )
    
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove watermark instead of adding it'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=['venv', 'env', '.venv', '__pycache__', '.git', 'node_modules', 'dist', 'build'],
        help='Directory names to exclude (default: venv env .venv __pycache__ .git node_modules dist build)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Get author and GitHub URL
    author, github_url = get_author_info(args.author)
    
    # Override GitHub URL if provided via command line
    if args.github_url:
        github_url = args.github_url
        print(f"✓ Using GitHub URL from command line: {github_url}")
    
    # Verify we have a valid GitHub URL
    if 'github.com' not in github_url.lower() and not args.github_url:
        print(f"\n⚠ Warning: GitHub URL doesn't appear to be a GitHub URL: {github_url}")
        response = input("Continue with this URL? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    print(f"\n{'='*50}")
    print(f"Author: {author}")
    print(f"GitHub: {github_url}")
    print(f"{'='*50}\n")
    
    if args.remove:
        # Remove watermark
        count = 0
        for file_path in Path(args.directory).glob('**/*.py' if args.recursive else '*.py'):
            if any(excluded in file_path.parts for excluded in args.exclude):
                continue
            if file_path.name == 'watermark.py':
                continue
            if remove_watermark(file_path, author, github_url, args.dry_run):
                count += 1
        
        print(f"\n✓ Removed watermark from {count} files")
        if args.dry_run:
            print("[DRY RUN] No files were actually modified.")
    else:
        # Add watermark
        stats = process_directory(
            args.directory, 
            args.recursive, 
            args.dry_run, 
            args.exclude,
            author,
            github_url
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"SUMMARY:")
        print(f"  Files processed: {stats['processed']}")
        print(f"  Files skipped (already have watermark): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")
        
        if args.dry_run:
            print("\n[DRY RUN] No files were actually modified.")
        
        if stats['processed'] == 0 and stats['skipped'] == 0:
            print("\nNo Python files found to process.")


if __name__ == "__main__":
    main()
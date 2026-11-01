"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Union


# ----------------------------------------------------------------------------------------------------------------------
def run_test_scripts(  # NOSONAR - Ignore cognitive complexity
    test_dir: Optional[Union[str, Path]] = None, timeout: int = 300
) -> bool:
    """
    Run all Python scripts in the test directory.

    Parameters
    ----------
    test_dir : Optional[Union[str, Path]]
        Path to the test directory containing test scripts to run. The module containing this function is excluded.
        Defaults to None, which uses the parent directory of the current module.
        Optional. Default: None.
    timeout : int
        Timeout in seconds for each test.
        Optional. Default: 300.

    Returns
    -------
    bool
        If True, all tests passed. False otherwise.

    """

    if test_dir is None:
        test_dir = Path(__file__).parent

    test_path = Path(test_dir)

    if not test_path.exists():
        print(f"Test directory '{test_dir}' not found.")
        return False

    test_files = list(test_path.glob("*.py"))

    current_script = Path(__file__).name
    test_files = [f for f in test_files if f.name != current_script]

    if not test_files:
        print(f"No Python files found in '{test_dir}'.")
        return True

    print(f"Running {len(test_files)} test scripts...\n")

    failed_tests = []
    passed_tests = []

    for test_file in sorted(test_files):
        print(f"Running {test_file.name}... ", end="", flush=True)

        try:
            result = subprocess.run(
                args=[sys.executable, str(test_file)], capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0:
                print("PASSED")
                passed_tests.append(test_file.name)
            else:
                print("FAILED")
                if result.stderr.strip():
                    print(f"  Error: {result.stderr.strip()}")
                failed_tests.append(test_file.name)

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed_tests.append(test_file.name)
        except Exception as e:
            print(f"ERROR: {e}")
            failed_tests.append(test_file.name)

    # Summary
    print(f"\n{'-' * 40}")
    print(f"Results: {len(passed_tests)} passed, {len(failed_tests)} failed.")

    if failed_tests:
        print(f"Failed: {', '.join(failed_tests)}")
        return False
    else:
        print("All tests passed.")
        return True


# ======================================================================================================================
if __name__ == "__main__":
    success = run_test_scripts()
    sys.exit(0 if success else 1)

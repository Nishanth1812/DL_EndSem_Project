"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import os
import sys


# ----------------------------------------------------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # noqa - Ignore expected type warning
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

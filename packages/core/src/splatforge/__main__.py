"""Ermöglicht ``python -m splatforge``."""

import sys

from .cli import main

sys.exit(main())

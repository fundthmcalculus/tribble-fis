"""Make the isolated ``fuzzytree`` package importable when pytest is invoked from
the repository root (e.g. ``pytest tribble-tree/tests``)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
DUST3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../'))
# workaround for sibling import
sys.path.insert(0, DUST3R_REPO_PATH)

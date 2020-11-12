import os
import shutil
import tempfile

from contextlib import contextmanager


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

NODE_POPULATION = 'fake_node'
EDGE_POPULATION = 'fake_edge'


@contextmanager
def setup_tempdir(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

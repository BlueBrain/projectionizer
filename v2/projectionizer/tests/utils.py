import shutil
import tempfile

from contextlib import contextmanager


@contextmanager
def setup_tempdir(prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

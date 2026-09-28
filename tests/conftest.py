import pytest

from sample_data import write_all


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    """Paths of all synthetic sample files, keyed by file name."""
    return write_all(tmp_path_factory.mktemp("samples"))

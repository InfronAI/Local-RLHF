import pytest

from local_rlhf.testing import make_tiny_model


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tiny-model"))

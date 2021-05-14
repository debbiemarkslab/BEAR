import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cmdopt", action="store", default=".",
        help='Path to folder with kmc scripts (kmc and kmc_dump).')


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")

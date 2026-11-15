import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onzeprime import data, optimizer  # noqa: E402


@pytest.fixture(scope="session")
def game_data():
    clubs, by_club = data.load()
    return clubs, by_club


@pytest.fixture(scope="session")
def pool(game_data):
    return optimizer.pool_for_mode(game_data[1], "grandes")

import pytest

from kycgraph.pipeline import analyse, engine_for
from kycgraph.synth import generate

SEED = 7


@pytest.fixture(scope="session")
def dataset():
    return generate(SEED)


@pytest.fixture(scope="session")
def engine():
    return engine_for(SEED)


@pytest.fixture(scope="session")
def analysis(dataset, engine):
    return analyse(dataset, engine.models, engine.champions)


@pytest.fixture(scope="session")
def assessments(analysis, engine):
    return engine.assess_all(analysis)

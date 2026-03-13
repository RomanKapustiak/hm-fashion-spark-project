"""
tests/conftest.py
=================
Shared pytest fixtures for the entire test suite.

A single session-scoped SparkSession is created once and reused by all
test modules. This prevents JVM restart errors when multiple test files
each try to start their own SparkSession.
"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """
    Session-scoped SparkSession shared across ALL test modules.
    Created once at the start of the test run, stopped at the end.
    """
    session = (
        SparkSession.builder
        .appName("hm-test-suite")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

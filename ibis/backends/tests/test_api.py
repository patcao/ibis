import pytest


def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


# 1. `current_database` returns '.', but isn't listed in list_databases()
# 2. list_databases() returns directories which don't make sense as HDF5
#    databases
@pytest.mark.never(["dask", "pandas"], reason="pass")
@pytest.mark.notimpl(["datafusion", "duckdb"])
def test_database_consistency(con):
    # every backend has a different set of databases, not testing the
    # exact names for now
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) >= 1
    assert all(isinstance(database, str) for database in databases)

    current_database = con.current_database
    assert isinstance(current_database, str)
    assert current_database in databases


def test_list_tables(con):
    tables = con.list_tables()
    assert isinstance(tables, list)
    # only table that is garanteed to be in all backends
    assert 'functional_alltypes' in tables
    assert all(isinstance(table, str) for table in tables)

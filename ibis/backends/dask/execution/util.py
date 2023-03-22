from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.backends.pandas.execution.util as pd_util
import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.dask.core import execute
from ibis.backends.pandas.client import ibis_dtype_to_pandas
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher
from ibis.expr import datatypes as dt
from ibis.expr import lineage as lin
from ibis.expr import types as ir
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext
from ibis.common import graph
from ibis.expr.operations.sortkeys import SortKey

DispatchRule = Tuple[Tuple[Union[Type, Tuple], ...], Callable]

TypeRegistrationDict = Dict[
    Union[Type[ops.Node], Tuple[Type[ops.Node], ...]], List[DispatchRule]
]


def register_types_to_dispatcher(
    dispatcher: TraceTwoLevelDispatcher, types: TypeRegistrationDict
):
    """
    Many dask operations utilize the functions defined in the pandas backend
    without modification. This function helps perform registrations in bulk
    """
    for ibis_op, registration_list in types.items():
        for types_to_register, fn in registration_list:
            dispatcher.register(ibis_op, *types_to_register)(fn)


def make_meta_series(
    dtype: np.dtype,
    name: Optional[str] = None,
    meta_index: Optional[pd.Index] = None,
):
    if isinstance(meta_index, pd.MultiIndex):
        index_names = meta_index.names
        series_index = pd.MultiIndex(
            levels=[[]] * len(index_names),
            codes=[[]] * len(index_names),
            names=index_names,
        )
    elif isinstance(meta_index, pd.Index):
        series_index = pd.Index([], name=meta_index.name)
    else:
        series_index = pd.Index([])

    return pd.Series(
        [],
        index=series_index,
        dtype=dtype,
        name=name,
    )


def make_selected_obj(gs: SeriesGroupBy) -> Union[dd.DataFrame, dd.Series]:
    """
    When you select a column from a `pandas.DataFrameGroupBy` the underlying
    `.obj` reflects that selection. This function emulates that behavior.
    """
    # TODO profile this for data shuffling
    # We specify drop=False in the case that we are grouping on the column
    # we are selecting
    if isinstance(gs.obj, dd.Series):
        return gs.obj
    else:
        return gs.obj.set_index(gs.index, drop=False)[
            gs._meta._selected_obj.name
        ]


def coerce_to_output(
    result: Any, expr: ir.Expr, index: Optional[pd.Index] = None
) -> Union[dd.Series, dd.DataFrame]:
    """Cast the result to either a Series of DataFrame, renaming as needed.

    Reimplementation of `coerce_to_output` in the pandas backend, but
    creates dask objects and adds special handling for dd.Scalars.

    Parameters
    ----------
    result: Any
        The result to cast
    expr: ibis.expr.types.Expr
        The expression associated with the result
    index: pd.Index
        Optional. If passed, scalar results will be broadcasted according
        to the index.

    Returns
    -------
    result: A `dd.Series` or `dd.DataFrame`

    Raises
    ------
    ValueError
        If unable to coerce result

    Examples
    --------
    Examples below use pandas objects for legibility, but functionality is the
    same on dask objects.

    >>> coerce_to_output(pd.Series(1), expr)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr, [1,2,3])
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1,2,3], expr)
    0    [1, 2, 3]
    Name: result, dtype: object
    """
    result_name = expr._safe_name
    dataframe_exprs = (
        ir.DestructColumn,
        ir.StructColumn,
        ir.DestructScalar,
        ir.StructScalar,
    )
    if isinstance(expr, dataframe_exprs):
        return _coerce_to_dataframe(
            result, expr.type().names, expr.type().types
        )
    elif isinstance(result, (pd.Series, dd.Series)):
        # Series from https://github.com/ibis-project/ibis/issues/2711
        return result.rename(result_name)
    elif isinstance(expr, ir.Scalar):
        if isinstance(result, dd.core.Scalar):
            # wrap the scalar in a series
            out_dtype = _pandas_dtype_from_dd_scalar(result)
            out_len = 1 if index is None else len(index)
            meta = make_meta_series(dtype=out_dtype, name=result_name)
            # Specify `divisions` so that the created Dask object has
            # known divisions (to be concatenatable with Dask objects
            # created using `dd.from_pandas`)
            series = dd.from_delayed(
                _wrap_dd_scalar(result, result_name, out_len),
                meta=meta,
                divisions=(0, out_len - 1),
            )

            return series
        else:
            return dd.from_pandas(
                pd_util.coerce_to_output(result, expr, index), npartitions=1
            )
    else:
        raise ValueError(f"Cannot coerce_to_output. Result: {result}")


@dask.delayed
def _wrap_dd_scalar(x, name=None, series_len=1):
    return pd.Series([x for _ in range(series_len)], name=name)


def _pandas_dtype_from_dd_scalar(x: dd.core.Scalar):
    try:
        return x.dtype
    except AttributeError:
        return pd.Series([x._meta]).dtype


def _coerce_to_dataframe(
    data: Any,
    column_names: List[str],
    types: List[dt.DataType],
) -> dd.DataFrame:
    """
    Clone of ibis.util.coerce_to_dataframe that deals well with dask types

    Coerce the following shapes to a DataFrame.

    The following shapes are allowed:
    (1) A list/tuple of Series -> each series is a column
    (2) A list/tuple of scalars -> each scalar is a column
    (3) A Dask Series of list/tuple -> each element inside becomes a column
    (4) dd.DataFrame -> the data is unchanged

    Examples
    --------
    Note: these examples demonstrate functionality with pandas objects in order
    to make them more legible, but this works the same with dask.

    >>> coerce_to_dataframe(pd.DataFrame({'a': [1, 2, 3]}), ['b'])
       b
    0  1
    1  2
    2  3
    >>> coerce_to_dataframe(pd.Series([[1, 2, 3]]), ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    >>> coerce_to_dataframe(pd.Series([range(3), range(3)]), ['a', 'b', 'c'])
       a  b  c
    0  0  1  2
    1  0  1  2
    >>> coerce_to_dataframe([pd.Series(x) for x in [1, 2, 3]], ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    >>>  coerce_to_dataframe([1, 2, 3], ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    """
    if isinstance(data, dd.DataFrame):
        result = data

    elif isinstance(data, dd.Series):
        # This takes a series where the values are iterables and converts each
        # value into its own row in a new dataframe.

        # NOTE - We add a detailed meta here so we do not drop the key index
        # downstream. This seems to be fixed in versions of dask > 2020.12.0
        dtypes = map(ibis_dtype_to_pandas, types)
        series = [
            data.apply(
                _select_item_in_iter,
                selection=i,
                meta=make_meta_series(
                    dtype, meta_index=data._meta_nonempty.index
                ),
            )
            for i, dtype in enumerate(dtypes)
        ]

        result = dd.concat(series, axis=1)

    elif isinstance(data, (tuple, list)):
        if len(data) == 0:
            result = dd.from_pandas(
                pd.DataFrame(columns=column_names), npartitions=1
            )
        elif isinstance(data[0], dd.Series):
            result = dd.concat(data, axis=1)
        else:
            result = dd.from_pandas(
                pd.concat([pd.Series([v]) for v in data], axis=1),
                npartitions=1,
            )
    else:
        raise ValueError(f"Cannot coerce to DataFrame: {data}")

    result.columns = column_names
    return result


def _select_item_in_iter(t, selection):
    return t[selection]


def safe_concat(dfs: List[Union[dd.Series, dd.DataFrame]]) -> dd.DataFrame:
    """
    Concat a list of `dd.Series` or `dd.DataFrame` objects into one DataFrame

    This will use `DataFrame.concat` if all pieces are the same length.
    Otherwise we will iterratively join.

    When axis=1 and divisions are unknown, Dask `DataFrame.concat` can only
    operate on objects with equal lengths, otherwise it will raise a
    ValueError in `concat_and_check`.

    See https://github.com/dask/dask/blob/2c2e837674895cafdb0612be81250ef2657d947e/dask/dataframe/multi.py#L907.

    Note - Repeatedly joining dataframes is likely to be quite slow, but this
    should be hit rarely in real usage. A situtation that triggeres this slow
    path is aggregations where aggregations return different numbers of rows
    (see `test_aggregation_group_by` for a specific example).
    TODO - performance.
    """  # noqa: E501
    if len(dfs) == 1:
        maybe_df = dfs[0]
        if isinstance(maybe_df, dd.Series):
            return maybe_df.to_frame()
        else:
            return maybe_df

    lengths = list(map(len, dfs))
    if len(set(lengths)) != 1:
        result = dfs[0].to_frame()

        for other in dfs[1:]:
            result = result.join(other.to_frame(), how="outer")
    else:
        result = dd.concat(dfs, axis=1)

    return result


def compute_sort_key(
    key: str | SortKey,
    data: dd.DataFrame,
    timecontext: Optional[TimeContext] = None,
    scope: Scope = None,
    **kwargs,
):
    """
    Note - we use this function instead of the pandas.execution.util so that we
    use the dask `execute` method

    This function borrows the logic in the pandas backend. ``by`` can be a
    string or an expression. If ``by.get_name()`` raises an exception, we must
    ``execute`` the expression and sort by the new derived column.
    """
    by = key.to_expr()
    name = ibis.util.guid()
    try:
        if isinstance(by, str):
            return name, data[by]
        return name, data[by.get_name()]
    except com.ExpressionError:
        if scope is None:
            scope = Scope()
        scope = scope.merge_scopes(
            Scope({t: data}, timecontext) for t in by.op().root_tables()
        )
        new_column = execute(by, scope=scope, **kwargs)
        new_column.name = name
        return name, new_column


def compute_sorted_frame(
    df: dd.DataFrame,
    order_by: list[str | SortKey],
    group_by: list[str | SortKey] = None,
    timecontext=None,
    **kwargs,
) -> dd.DataFrame:
    sort_keys = []
    ascending = []

    if group_by is None:
        group_by = []

    for value in group_by:
        sort_keys.append(value)
        ascending.append(True)

    for key in order_by:
        sort_keys.append(key)
        ascending.append(key.ascending)

    new_columns = {}
    computed_sort_keys = []
    for key in sort_keys:
        computed_sort_key, temporary_column = compute_sort_key(
            key, df, timecontext, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)
        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(
        computed_sort_keys, ascending=ascending, kind='mergesort'
    )
    # TODO: we'll eventually need to return this frame with the temporary
    # columns and drop them in the caller (maybe using post_execute?)
    ngrouping_keys = len(group_by)
    return (
        result,
        computed_sort_keys[:ngrouping_keys],
        computed_sort_keys[ngrouping_keys:],
    )


def assert_identical_grouping_keys(*args):
    indices = [arg.index for arg in args]
    # Depending on whether groupby was called like groupby("col") or
    # groupby(["cold"]) index will be a string or a list
    if isinstance(indices[0], list):
        indices = [tuple(index) for index in indices]
    grouping_keys = set(indices)
    if len(grouping_keys) != 1:
        raise AssertionError(
            f"Differing grouping keys passed: {grouping_keys}"
        )


def add_globally_consecutive_column(
    df: dd.DataFrame | dd.Series,
    col_name: str = '_ibis_index',
    set_as_index: bool = True,
) -> dd.DataFrame:
    """Add a column that is globally consecutive across the distributed data.

    By construction, this column is already sorted and can be used to partition
    the data.
    This column can act as if we had a global index across the distributed data.
    This index needs to be consecutive in the range of [0, len(df)), allows
    downstream operations to work properly.
    The default index of dask dataframes is to be consecutive within each partition.

    Important properties:
    - Each row has a unique id (i.e. a value in this column)
    - The global index that's added is consecutive in the same order that the rows currently are in.
    - IDs within each partition are already sorted

    We also do not explicity deal with overflow in the bounds.

    Parameters
    ----------
    df : dd.DataFrame
        Dataframe to add the column to
    col_name: str
        Name of the column to use. Default is _ibis_index
    set_as_index: bool
        If True, will set the consecutive column as the index. Default is True.

    Returns
    -------
    dd.DataFrame
        New dask dataframe with sorted partitioned index
    """
    if isinstance(df, dd.Series):
        df = df.to_frame()

    if col_name in df.columns:
        raise ValueError(f"Column {col_name} is already present in DataFrame")

    df = df.assign(**{col_name: 1})
    df = df.assign(**{col_name: df[col_name].cumsum() - 1})
    if set_as_index:
        df = df.reset_index(drop=True)
        df = df.set_index(col_name, sorted=True)
    return df


def is_row_order_preserving(exprs) -> bool:
    """Detects if the operation preserves row ordering.

    Certain operations we know will not affect the ordering of rows in the
    dataframe (for example elementwise operations on ungrouped dataframes).
    In these cases we may be able to avoid expensive joins and assign directly
    into the parent dataframe.
    """

    def _is_row_order_preserving(expr: ir.Expr):
        if isinstance(expr.op(), (ops.Reduction, ops.Window)):
            return (lin.halt, False)
        else:
            return (lin.proceed, True)

    return lin.traverse(_is_row_order_preserving, exprs)


def rename_index(df: dd.DataFrame, new_index_name: str) -> dd.DataFrame:
    # No elegant way to rename index
    # https://github.com/dask/dask/issues/4950
    df = df.map_partitions(pd.DataFrame.rename_axis, new_index_name, axis='index')
    return df

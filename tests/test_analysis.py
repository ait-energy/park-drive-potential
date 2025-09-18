import pandas as pd

from intro_matsim.analysis import (
    _collect_links_between_nodes,
    _link_map,
)


def test_collect_links_between_nodes_trivial():
    df = pd.DataFrame({"fromNode": ["A"], "toNode": ["B"]})
    result = _collect_links_between_nodes(df)
    assert len(result) == 1
    assert result["A", "B"] == {0}


def test_collect_links_between_nodes_trivial_reversed():
    df = pd.DataFrame({"fromNode": ["B"], "toNode": ["A"]})
    result = _collect_links_between_nodes(df)
    assert len(result) == 1
    assert result["A", "B"] == {0}


def test_collect_links_between_nodes_trivial_custom_index():
    df = pd.DataFrame({"fromNode": ["A"], "toNode": ["B"]}, index=["row1"])
    result = _collect_links_between_nodes(df)
    assert len(result) == 1
    assert result["A", "B"] == {"row1"}


def test_collect_links_between_nodes_simple():
    df = pd.DataFrame({"fromNode": ["A", "B", "C"], "toNode": ["X", "Y", "Z"]})
    result = _collect_links_between_nodes(df)
    assert len(result) == 3
    assert result["A", "X"] == {0}
    assert result["B", "Y"] == {1}
    assert result["C", "Z"] == {2}


def test_collect_links_between_nodes_to_and_from_link():
    df = pd.DataFrame({"fromNode": ["A", "B"], "toNode": ["B", "A"]})
    result = _collect_links_between_nodes(df)
    assert len(result) == 1
    assert result["A", "B"] == {0, 1}


def test_collect_links_between_nodes_double_to_link():
    df = pd.DataFrame(
        {"fromNode": ["A", "A", "C"], "toNode": ["X", "X", "Z"]},
        index=["one", "one_dup", "two"],
    )
    result = _collect_links_between_nodes(df)
    assert len(result) == 2
    assert result["A", "X"] == {"one", "one_dup"}
    assert result["C", "Z"] == {"two"}


def test_collect_links_between_nodes_complex():
    df = pd.DataFrame(
        {
            "fromNode": ["A", "A", "B", "B", "D", "E", "F"],
            "toNode": ["B", "B", "A", "A", "E", "D", "G"],
        }
    )

    result = _collect_links_between_nodes(df)
    assert len(result) == 3
    assert result["A", "B"] == {0, 1, 2, 3}
    assert result["D", "E"] == {4, 5}
    assert result["F", "G"] == {6}


def test_link_map():
    df = pd.DataFrame(
        {
            "fromNode": ["A", "A", "B", "B", "D", "E", "F"],
            "toNode": ["B", "B", "A", "A", "E", "D", "G"],
        },
        index=["l2", "l1", "l3", "l4", "l5", "l6", "l7"],
    )

    # l1 must be the reference, even if it was not the first one (sorted!)
    result = _link_map(_collect_links_between_nodes(df).values())
    assert isinstance(result, dict)
    assert len(result) == 7
    assert result["l1"] == "l1"
    assert result["l2"] == "l1"
    assert result["l3"] == "l1"
    assert result["l4"] == "l1"
    assert result["l5"] == "l5"
    assert result["l6"] == "l5"
    assert result["l7"] == "l7"

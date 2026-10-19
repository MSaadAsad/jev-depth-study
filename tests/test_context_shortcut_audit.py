import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('context_report', Path(__file__).parents[1] / 'scripts/report_context_factorial.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_endpoint_role_answers_without_traversing_the_path():
    example = {'x': 'A', 'y': 'C', 'premises': [
        {'greater': 'A', 'lesser': 'B'}, {'greater': 'B', 'lesser': 'C'}]}
    assert module.endpoint_prediction(example) is True
    assert module.endpoint_prediction({**example, 'x': 'C', 'y': 'A'}) is False


def test_adding_off_path_edges_can_remove_the_endpoint_cue():
    example = {'x': 'A', 'y': 'C', 'premises': [
        {'greater': 'A', 'lesser': 'B'}, {'greater': 'B', 'lesser': 'C'},
        {'greater': 'X', 'lesser': 'A'}, {'greater': 'C', 'lesser': 'Y'}]}
    assert module.endpoint_prediction(example) is None

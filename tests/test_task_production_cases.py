"""Production task-case geometry and evidence contracts."""

from data.task_production_cases import (
    DAGShape,
    ProductionScenario,
    TaskConfusability,
    dag_shape_case,
    join_capacity_case,
    production_task_cases,
)
from pra_hf.task_context import TaskGraph, TaskProvenance


def test_main_cases_cover_confusability_and_end_to_end_families() -> None:
    cases = production_task_cases()

    assert len(cases) == 15
    assert {case.confusability for case in cases} == set(TaskConfusability)
    assert {case.scenario for case in cases} == set(ProductionScenario)
    assert len({case.seed for case in cases}) == 5
    for case in cases:
        assert case.required_record_ids
        assert all(TaskProvenance.from_record(record) for record in case.records)
        assert set(case.required_record_ids) <= {record.record_id for record in case.records}


def test_join_capacity_requires_every_predecessor() -> None:
    case = join_capacity_case(8, seed=11)
    graph = TaskGraph(case.graph)

    assert case.join_fan_in == 8
    assert len(case.required_record_ids) == 8
    assert set(graph.structural_closure(case.active_task_id)) == {
        case.active_task_id,
        *(f"p{index}" for index in range(8)),
    }


def test_dag_shapes_have_distinct_structural_geometry() -> None:
    cases = [dag_shape_case(shape, seed=11) for shape in DAGShape]
    closures = {
        case.dag_shape: TaskGraph(case.graph).structural_closure(case.active_task_id)
        for case in cases
    }

    assert set(closures) == {shape.value for shape in DAGShape}
    assert len({tuple(value) for value in closures.values()}) >= 3

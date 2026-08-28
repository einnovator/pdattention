"""Naturalistic typed workflows for the Paper 8 production-PRA iteration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.task_context import (
    TaskDescriptor,
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskProvenance,
    attach_task_provenance,
)


class TaskConfusability(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProductionScenario(str, Enum):
    INDEPENDENT = "independent_parallel"
    LINEAR = "linear_dependency"
    JOIN = "join"
    RESUMPTION = "resumption"
    CONFLICT = "conflicting_state"


class DAGShape(str, Enum):
    DEEP_NARROW = "deep_narrow"
    SHALLOW_WIDE = "shallow_wide"
    MULTI_FORK = "multi_fork"
    MULTI_JOIN = "multi_join"
    BALANCED = "balanced"


@dataclass(frozen=True)
class EvidenceTextAnnotation:
    """Tokenizer-independent anchors for one required record's source interval."""

    record_id: str
    answer: str
    semantic_anchors: tuple[str, ...] = ("Acme Atlas", "verification")


@dataclass(frozen=True)
class ProductionTaskCase:
    """One oracle-graph task continuation with exact evidence identities."""

    case_id: str
    seed: int
    scenario: ProductionScenario | str
    confusability: TaskConfusability | str
    graph: TaskDescriptor
    active_task_id: str
    records: tuple[ContextRecord, ...]
    query: str
    expected_answer: str
    required_record_ids: tuple[str, ...]
    distractor_answers: tuple[str, ...]
    join_fan_in: int = 0
    dag_shape: str = ""
    evidence_annotations: tuple[EvidenceTextAnnotation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", ProductionScenario(self.scenario))
        object.__setattr__(self, "confusability", TaskConfusability(self.confusability))


_SCENARIOS = (
    ProductionScenario.INDEPENDENT,
    ProductionScenario.LINEAR,
    ProductionScenario.JOIN,
    ProductionScenario.RESUMPTION,
    ProductionScenario.CONFLICT,
)

_ANSWER_WORDS = (
    "amber", "birch", "cobalt", "delta", "ember", "forest", "granite", "harbor",
    "indigo", "juniper", "kernel", "linen", "maple", "nectar", "onyx", "pearl",
    "quartz", "river", "saffron", "timber", "umber", "violet", "willow", "xenon",
    "yellow", "zenith", "acorn", "beacon", "canyon", "drift", "elm", "frost",
    "grove", "heather", "iris", "jade", "kelp", "lilac", "moss", "north",
)


def _answer(seed: int, confusability: TaskConfusability, ordinal: int) -> str:
    conf_offset = list(TaskConfusability).index(confusability) * 7
    return _ANSWER_WORDS[(seed + conf_offset + ordinal) % len(_ANSWER_WORDS)]


def _task_description(
    task_id: str,
    ordinal: int,
    confusability: TaskConfusability,
) -> str:
    if confusability == TaskConfusability.LOW:
        domains = ("billing ledger", "release pipeline", "legal filing", "service incident", "style review")
        return f"Resolve {domains[ordinal % len(domains)]} for scope {task_id}"
    if confusability == TaskConfusability.MEDIUM:
        goals = ("billing", "access", "deployment", "compliance", "support")
        return f"Resolve Acme Atlas {goals[ordinal % len(goals)]} state for scope {task_id}"
    return f"Resolve Acme Atlas release status and verification code for scope {task_id}"


def _payload(
    record_type: RecordType,
    *,
    task_id: str,
    description: str,
    answer: str,
    ordinal: int,
    required: bool,
) -> object:
    status = "verified" if required else "superseded"
    common = (
        f"Task {task_id}. {description}. Acme Atlas release status {status}. "
        f"The authoritative verification code is {answer}."
    )
    if record_type == RecordType.DB_RESULT:
        return {
            "columns": ["task", "project", "status", "verification_code"],
            "rows": [{
                "task": task_id,
                "project": "Acme Atlas",
                "status": status,
                "verification_code": answer,
            }],
        }
    if record_type == RecordType.TOOL_RESPONSE:
        return {
            "tool": "deployment_status",
            "task": task_id,
            "result": common,
            "verified": required,
        }
    if record_type == RecordType.TERMINAL_OUTPUT:
        return f"$ verify --task {task_id}\n{common}\nexit_code=0"
    if record_type == RecordType.FILE_READ:
        return f"# Task result {ordinal}\n\n{common}\n"
    if record_type == RecordType.TASK_STATE:
        return {
            "task_id": task_id,
            "status": "active" if required else "completed",
            "description": description,
            "result_ref": answer,
        }
    return common


def _record(
    case_id: str,
    task_id: str,
    ordinal: int,
    sequence: int,
    record_type: RecordType,
    description: str,
    answer: str,
    *,
    required: bool,
) -> ContextRecord:
    record = ContextRecord(
        f"{case_id}:record:{task_id}:{ordinal}",
        record_type,
        _payload(
            record_type,
            task_id=task_id,
            description=description,
            answer=answer,
            ordinal=ordinal,
            required=required,
        ),
        selection_provenance={"semantic_role": "task_result" if required else "tool_observation"},
    )
    return attach_task_provenance(
        record,
        TaskProvenance(task_id, event_sequence=sequence),
    )


def _build_graph(
    task_ids: Iterable[str],
    dependencies: dict[str, tuple[str, ...]],
    active_task_id: str,
    confusability: TaskConfusability,
) -> TaskDescriptor:
    graph = TaskGraph()
    sequence = 0
    for ordinal, task_id in enumerate(task_ids):
        sequence += 1
        graph.apply(TaskEvent(
            f"create:{task_id}",
            sequence,
            TaskEventType.CREATE,
            task_id,
            payload={
                "description": _task_description(task_id, ordinal, confusability),
                "depends_on": dependencies.get(task_id, ()),
            },
        ))
    sequence += 1
    graph.apply(TaskEvent(
        f"activate:{active_task_id}",
        sequence,
        TaskEventType.ACTIVATE,
        active_task_id,
        expected_version=1,
    ))
    return graph.snapshot()


def production_task_case(
    scenario: ProductionScenario | str,
    confusability: TaskConfusability | str,
    *,
    seed: int,
) -> ProductionTaskCase:
    """Create one bounded end-to-end case with task-scoped exact answers."""

    scenario = ProductionScenario(scenario)
    confusability = TaskConfusability(confusability)
    case_id = f"{scenario.value}-{confusability.value}-s{seed}"
    task_ids = tuple(f"t{index}" for index in range(8))
    dependencies: dict[str, tuple[str, ...]] = {task_id: () for task_id in task_ids}
    active = "t7"
    required_tasks = (active,)
    if scenario == ProductionScenario.LINEAR:
        dependencies["t7"] = ("t6",)
        required_tasks = ("t6", "t7")
    elif scenario == ProductionScenario.JOIN:
        dependencies["t7"] = ("t4", "t5", "t6")
        required_tasks = ("t4", "t5", "t6")
    elif scenario == ProductionScenario.RESUMPTION:
        active = "t0"
        required_tasks = (active,)
    elif scenario == ProductionScenario.CONFLICT:
        active = "t6"
        required_tasks = (active,)

    graph = _build_graph(task_ids, dependencies, active, confusability)
    types = (
        RecordType.FILE_READ,
        RecordType.DB_RESULT,
        RecordType.TERMINAL_OUTPUT,
        RecordType.TOOL_RESPONSE,
    )
    records = []
    required_ids = []
    answers = []
    distractors = []
    evidence_annotations = []
    for ordinal, task_id in enumerate(task_ids):
        answer = _answer(seed, confusability, ordinal)
        required = task_id in required_tasks
        sequence = ordinal + 1
        if scenario == ProductionScenario.RESUMPTION and task_id != active:
            sequence += 100
        if scenario == ProductionScenario.CONFLICT:
            description = "Resolve Acme Atlas account status for the current customer case"
        else:
            description = _task_description(task_id, ordinal, confusability)
        record = _record(
            case_id,
            task_id,
            ordinal,
            sequence,
            types[ordinal % len(types)],
            description,
            answer,
            required=required,
        )
        records.append(record)
        if required:
            required_ids.append(record.record_id)
            answers.append(answer)
            evidence_annotations.append(EvidenceTextAnnotation(record.record_id, answer))
        else:
            distractors.append(answer)

    expected = "+".join(answers)
    if len(answers) == 1:
        request = "Return its authoritative verification code"
    else:
        request = "Return every required code in dependency order, joined by +"
    active_description = (
        "Resolve Acme Atlas account status for the current customer case"
        if scenario == ProductionScenario.CONFLICT
        else _task_description(active, int(active.removeprefix("t")), confusability)
    )
    query = (
        f"Continue the active task: {active_description}. {request} as ANSWER=<value>. "
        "Ignore records belonging only to other tasks."
    )
    return ProductionTaskCase(
        case_id,
        seed,
        scenario,
        confusability,
        graph,
        active,
        tuple(records),
        query,
        expected,
        tuple(required_ids),
        tuple(distractors),
        join_fan_in=(3 if scenario == ProductionScenario.JOIN else 0),
        evidence_annotations=tuple(evidence_annotations),
    )


def production_task_cases(
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
) -> tuple[ProductionTaskCase, ...]:
    """Cover all end-to-end families once per confusability level."""

    rows = []
    for confusability in TaskConfusability:
        for index, seed in enumerate(seeds):
            rows.append(production_task_case(
                _SCENARIOS[index % len(_SCENARIOS)],
                confusability,
                seed=seed,
            ))
    return tuple(rows)


def join_capacity_case(
    fan_in: int,
    *,
    seed: int,
    confusability: TaskConfusability | str = TaskConfusability.HIGH,
) -> ProductionTaskCase:
    """Build a join whose exact predecessor set can exceed the routing budget."""

    if fan_in < 2:
        raise ValueError("Join fan-in must be at least two.")
    confusability = TaskConfusability(confusability)
    case_id = f"join-capacity-{fan_in}-s{seed}"
    predecessors = tuple(f"p{index}" for index in range(fan_in))
    active = "join"
    task_ids = (*predecessors, active)
    dependencies = {task_id: () for task_id in predecessors}
    dependencies[active] = predecessors
    graph = _build_graph(task_ids, dependencies, active, confusability)
    records = []
    answers = []
    for ordinal, task_id in enumerate(predecessors):
        answer = _ANSWER_WORDS[(seed + ordinal) % len(_ANSWER_WORDS)]
        answers.append(answer)
        records.append(_record(
            case_id,
            task_id,
            ordinal,
            ordinal + 1,
            (RecordType.DB_RESULT, RecordType.FILE_READ, RecordType.TOOL_RESPONSE)[ordinal % 3],
            _task_description(task_id, ordinal, confusability),
            answer,
            required=True,
        ))
    records.append(_record(
        case_id,
        active,
        fan_in,
        fan_in + 1,
        RecordType.TASK_STATE,
        "Combine all predecessor verification codes",
        "JOIN-PENDING",
        required=False,
    ))
    expected = "+".join(answers)
    return ProductionTaskCase(
        case_id,
        seed,
        ProductionScenario.JOIN,
        confusability,
        graph,
        active,
        tuple(records),
        "For task join, return every predecessor verification code in numeric task order, "
        "joined by +, as ANSWER=<value>.",
        expected,
        tuple(record.record_id for record in records[:-1]),
        ("JOIN-PENDING",),
        join_fan_in=fan_in,
    )


def dag_shape_case(shape: DAGShape | str, *, seed: int) -> ProductionTaskCase:
    """Generate independent DAG geometries with one active terminal task."""

    shape = DAGShape(shape)
    count = 8
    task_ids = tuple(f"d{index}" for index in range(count))
    dependencies = {task_id: () for task_id in task_ids}
    if shape == DAGShape.DEEP_NARROW:
        for index in range(1, count):
            dependencies[task_ids[index]] = (task_ids[index - 1],)
    elif shape == DAGShape.SHALLOW_WIDE:
        dependencies[task_ids[-1]] = task_ids[:-1]
    elif shape == DAGShape.MULTI_FORK:
        for index in range(1, 4):
            dependencies[task_ids[index]] = (task_ids[0],)
        for index in range(4, count):
            dependencies[task_ids[index]] = (task_ids[1 + (index % 3)],)
    elif shape == DAGShape.MULTI_JOIN:
        dependencies[task_ids[3]] = (task_ids[0], task_ids[1], task_ids[2])
        dependencies[task_ids[6]] = (task_ids[3], task_ids[4], task_ids[5])
        dependencies[task_ids[7]] = (task_ids[6],)
    else:
        dependencies.update({
            "d2": ("d0",), "d3": ("d0",), "d4": ("d1",), "d5": ("d1",),
            "d6": ("d2", "d3"), "d7": ("d4", "d5", "d6"),
        })
    active = task_ids[-1]
    graph = _build_graph(task_ids, dependencies, active, TaskConfusability.HIGH)
    closure = set(TaskGraph(graph).structural_closure(active)) - {active}
    records = []
    required_ids = []
    answers = []
    for ordinal, task_id in enumerate(task_ids):
        answer = _ANSWER_WORDS[(seed + 13 + ordinal) % len(_ANSWER_WORDS)]
        required = task_id in closure
        record = _record(
            f"dag-{shape.value}-s{seed}",
            task_id,
            ordinal,
            ordinal + 1,
            (RecordType.TOOL_RESPONSE, RecordType.TERMINAL_OUTPUT)[ordinal % 2],
            _task_description(task_id, ordinal, TaskConfusability.HIGH),
            answer,
            required=required,
        )
        records.append(record)
        if required:
            required_ids.append(record.record_id)
            answers.append(answer)
    expected = "+".join(answers)
    return ProductionTaskCase(
        f"dag-{shape.value}-s{seed}",
        seed,
        ProductionScenario.JOIN,
        TaskConfusability.HIGH,
        graph,
        active,
        tuple(records),
        f"Continue {active}; return all structural predecessor verification codes in "
        "numeric task order, joined by +, as ANSWER=<value>.",
        expected,
        tuple(required_ids),
        (),
        join_fan_in=len(required_ids),
        dag_shape=shape.value,
    )

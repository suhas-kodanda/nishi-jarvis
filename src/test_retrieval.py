from memory.service import MemoryService
from memory.models import MemoryLayer, MemoryKind

from retrieval import (
    calculate_p2_score,
    rerank_candidates,
    select_memories,
    get_selected_memory_ids,
    retrieve_and_select,
    retrieve_final_context,
)


OWNER_ID = "test_user"


def create_test_memories(service):

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="learning_style",
        layer=MemoryLayer.L1,
        kind=MemoryKind.PERSONALITY,
        content="User prefers simple beginner friendly explanations.",
        importance=0.9,
        confidence=1.0,
    )

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="programming_goal",
        layer=MemoryLayer.L2,
        kind=MemoryKind.GOAL,
        content="User wants to improve programming and coding skills.",
        importance=0.9,
        confidence=1.0,
    )

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="ai_project",
        layer=MemoryLayer.L2,
        kind=MemoryKind.GOAL,
        content="User is building an Agentic AI project called NISHI.",
        importance=1.0,
        confidence=1.0,
    )

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="c_pointers",
        layer=MemoryLayer.L3,
        kind=MemoryKind.HISTORY,
        content="User previously studied C pointers and structures.",
        importance=0.7,
        confidence=0.9,
    )

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="verilog",
        layer=MemoryLayer.L3,
        kind=MemoryKind.HISTORY,
        content="User worked on Verilog MUX and D Flip Flop programs.",
        importance=0.6,
        confidence=0.9,
    )

    service.save_memory(
        owner_id=OWNER_ID,
        stable_key="verified_action",
        layer=MemoryLayer.L4,
        kind=MemoryKind.ACTION_EVENT,
        content="NISHI successfully completed a verified memory retrieval action.",
        importance=0.5,
        confidence=1.0,
    )


def test_score(service):

    candidates = service.candidate_retrieval(
        owner_id=OWNER_ID,
        query="programming coding",
        limit=20,
    )

    assert len(candidates) > 0

    score = calculate_p2_score(
        candidates[0]
    )

    assert score >= 0

    print("TEST 1 PASSED: P2 score calculation")


def test_reranking(service):

    candidates = service.candidate_retrieval(
        owner_id=OWNER_ID,
        query="programming coding",
        limit=20,
    )

    ranked = rerank_candidates(candidates)

    assert len(ranked) > 0

    print("TEST 2 PASSED: Candidate reranking")


def test_top_n(service):

    candidates = service.candidate_retrieval(
        owner_id=OWNER_ID,
        query="user programming coding",
        limit=20,
    )

    selected = select_memories(
        candidates,
        top_n=5,
    )

    assert len(selected) <= 5

    print("TEST 3 PASSED: Top-N selection")


def test_selected_ids(service):

    candidates = service.candidate_retrieval(
        owner_id=OWNER_ID,
        query="programming",
        limit=20,
    )

    selected = select_memories(
        candidates,
        top_n=5,
    )

    ids = get_selected_memory_ids(selected)

    assert len(ids) == len(selected)

    for memory_id in ids:
        assert isinstance(memory_id, str)

    print("TEST 4 PASSED: Selected memory IDs")


def test_p1_p2_pipeline(service):

    selected_memories, selected_ids = retrieve_and_select(
        provider=service,
        owner_id=OWNER_ID,
        query="NISHI AI project",
        top_n=5,
    )

    assert len(selected_memories) <= 5
    assert len(selected_ids) == len(selected_memories)

    print("TEST 5 PASSED: P1 -> P2 pipeline")


def test_final_context(service):

    context = retrieve_final_context(
        provider=service,
        owner_id=OWNER_ID,
        query="NISHI AI project",
        top_n=5,
    )

    assert context.query == "NISHI AI project"
    assert len(context.memories) <= 5

    print("TEST 6 PASSED: P1 -> P2 -> P1 final context")


def display_results(service, query):

    print("\n" + "=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    context = retrieve_final_context(
        provider=service,
        owner_id=OWNER_ID,
        query=query,
        top_n=5,
    )

    if len(context.memories) == 0:
        print("No relevant memories found.")
        return

    print("\nSelected memories:")

    for i, memory in enumerate(
        context.memories,
        start=1,
    ):

        print("\n", i)
        print("ID:", memory.memory_id)
        print("Layer:", memory.layer.value)
        print("Kind:", memory.kind.value)
        print("Content:", memory.content)
        print("Importance:", memory.importance)
        print("Confidence:", memory.confidence)


def manual_test(service):

    print("\n" + "=" * 60)
    print("MANUAL TEST")
    print("=" * 60)

    print("Type a query.")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nEnter your query: ").strip()

        if query.lower() == "exit":
            print("Manual test finished.")
            break

        if query == "":
            print("Please enter a query.")
            continue

        display_results(
            service,
            query,
        )


def main():

    service = MemoryService()

    create_test_memories(service)

    print("\nRunning automatic tests...\n")

    test_score(service)
    test_reranking(service)
    test_top_n(service)
    test_selected_ids(service)
    test_p1_p2_pipeline(service)
    test_final_context(service)

    print("\nALL AUTOMATIC TESTS PASSED!")

    manual_test(service)

    service.storage.close()


if __name__ == "__main__":
    main()
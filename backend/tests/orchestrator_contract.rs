use std::collections::HashSet;

use socartes_backend::{SocartesOrchestrator, StoryChunk, StoryRagIndex, haunted_pajamas_index};

#[test]
fn orchestrator_runs_full_agentic_learning_cycle() {
    let orchestrator = SocartesOrchestrator::new();

    let trace = orchestrator.run(
        "Compare RAG agents with MCP tool-using agents for a research workflow.",
        "The learner wants a concise, citation-backed explanation.",
    );

    assert_eq!(trace.plan.agent, "planner");
    assert_eq!(
        trace
            .plan
            .tasks
            .iter()
            .map(|task| task.owner.as_str())
            .collect::<Vec<_>>(),
        vec!["planner", "retriever", "executor", "critic"]
    );
    assert!(!trace.retrieved_context.is_empty());

    let source_ids = trace
        .retrieved_context
        .iter()
        .map(|chunk| chunk.source_id.as_str())
        .collect::<HashSet<_>>();
    assert!(source_ids.contains("rag-index-18"));
    assert!(source_ids.contains("workflow-note-01"));

    let adapters = trace
        .tool_results
        .iter()
        .map(|result| result.adapter.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(
        adapters,
        HashSet::from(["external_api", "knowledge_database", "filesystem"])
    );
    assert!(!trace.draft.citations.is_empty());
    assert_eq!(trace.review.agent, "critic");
    assert_eq!(trace.review.status, "approved");
    assert!(!trace.reflection_events.is_empty());
    assert_eq!(
        trace.reflection_events.last().unwrap().event_type,
        "planner_update"
    );
    assert!(trace.final_answer.contains("RAG"));
    assert!(trace.final_answer.contains("MCP"));
}

#[test]
fn agent_catalog_exposes_role_boundaries() {
    let orchestrator = SocartesOrchestrator::new();

    let catalog = orchestrator.agent_catalog();

    for role in ["planner", "executor", "critic", "retriever", "tool_adapter"] {
        assert!(catalog.contains_key(role));
    }
    assert!(catalog["planner"]["responsibility"].starts_with("Convert learner goals"));
    assert!(catalog["critic"]["checks"].contains("acceptance criteria"));
}

#[test]
fn story_rag_answers_obscure_plot_questions_from_database_chunks() {
    let index = StoryRagIndex::new(vec![
        StoryChunk::new(
            "haunted-pajamas-ch01-muffler",
            "The Haunted Pajamas, Chapter 1",
            "The narrator tells Jenkins that the tight roll of bright red silk looks like it might be a red silk muffler.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch01-present",
            "The Haunted Pajamas, Chapter 1",
            "After untying the string, the narrator exclaims that the gift is a suit of pajamas.",
        ),
        StoryChunk::new(
            "haunted-pajamas-ch01-tarantula",
            "The Haunted Pajamas, Chapter 1",
            "Jenkins looks into one leg of the pajamas and says there is a tarantula in there, big as a sand crab, and alive.",
        ),
    ]);

    let muffler_answer = index.ask("What did the narrator first think the red silk roll might be?");
    assert!(muffler_answer.grounded);
    assert_eq!(
        muffler_answer.source_ids,
        vec!["haunted-pajamas-ch01-muffler"]
    );
    assert!(
        muffler_answer
            .answer
            .to_lowercase()
            .contains("red silk muffler")
    );

    let present_answer = index.ask("What was the gift after the string was untied?");
    assert!(present_answer.grounded);
    assert_eq!(
        present_answer.source_ids,
        vec!["haunted-pajamas-ch01-present"]
    );

    let tarantula_answer = index.ask("What did Jenkins say was in the pajama leg?");
    assert!(tarantula_answer.grounded);
    assert_eq!(
        tarantula_answer.source_ids,
        vec!["haunted-pajamas-ch01-tarantula"]
    );
    assert!(tarantula_answer.answer.to_lowercase().contains("tarantula"));
    assert!(tarantula_answer.answer.to_lowercase().contains("sand crab"));
}

#[test]
fn story_rag_refuses_unrelated_question_even_when_title_is_named() {
    let answer = haunted_pajamas_index().ask("In The Haunted Pajamas, who kills Sherlock Holmes?");

    assert!(!answer.grounded);
    assert!(answer.source_ids.is_empty());
    assert!(answer.answer.contains("not have enough evidence"));
}

#[test]
fn story_rag_covers_expanded_twelve_question_evaluation_set() {
    let index = haunted_pajamas_index();
    let answerable_questions = [
        (
            "What name and address were printed on the package box?",
            "haunted-pajamas-ch01-sender",
            "roland mastermann",
        ),
        (
            "Who did Jenkins think Mastermann was?",
            "haunted-pajamas-ch01-carlton",
            "carlton",
        ),
        (
            "What did the narrator first think the red silk roll might be?",
            "haunted-pajamas-ch01-muffler",
            "red silk muffler",
        ),
        (
            "What debt did Mastermann say every puff of the rare cigars reminded him of?",
            "haunted-pajamas-ch01-debt",
            "still unpaid",
        ),
        (
            "Which cheap cigar brand was sent by mistake instead of Paloma perfectos?",
            "haunted-pajamas-ch02-hickeys-pride",
            "hickey's pride",
        ),
        (
            "What did Jenkins say a twofer meant?",
            "haunted-pajamas-ch02-twofer",
            "two for five",
        ),
        (
            "What was the gift after the string was untied?",
            "haunted-pajamas-ch02-present",
            "suit of pajamas",
        ),
        (
            "Who did Jenkins say the red pajamas reminded him of?",
            "haunted-pajamas-ch02-memphis-tuffles",
            "old memphis tuffles",
        ),
        (
            "What dropped into a fold of the pajamas?",
            "haunted-pajamas-ch02-spider",
            "little spider",
        ),
        (
            "What did Jenkins say was in the pajama leg?",
            "haunted-pajamas-ch02-tarantula",
            "tarantula",
        ),
    ];

    for (question, source_id, expected_text) in answerable_questions {
        let answer = index.ask(question);
        assert!(answer.grounded, "{question}");
        assert_eq!(answer.source_ids, vec![source_id], "{question}");
        assert!(
            answer.answer.to_lowercase().contains(expected_text),
            "{question}"
        );
    }

    for question in [
        "In The Haunted Pajamas, who kills Sherlock Holmes?",
        "In The Haunted Pajamas, what is the name of the spaceship captain?",
    ] {
        let answer = index.ask(question);
        assert!(!answer.grounded, "{question}");
        assert!(answer.source_ids.is_empty(), "{question}");
    }
}

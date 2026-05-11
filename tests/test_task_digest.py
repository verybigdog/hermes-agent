from agent.task_digest import RedactedTaskDigest, build_task_digest, format_task_digest_context


def test_formats_operator_digest_as_human_readable_prose_with_fallback_title():
    digest = build_task_digest(
        {
            "session_title": "Same-channel hydration investigation",
            "summary": "Checked diagnostics and found metadata-only preview path.",
            "progress": ["diagnostic script exists", "preview remains no-write"],
            "next_actions": ["run artifact preview before enabling injection"],
            "uncertainties": ["live gateway config not inspected in this slice"],
        }
    )

    rendered = format_task_digest_context(digest)

    assert isinstance(digest, RedactedTaskDigest)
    assert digest.title == "Same-channel hydration investigation"
    assert rendered.startswith("RedactedTaskDigest:")
    assert "작업:" in rendered
    assert "요약:" in rendered
    assert "진행:" in rendered
    assert "다음:" in rendered
    assert "불확실:" in rendered
    assert "live gateway config not inspected in this slice" in rendered
    assert "{" not in rendered
    assert "}" not in rendered


def test_redacts_obvious_secrets_env_tokens_and_private_payloads_from_all_fields():
    secret = "sk-test-redaction-fixture-1234567890"
    env_secret = "OPENAI_API_KEY=opensesame1234567890"
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.fixture.cdef"
    digest = build_task_digest(
        {
            "title": f"deploy token {secret}",
            "summary": f"env was {env_secret}",
            "progress": [f"auth header Authorization: Bearer {jwt}"],
            "next_actions": ["rotate if any raw value reached logs"],
            "uncertainties": ["raw_payload may contain hidden operator text"],
            "raw_payload": {"api_key": secret, "note": "must never render"},
        }
    )

    rendered = format_task_digest_context(digest)

    assert secret not in rendered
    assert "opensesame1234567890" not in rendered
    assert jwt not in rendered
    assert "must never render" not in rendered
    assert "raw_payload" not in rendered
    assert "***" in rendered or "..." in rendered


def test_safe_fields_take_precedence_over_raw_equivalents_in_formatted_digest():
    digest = build_task_digest(
        {
            "title": "RAW private task title",
            "safe_title": "Safe task title",
            "summary": "RAW private summary",
            "safe_summary": "Safe summary",
            "progress": ["RAW private progress"],
            "safe_progress": ["Safe progress"],
            "next_actions": ["RAW private next action"],
            "safe_next_actions": ["Safe next action"],
            "uncertainties": ["RAW private uncertainty"],
            "safe_uncertainties": ["Safe uncertainty"],
        }
    )

    rendered = digest.format_for_context()

    assert "Safe task title" in rendered
    assert "Safe summary" in rendered
    assert "Safe progress" in rendered
    assert "Safe next action" in rendered
    assert "Safe uncertainty" in rendered
    assert "RAW private" not in rendered


def test_missing_details_are_not_inflated_and_include_explicit_uncertainty():
    digest = build_task_digest({"title": "K0.7 slice"})

    rendered = format_task_digest_context(digest)

    assert "K0.7 slice" in rendered
    assert "제공된 요약 없음" in rendered
    assert "확인된 진행 항목 없음" in rendered
    assert "확인된 다음 action 없음" in rendered
    assert "세부 정보가 제공되지 않아 불확실함" in rendered
    assert "completed" not in rendered.lower()
    assert "done" not in rendered.lower()


def test_accepts_safe_marker_like_nested_inputs_without_rendering_raw_private_values():
    digest = build_task_digest(
        {
            "safe_title": "Kanban handoff",
            "safe_summary": {"text": "Need review of RedactedTaskDigest formatting", "confidence": "partial"},
            "safe_progress": [{"text": "RED test written"}, {"label": "implementation pending"}],
            "safe_next_actions": {"text": "review targeted diff"},
            "safe_uncertainties": [None, {"text": "schema migration deliberately omitted"}],
            "private_notes": "do not render this operator payload",
        }
    )

    rendered = digest.format_for_context()

    assert "Kanban handoff" in rendered
    assert "Need review of RedactedTaskDigest formatting" in rendered
    assert "partial" in rendered
    assert "RED test written" in rendered
    assert "implementation pending" in rendered
    assert "review targeted diff" in rendered
    assert "schema migration deliberately omitted" in rendered
    assert "private_notes" not in rendered
    assert "do not render this operator payload" not in rendered

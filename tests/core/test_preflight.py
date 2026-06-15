"""Tests for the API-key preflight check."""

from evolution.core.preflight import missing_api_keys, provider_of


def test_provider_resolution():
    assert provider_of("openai/gpt-4.1") == "openai"
    assert provider_of("openrouter/google/gemini-2.5-flash") == "openrouter"
    assert provider_of("ollama/qwen2.5:7b") == "ollama"
    assert provider_of("gpt-4.1") == "openai"  # bare name defaults to openai


def test_missing_keys_flagged_when_env_empty():
    errs = missing_api_keys(
        ["openai/gpt-4.1", "openrouter/google/gemini-2.5-flash"], env={}
    )
    assert any("OPENAI_API_KEY" in e for e in errs)
    assert any("OPENROUTER_API_KEY" in e for e in errs)


def test_present_keys_pass():
    env = {"OPENAI_API_KEY": "x", "OPENROUTER_API_KEY": "y"}
    assert missing_api_keys(["openai/gpt-4.1", "openrouter/foo"], env=env) == []


def test_keyless_providers_never_flagged():
    assert missing_api_keys(["ollama/qwen2.5:7b", "local/model"], env={}) == []


def test_unknown_provider_not_flagged():
    # We only assert credentials we know how to check.
    assert missing_api_keys(["mystery/model"], env={}) == []


def test_duplicate_provider_reported_once():
    errs = missing_api_keys(
        ["openai/gpt-4.1", "openai/gpt-4.1-mini"], env={}
    )
    assert len(errs) == 1


def test_gemini_accepts_either_env_var():
    assert missing_api_keys(["gemini/pro"], env={"GOOGLE_API_KEY": "z"}) == []
    assert missing_api_keys(["gemini/pro"], env={"GEMINI_API_KEY": "z"}) == []

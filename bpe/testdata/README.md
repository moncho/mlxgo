# Tokenizer Test Fixtures

`tokenizer.json` is a reduced and reserialized derivative of the pinned Qwen
tokenizer, not an original or full checkpoint tokenizer. It retains only the
vocabulary and merge rules exercised by the test corpus. `qwen2/make_fixture.py`
generates this fixture and the independent parity cases.

See [provenance and modification notice](../../THIRD_PARTY_NOTICES.md) and the
included [upstream Apache 2.0 license](../THIRD_PARTY_LICENSE).

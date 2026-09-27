# Releasing mlxgo

Releases are explicit maintainer actions, not automatic publications on every
merge. Release source only; do not include local model weights, training data,
credentials, caches or unrequested design specs.

1. Choose a new semantic version following `COMPATIBILITY.md`. Check both
   existing Git tags and GitHub releases. Never replace a published tag.
2. Update `VERSION.txt` and `RELEASE_NOTES.md`; the first notes line must be
   `# vX.Y.Z`. Review install instructions, supported configurations, license
   notices and known limitations. Keep model packages labeled experimental.
3. Run stub/native tests and vet, the native runtime suite, focused race tests
   and examples. Run available pinned real-weight tests separately; record
   skipped coverage rather than treating skipped tests as successful validation.
4. Commit only reviewed release files as the maintainer and push the candidate.
   Require its complete Go 1.26/1.27 CI matrix to pass. Check the working tree
   for unintended tracked changes and explicitly exclude local specs.
5. Create an annotated `vX.Y.Z` tag at that exact tested commit and push it.
   The Release workflow validates version/notes, reruns the complete CI matrix,
   and only then creates the GitHub source release. A tag alone does not prove
   the publication gate passed; Go tooling can still resolve a pushed tag.
6. Verify the published release points to the intended tag/commit and check
   module resolution from a clean temporary consumer. GitHub supplies source
   archives; native dependencies and model weights are installed separately.

If a tag's checks fail, do not bypass the gate or move the tag. Diagnose the
failure, fix it in a new commit, and choose a new version when necessary. A
transient infrastructure failure can be rerun on the same unchanged commit.

The workflow has read-only permissions during validation; only the final
publication job receives repository-content write permission. No local release
script stores or prints authentication tokens.

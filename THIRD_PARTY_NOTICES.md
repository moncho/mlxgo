# Third-Party Notices

The Go implementation is distributed under the repository's MIT license.
The following bundled reference material retains its upstream terms.

## Qwen Tokenizer Fixture

`bpe/testdata/tokenizer.json` is a modified, reduced fixture derived from
Qwen/Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The generator
`qwen2/make_fixture.py` retains the vocabulary and merge rules needed by the
test corpus and serializes the reduced tokenizer; it is not the full tokenizer.

Source: [pinned Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).
The upstream Apache License 2.0 is included at
[bpe/THIRD_PARTY_LICENSE](bpe/THIRD_PARTY_LICENSE), copied from the
[pinned license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE).
This notice does not relicense that fixture under MIT.

## DeepSeek Reference Material

The DeepSeek package records its reference revision and attribution in its
source and validation reports. Its upstream notice is retained at
[deepseek/THIRD_PARTY_LICENSE](deepseek/THIRD_PARTY_LICENSE).

## External Dependencies

MLX, mlx-c, Go module dependencies and downloaded model checkpoints are not
bundled as native libraries or model weights in this release. Their own
licenses and notices apply separately. The source release includes only the
documented small reference fixtures, not full model weights.

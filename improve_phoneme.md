# Vietnamese Phoneme Frontend Improvements for Kokoro

This note summarizes the Vietnamese phoneme work implemented for Kokoro
Vietnamese. It is intended as a technical basis for writing a paper comparing
this frontend with an espeak-style Vietnamese frontend.

## Objective

The goal was not only to replace the G2P backend, but to build a consistent
Vietnamese phoneme frontend for the full TTS pipeline:

- dataset preprocessing
- StyleTTS2 training lists
- TensorBoard preview synthesis
- Gradio checkpoint testing
- packaged inference

The main design constraint is that Kokoro has a fixed text vocabulary. A
Vietnamese frontend therefore needs two layers:

1. A Vietnamese-aware G2P engine.
2. A Kokoro-compatible adapter that preserves Vietnamese contrasts before
   mapping symbols into the Kokoro vocabulary.

## Frontend Architecture

The implemented flow is:

```text
Vietnamese text
  -> vig2p / sea-g2p based Vietnamese G2P
  -> contrast-preserving Kokoro vocab adapter
  -> Kokoro/StyleTTS2 text tokens
```

The same frontend is used across training and inference. This avoids a common
failure mode where training data is phonemized one way, while Gradio or packaged
inference uses another frontend.

Implemented entrypoints include:

- `scripts/prepare_vietnamese_dataset.py`
- `scripts/audit_vietnamese_frontend.py`
- `StyleTTS2/kokoro_tb_utils.py`
- `kokoro/kokoro/pipeline.py`
- `src/kokoro_vietnamese/core.py`

## Why Not Plain Espeak-Vi

An espeak-style frontend can be useful as a baseline, but it is not enough for
this Kokoro Vietnamese setup for three reasons:

1. Vietnamese contrasts can collapse after mapping into Kokoro's constrained
   vocabulary.
2. Tone needs to remain explicit in the text token stream.
3. Mixed Vietnamese and English text needs controlled behavior rather than a
   blanket Vietnamese rewrite.

The implemented frontend therefore focuses on preserving contrast before
vocabulary adaptation.

## Kokoro Vocabulary Adapter

The adapter maps G2P output into symbols supported by Kokoro.

Core mappings:

```text
tʃ  -> ʧ
e-  -> æ
1/7 -> →
2   -> ↘
3/ɜ -> ↗
4   -> ↓
5   -> ʔ↗
6   -> ʔ↓
ɗ   -> d
ʐ   -> ʒ
đ   -> d
–   -> —
```

Cleanup rules:

```text
strip leftover: -, ', ’, ‘, ̪, ̩
remove unsupported: *
replace unsupported separators: /, &
```

The important detail is that these mappings are not applied as blind cleanup
only. Orthographic context is used first where needed, so that important
Vietnamese contrasts are not lost.

## Preserved Vietnamese Contrasts

### `t` vs `th`

The original bug that exposed this issue was:

```text
tường nhà khách
```

being pronounced too close to:

```text
thường nhà khách
```

The frontend now preserves the onset contrast:

```text
tường  -> tˈyə↘ŋ
thường -> θˈyə↘ŋ

teo  -> tˈɛw
theo -> θˈɛw
```

The rule also works for unaccented Vietnamese words such as `theo`.

### `tr` vs `ch`

The frontend preserves the orthographic onset distinction:

```text
tr -> ʈʂ
ch -> ʧ
```

Examples:

```text
trước -> ʈʂˈyə↗c
chước -> ʧˈyə↗c
```

Regression pairs include:

```text
trước/chước
trúng/chúng
trị/chị
trắc/chắc
```

### `s` vs `x`

The frontend avoids collapsing Vietnamese `s` and `x`:

```text
Vietnamese s -> ʂ
Vietnamese x -> s
```

Examples:

```text
số   -> ʂˈo↗
xố   -> sˈo↗

sinh -> ʂˈiɲ
xinh -> sˈiɲ

sao  -> ʂˈaːw
xao  -> sˈaːw
```

There is an English guard for obvious non-Vietnamese `s` clusters, so words such
as `start` and `style` remain:

```text
start -> stˈɑːɹt
style -> stˈaɪl
```

### `gi` vs `d`

The frontend preserves `gi` and `d` as separate onsets:

```text
gi -> ʝ
d  -> z
```

Examples:

```text
giải -> ʝˈaː↓j
dải  -> zˈaː↓j

gia -> ʝˈaː
da  -> zˈaː
```

### Vowel Contrast From `e-`

The raw G2P output can use `e-` for a vowel contrast that would be lost if `-`
were stripped too early. The adapter maps:

```text
e- -> æ
```

Examples:

```text
cách -> kˈæ↗c
kếch -> kˈe↗c

mạnh -> mˈæʔ↓ɲ
mệnh -> mˈeʔ↓ɲ
```

## Tone Encoding

Vietnamese tones are kept as explicit text tokens rather than removed.

Current tone mapping:

```text
1/7 -> →
2   -> ↘
3/ɜ -> ↗
4   -> ↓
5   -> ʔ↗
6   -> ʔ↓
```

Examples:

```text
tường -> tˈyə↘ŋ
trước -> ʈʂˈyə↗c
mạnh  -> mˈæʔ↓ɲ
```

This gives the text encoder a direct representation of Vietnamese lexical tone.

## Mixed Vietnamese and English Text

The frontend includes safeguards for mixed-language input. It avoids rewriting
obvious English words as Vietnamese phonemes.

Example:

```text
team start heart don't
```

becomes:

```text
tˈiːm stˈɑːɹt hˈɑːɹt dˈoʊnt
```

It does not incorrectly produce `θˈiːm` for `team`, nor rewrite `start` into a
Vietnamese `s/th` sequence.

Punctuation normalization also prevents unsupported symbols from entering the
training phoneme stream:

```text
don’t -> normalized apostrophe behavior
–     -> —
*, /, & are removed or converted safely
```

## Dataset Audit

An audit script was added to detect frontend problems before GPU training:

```bash
python scripts/audit_vietnamese_frontend.py \
  --metadata metadata.csv \
  --fail-on-minimal-pairs \
  --fail-on-adapter-collisions
```

The audit checks:

- unknown symbols outside Kokoro vocabulary
- minimal-pair collapse
- adapter-induced phoneme collisions
- same-phoneme groups in the dataset vocabulary

Default minimal pairs include:

```text
tường/thường
tơ/thơ
cách/kếch
mạnh/mệnh
lạnh/lệnh
bạch/bệch
cành/kềnh
ngách/nghếch
đanh/đênh
gành/ghềnh
trước/chước
trúng/chúng
trị/chị
trắc/chắc
số/xố
sử/xử
sếp/xếp
giải/dải
gì/dì
gia/da
```

This audit is designed to catch silent frontend errors before long training
runs.

## Regression Tests

The frontend is covered by tests for:

- `t/th` contrast
- unaccented `t/th` words
- `tr/ch` contrast
- `s/x` contrast
- `gi/d` contrast
- `e-/e` vowel contrast
- tone-token mapping
- mixed Vietnamese/English text
- unsupported punctuation cleanup
- consistency across preprocess, TensorBoard preview, Gradio, and inference

Representative expected outputs:

```text
tường nhà khách  -> tˈyə↘ŋ ɲˈaː↘ xˈæ↗c
thường nhà khách -> θˈyə↘ŋ ɲˈaː↘ xˈæ↗c

trước -> ʈʂˈyə↗c
chước -> ʧˈyə↗c

sinh -> ʂˈiɲ
xinh -> sˈiɲ

giải -> ʝˈaː↓j
dải  -> zˈaː↓j
```

## Paper-Ready Summary

The proposed frontend replaces an espeak-style Vietnamese frontend with a
Vietnamese-specific G2P plus a Kokoro-compatible adapter. The adapter preserves
Vietnamese onset, vowel, and tone contrasts before mapping into Kokoro's fixed
symbol inventory. In particular, it prevents collapse of `t/th`, `tr/ch`,
`s/x`, `gi/d`, and selected `e-/e` vowel contrasts, while encoding Vietnamese
tones as explicit text tokens. A dataset-level audit detects minimal-pair
collapse, unknown vocabulary symbols, and adapter-induced collisions before
training.

## Claims That Are Supported by Implementation

Supported claims:

- the frontend is consistent between training and inference
- Vietnamese lexical tone is preserved as explicit token information
- several Vietnamese minimal-pair contrasts are preserved by construction
- the adapter is Kokoro-vocabulary-compatible
- the audit can detect minimal-pair collapse before training

Claims that require separate experiments:

- MOS improvement over espeak
- speaker similarity improvement
- WER/CER improvement
- prosody improvement measured by objective metrics
- statistically significant preference in listening tests

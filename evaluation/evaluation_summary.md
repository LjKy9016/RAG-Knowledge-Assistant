# RAG Evaluation Summary

## 1. Evaluation Setup

The evaluation used 22 test turns:

- 18 independent questions
- 4 turns from two multi-turn conversations
- English and Chinese questions
- Answerable and unanswerable questions
- Questions covering all seven knowledge base documents

The embedding and reranker models were loaded before latency measurement. This prevented model download and initial loading time from affecting the request latency results.

The default configuration was:

| Setting | Value |
|---|---:|
| Embedding model | `intfloat/multilingual-e5-small` |
| Reranker | `amber-tech/bert-multilingual-passage-reranking-msmarco` |
| LLM | `openai/gpt-oss-120b` |
| Final top-k | 2 |
| Reranker | Enabled |
| Temperature | 0.2 |

## 2. Scoring Method

Accuracy and faithfulness were checked manually against the reference answers and expected documents. Manual scoring was used instead of asking the same LLM to evaluate its own answers.

Accuracy was scored as follows:

- `1`: the response contained the required correct information, or correctly refused an unanswerable question
- `0`: the response was incorrect, missed required information, or refused an answerable question

Faithfulness was scored as follows:

- `1`: all factual claims were supported by the expected document
- `0.5`: the main answer was supported, but a minor factual claim was unclear or unsupported
- `0`: the answer contained an unsupported or conflicting factual claim

Responses without a substantive factual answer were excluded from the faithfulness calculation. Faithfulness was therefore calculated across 15 factual answers.

Context Precision was calculated at source level. A returned source was relevant when its file name matched the expected document. For an unanswerable question, the score was `1` only when no source was returned.

## 3. Baseline Results

| Metric | Result | Target | Status |
|---|---:|---:|---|
| Accuracy | 81.82% | ≥80% | Passed |
| Faithfulness | 100.00% | ≥85% | Passed |
| Context Precision | 79.55% | ≥70% | Passed |
| Requests completed within 10 seconds | 100.00% | ≥90% | Passed |
| Average latency | 3,224.33 ms | — | — |
| P90 latency | 5,798.65 ms | <10,000 ms | Passed |
| Maximum latency | 9,982.64 ms | — | — |
| Average prompt tokens | 540.86 | — | — |
| Average completion tokens | 87.32 | — | — |
| Average total tokens | 628.18 | — | — |

All four multi-turn questions were answered correctly. The system also correctly refused all three questions that could not be answered from the knowledge base.

## 4. Failed Questions

Four answerable questions were not answered correctly.

| ID | Question | Observed issue |
|---|---|---|
| Q04 | 求职申请在决定作出后保留多长时间？ | The Chinese question did not retrieve the expected English data protection content. |
| Q09 | What is the standard Orbit API rate limit? | The expected API document was returned, but the selected context did not produce the required answer. |
| Q11 | What is the availability target for the Orbit system? | The relevant architecture content was not selected. |
| Q12 | What are the recovery time and recovery point objectives? | Unrelated documents were retrieved and the request was close to the 10-second limit. |

These failures show that the main remaining issue is retrieval recall. Possible improvements include hybrid semantic and keyword search, better document metadata filtering, and further tuning of chunk boundaries.

## 5. Sensitivity Analysis

Six representative questions were tested under three configurations. The subset included successful, failed, bilingual, technical and unanswerable questions. Because this was a deliberately difficult subset, its Accuracy and Context Precision should not be compared directly with the full 22-turn evaluation.

| Configuration | top-k | Reranker | Temperature | Accuracy | Context Precision | Average latency | P90 latency | Average tokens |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2 | Enabled | 0.2 | 50.00% | 58.33% | 1,935.10 ms | 2,875.59 ms | 565.17 |
| Precise | 1 | Enabled | 0.0 | 50.00% | 66.67% | 3,034.71 ms | 5,674.63 ms | 433.83 |
| Faster | 3 | Disabled | 0.5 | 66.67% | 50.00% | 1,943.11 ms | 3,434.71 ms | 693.67 |

The Precise configuration used the fewest tokens and achieved the highest Context Precision, but it did not improve Accuracy. The Faster configuration answered Q09 correctly because more raw retrieval results were supplied to the LLM. However, it returned more irrelevant context and used the most tokens.

The Baseline configuration remains the default because it passed all targets in the full evaluation and provides a more balanced combination of quality, latency and token use.

## 6. Cost Estimate

The prototype currently uses the Groq free tier, so no API charge was incurred during development and evaluation.

For a production estimate, the published Groq GPT-OSS 120B prices were used:

- Input: $0.15 per one million tokens
- Cached input: $0.075 per one million tokens
- Output: $0.60 per one million tokens

Source: [Groq GPT-OSS 120B model documentation](https://console.groq.com/docs/model/openai/gpt-oss-120b)

| Configuration | Estimated cost per 1,000 calls | With cached input |
|---|---:|---:|
| Full baseline evaluation | $0.1335 | $0.0930 |
| Sensitivity baseline | $0.1248 | $0.0891 |
| Precise | $0.1038 | $0.0778 |
| Faster | $0.1436 | $0.0982 |

The main production estimate is approximately **$0.1335 per 1,000 calls**. This is a conservative estimate using the non-cached input price.

The estimate does not include server, CPU, memory, storage, embedding or reranker infrastructure costs.

## 7. Security and PII Logging Checks

Prompt injection checks were tested in English and Chinese. Requests asking the assistant to ignore its rules or reveal the system prompt were refused before retrieval and LLM generation.

Example security result:

```json
{
  "outcome": "security_refusal",
  "sources": [],
  "prompt_tokens": 0,
  "completion_tokens": 0
}
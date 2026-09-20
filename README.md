# faithbench

[![CI](https://github.com/your-username/faithbench/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/faithbench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/faithbench.svg)](https://pypi.org/project/faithbench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Claim-level faithfulness checking for RAG answers. Breaks a generated answer into individual
claims and scores each one against your retrieved evidence, using whichever combination of
methods you want -- lexical overlap, NLI entailment, an LLM judge, RAGAS-style binary checking,
or self-consistency across independent re-generations.

Provider-agnostic by design: the LLM-based methods take a plain function you write, not a
specific SDK, so this works with OpenAI, Anthropic, a local model, or anything else.

```bash
pip install faithbench          # zero extra dependencies
pip install faithbench[rouge]   # + the ROUGE-L method (pure Python, lightweight)
pip install faithbench[nli]     # + a local NLI cross-encoder (downloads a ~140MB model)
pip install faithbench[all]     # both
```

```python
from faithbench import FaithfulnessChecker

checker = FaithfulnessChecker(methods=["rouge_l"])
result = checker.check(
    answer="RAG reduces hallucination by grounding answers in retrieved evidence.",
    context=["Retrieval-Augmented Generation grounds generated answers in retrieved evidence."],
)

result.overall_score          # 0.76
result.claims[0].status()     # "supported"
result.to_dict()              # plain dict, JSON-serializable
```

## Why not just use one method?

No single faithfulness metric perfectly predicts human judgment -- lexical overlap misses
paraphrases, NLI models miss claims with unusual structure, LLM judges are expensive and can be
inconsistent. faithbench doesn't pick one for you: it runs whichever methods you register and
gives you every method's score per claim, so you can compare them against your own labeled
examples and find out which one (or which combination) actually works for your data.

## Methods

| Method | Install | Needs an LLM? |
|---|---|---|
| `rouge_l` | `pip install faithbench[rouge]` | No |
| `nli` | `pip install faithbench[nli]` | No (local model) |
| `llm_judge` | built in | Yes -- you supply the call |
| `ragas_faithfulness` (`faithbench.methods.ragas`) | built in | Yes -- you supply the call |
| `self_consistency` | built in | Yes -- you supply the call |

### Using an LLM-based method

LLM methods take a plain function instead of an SDK, so you're never locked into a provider:

```python
from faithbench.methods import llm_judge

def my_llm(system_prompt: str, user_prompt: str) -> str:
    response = your_client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content

checker = FaithfulnessChecker(methods=["rouge_l", llm_judge.make_scorer(my_llm)])
```

`self_consistency` needs the original question too, since it re-answers it independently:

```python
from faithbench.methods import self_consistency

def my_generate(prompt: str) -> str:
    return your_client.complete(prompt)

sc = self_consistency.make_scorer(my_generate, question="How does RAG reduce hallucination?")
checker = FaithfulnessChecker(methods=["rouge_l", sc])
```

## No fabricated results

Any method that can't run -- missing dependency, network failure, an unparseable model response
-- returns `score=None` with an explanation in `.error`, for exactly the claims it couldn't
score. Other methods keep running. `None` is never silently treated as zero; check
`aggregate_score is None` (or `.status() == "unevaluated"`) to detect it.

## Custom methods

Any object with a `.name` string and a `.score_claims(claims, evidence) -> list[MethodScore]`
method works -- register it alongside the built-ins:

```python
class MyMethod:
    name = "my_method"
    def score_claims(self, claims, evidence):
        return [MethodScore(method="my_method", score=0.5) for _ in claims]

checker = FaithfulnessChecker(methods=["rouge_l", MyMethod()])
```

## Origin

Extracted from [ContextGuard](https://github.com/your-username/contextguard), a research
project comparing these same methods against human-labeled examples for a RAG hallucination
detection study.

## Publishing (for maintainers)

CI (`.github/workflows/ci.yml`) runs the test suite and a package-metadata check on every push.
Publishing to PyPI (`.github/workflows/publish.yml`) uses
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) -- no API token stored in the
repo. One-time setup, after the repo is pushed to GitHub:

1. Create a PyPI account (if you don't have one) and reserve the project name: `pip install
   build twine && python -m build && twine upload dist/*` for a *first* manual upload, **or**
   set up Trusted Publishing before any upload via PyPI's "Publish a new project" flow.
2. On PyPI: **Account settings -> Publishing -> Add a new pending publisher**, with:
   - PyPI project name: `faithbench` (or whatever you renamed it to)
   - Owner: your GitHub username/org, Repository: `faithbench`, Workflow: `publish.yml`,
     Environment: `pypi`
3. In the GitHub repo: **Settings -> Environments -> New environment** named `pypi` (matches
   the workflow's `environment: pypi`).
4. To release: bump `version` in `pyproject.toml`, commit, then create a GitHub Release with a
   tag like `v0.1.1` -- the publish workflow runs automatically and uploads to PyPI.

**Before your first publish:** replace every `your-username` placeholder in this README,
`pyproject.toml`'s `[project.urls]`, and the CI badge URLs with your actual GitHub username, and
double-check the package name is still available (`pypi.org/project/faithbench/` should 404).

## License

MIT

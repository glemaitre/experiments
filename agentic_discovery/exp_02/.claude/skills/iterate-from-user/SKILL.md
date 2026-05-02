---
name: iterate-from-user
description: >
  Source the next ML experiment proposal from the user directly,
  or from a GitHub issue tracker the user has pointed us at. Hand
  the proposal back to `iterate-ml-experiment`, which writes it
  into `plan/NN_short_name.md` and seeks the user's approval.
  Stops at "a proposal (question, motivation, method outline,
  success criteria) has been returned"; does not write any plan
  file itself.

  TRIGGER when: `iterate-ml-experiment` is picking a sourcing
  strategy and the user has offered a concrete idea ("I want to
  try X", "let's add Y", "tweak the encoder"); the user pastes or
  links a GitHub issue / discussion that describes the next
  experiment; the user says "use the issue tracker" or "check
  issue #N".

  SKIP when: the user is open-ended ("what's next?") with no idea
  in hand — try a different strategy (diagnostic / methodology /
  literature) first; the user is asking for a symbol lookup or
  pipeline mechanics (use the `*-api` skills); there is no `gh`
  CLI / GitHub access and the user wants the issue tracker —
  surface the gap and fall back to direct user input.

  HOW TO USE: this skill is shallow — it elicits the proposal
  and returns it. If the user has a verbal idea, ask the four
  shaping questions in the body and synthesize a proposal. If a
  GitHub issue is the source, fetch it with `gh issue view <N>`
  (or `gh api`), summarize it through the same four-question
  lens, and flag anything the issue doesn't answer. Always
  return: question / motivation / method outline / success
  criteria — not a plan file.
---

# Iterate from user

Source: the user (directly, or via a GitHub issue they own).
Output: a proposal handed back to `iterate-ml-experiment`.

## Output contract (read this before the body)

This skill **never writes `plan/` files**. It returns a
**Proposal block** back to `iterate-ml-experiment` (full shape
in § What is returned at the bottom): `Question`, `Motivation`
(with the user quote or issue link as `Source`), `Method
outline`, `Success`, `Open gaps`. Required:

- Every Proposal must answer all **four shaping questions**
  (see § The four shaping questions). Missing → ask the user
  before returning.
- **GitHub-issue path:** check `gh auth status` first;
  resolve repo per § Resolution priority; fetch comments if
  the issue body is short.
- **Goal shifts** (different output shape, downstream
  consumer, or metric class) require user confirmation of a
  PLAN.md Status update **before** returning the proposal —
  see § Stop conditions.

There is **no "no proposal" outcome** for this skill: it only
fires when the user has volunteered an idea (or pointed at an
issue). If the user is open-ended without an idea, the
dispatcher in `iterate-ml-experiment` asks them directly
instead of invoking this skill.

## Stop conditions

- **Don't write `plan/` files.** That belongs to
  `iterate-ml-experiment`. This skill returns a proposal as
  conversation text or structured fields; the parent skill
  drafts the file.
- **Don't infer an issue's content.** If the user references an
  issue, fetch it (`gh issue view <N>` / `gh api`) — don't
  reconstruct from the issue title alone.
- **Don't paper over missing fields.** If the user's idea (or
  the issue) doesn't answer one of the four shaping questions,
  surface the gap to the user before returning the proposal.
- **Check `gh` auth before fetching anything.** Before any
  `gh issue view` / `gh api` call, run `gh auth status`
  (cheap, cached). If unauthenticated or on the wrong host,
  **do not** retry blindly — ask the user to either run
  `gh auth login` themselves (suggest they type `! gh auth
  login` in the prompt so it runs in this session) or paste
  the issue body directly. A failed `gh` call surfaces a
  confusing error; the auth check makes the failure mode
  explicit.
- **Flag goal shifts before returning the proposal.** If the
  user's idea (or the issue) materially changes the **project
  goal** as recorded in `PLAN.md` Status — different output
  shape (point estimate → prediction interval), different
  downstream consumer (offline batch → online serving),
  different metric class (squared error → coverage) — that's
  not just a method change. Surface it as a question to the
  user **before** returning the proposal: *"this would update
  PLAN.md Status from <X> to <Y>; confirm or amend the goal
  first?"* The parent skill's per-experiment plan file should
  not silently re-define what success means while the Status
  block still reflects the old goal.

## The four shaping questions

Every proposal returned from this skill must answer:

1. **What are we trying to learn?** (turns "try X" into a
   hypothesis)
2. **Why now?** (the specific reason this idea surfaced — quote
   the user, link the issue)
3. **What changes vs. the previous experiment?** (which file in
   `src/<pkg>/` is touched, in prose — not code)
4. **How will we know it answered the question?** (a metric
   delta, a diagnostic flip, a plot change)

Missing → ask the user. Don't fabricate.

## Two intake paths

### Direct user input

The user has a verbal or written idea. Ask the four questions in
order, in plain language. Quote them back when summarizing the
proposal so the framing stays theirs. Hand the synthesis to
`iterate-ml-experiment`.

### GitHub issue tracker

The user pointed us at an issue (number, link, or "check the
tracker").

**Resolution priority (never silently guess).** Pick the
owner/repo using the first rule that matches:

1. **Explicit URL** in the user's message
   (`https://github.com/<owner>/<repo>/issues/<N>`) — wins
   unconditionally.
2. **`org/repo#N` shorthand** in the user's message
   (`probabl-ai/skore#42`) — wins over current context.
3. **Bare `#N` or "issue 42"** with no qualifier — fall back to
   the current `gh` context (`gh repo view --json
   nameWithOwner` to confirm). If that returns nothing, ask
   the user before fetching.

Then fetch:

- `gh issue view <N> --json title,body,labels,url` for the
  baseline.
- **If the issue body is short (<200 chars) or visibly
  under-specified**, also pull the latest comments —
  `gh issue view <N> --json title,body,labels,url,comments`
  or `gh api repos/<owner>/<repo>/issues/<N>/comments` — and
  read the most recent ~5. The actual proposal often lives
  in the thread, not the top-post.

Map the assembled issue (body + relevant comments) to the four
shaping questions; flag missing fields and ask the user to
clarify before returning the proposal. The proposal returned
must include the issue link as **Source** so the per-experiment
plan file can cite it.

## What is returned

A short structured block, not a plan file:

```
Proposal (from: user | issue #<N>):
  Question:        <one sentence>
  Motivation:      <quote / link>
  Method outline:  <prose; which file in src/<pkg>/ is touched>
  Success:         <metric delta / diagnostic flip / plot change>
  Open gaps:       <anything the user / issue didn't answer>
```

`iterate-ml-experiment` consumes this and drafts
`plan/NN_short_name.md`.

## Companion skills

- **`iterate-ml-experiment`** — the caller; owns the plan file.
- **`iterate-from-literature` / `iterate-from-methodology` /
  `iterate-from-diagnostic`** — sibling strategies for when the
  user is open-ended.

---
name: create-pr
description: Open a modmesh pull request that follows the project PR protocol (concise subject, clear description, "related to #xxx" wording, draft by default, ready-for-review via global comment). Use when the user asks to create, open, or draft a pull request.
---

# Create Pull Request (modmesh)

Authoritative reference is issue solvcon/modmesh#811 and the "Pull Request
Guidelines" section of `CLAUDE.md`. The rules below are a working summary;
if those extend them, follow them and flag the drift.

## Protocol the PR must satisfy

1. **Subject** -- concise and informative.
2. **Description** -- clear global description. State *what* changed and
   *why* in prose; use bullets for enumerated changes; include benchmark
   tables or verification notes when relevant.
3. **Issue reference** -- end with "Related to #xxx" or "For issue #xxx".
   **Never** use "close #xxx", "closes #xxx", "fixes #xxx", or any closing
   keyword. We do not let PR/commit text drive issue management.
4. **Draft by default** -- open as draft unless the user explicitly says it
   is ready for review. Pushing the "ready for review" button alone is
   *not* a review request in this project.
5. **Request review with a global comment** -- when the PR is ready, the
   author posts a global PR comment that explicitly asks for review. The
   button push is not quotable in follow-ups, so the comment is required.
6. **Inline annotations** -- the author is expected to add inline review
   annotations to guide the reviewer, unless the diff is one-liner-ish.
   The skill should remind the user; it does not write the annotations.
7. **Human authorship** -- all PR comments are written by humans. Tool
   assistance is OK, but the user must know what the text says. When the
   skill drafts text, present it for the user's review and edits before
   posting.

## Workflow

1. **Confirm scope with the user.** Ask directly when unclear from
   context:
   - Which issue does this PR relate to? (issue number)
   - Is this ready for review, or should it be opened as draft?
   - One-line gist of the change.

2. **Verify branch state.** modmesh's main branch is `master`. Run in parallel:
   - `git status` -- working tree must be clean (or surface uncommitted
     work and ask before continuing).
   - `git log --oneline origin/master..HEAD` -- list the commits the PR
     will carry.
   - `git diff --stat origin/master...HEAD` -- summarize the diff
     (three-dot uses the merge base).
   - `git rev-parse --abbrev-ref --symbolic-full-name @{u}` -- check if
     the branch tracks a remote.
   If the branch is not pushed, push with `-u` after confirming with the
   user. Do not force-push.

3. **Draft the subject and body.** Inspect the diff and the user's gist,
   then propose a subject and a body. The body should follow the shape of
   recent merged PRs:
   - Short opening paragraph stating the change and motivation.
   - Bulleted list of notable file/area changes when more than one.
   - Verification notes (tests run, benchmarks, manual checks) when
     non-trivial.
   - Closing line: `Related to #xxx.` or `For issue #xxx.`.

   Present the draft to the user (subject and body) and wait for edits or
   approval before opening the PR. Do not post Claude-Code attribution
   trailers in the PR body -- this project treats PR text as
   human-authored.

4. **Open the PR.** Load the approved title and body into shell
   variables using **quoted heredocs** -- the single-quoted delimiter
   suppresses every form of shell expansion, so the placeholder text
   can contain backticks, `$`, `"`, or apostrophes without escaping.
   Then pass the title as a double-quoted variable expansion (safe
   because the value is already in memory) and the body via a temp
   file:

   ```bash
   title=$(cat <<'TITLE'
   <approved subject>
   TITLE
   )

   body_file=$(mktemp)
   trap 'rm -f "$body_file"' EXIT
   cat >"$body_file" <<'BODY'
   <approved body, already ending with "Related to #xxx." or
   "For issue #xxx." from step 3>
   BODY

   gh pr create --draft \
     --title "$title" \
     --body-file "$body_file"
   ```

   Drop `--draft` only when the user has explicitly said the PR is
   ready for review. Do not append a second issue-reference footer here;
   the body from step 3 already has one. The `trap` ensures the temp
   file is removed on any exit path. If the approved body would itself
   contain the literal token `BODY` on its own line, swap the delimiter
   to something unique (e.g. `BODY_EOF_811`).

5. **After creation.** Report the PR URL back to the user. Then:
   - If the PR is draft, remind the user that "ready for review" requires
     a global comment, not just the button.
   - If the PR is ready, the global review-request comment is **required**
     by the protocol -- not optional. Draft a one-line review-request
     comment, present it for approval, and only after the user approves
     post it via `gh pr comment <num> --body-file <file>`. The comment
     must be authored by the user; if the user declines to post, output
     `blocked: review-request comment not posted` and stop.
   - If the diff is more than one-liner-ish, remind the user to add inline
     annotations to guide review. The skill does not write them.

## Guardrails

- **Closing keywords.** After loading `$title` and writing `$body_file`
  in step 4, but **before** the `gh pr create` call, scan title and
  body together with a single case-insensitive check:

  ```bash
  { printf '%s\n' "$title"; cat "$body_file"; } \
      | grep -iEn '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)[[:space:]]+#[0-9]+'
  ```

  Any hit means the text uses a banned closing keyword (close/closes/
  closed/fix/fixes/fixed/resolve/resolves/resolved followed by `#nnn`).
  Rewrite the offending line to "Related to #xxx" or "For issue #xxx"
  and re-confirm with the user before retrying.
- **Branch protection.** Never force-push, never push to `master`/`main`
  directly, never `--no-verify`. If `gh pr create` fails, surface the
  error and stop -- do not work around it.
- **No fabricated context.** Do not invent benchmark numbers, test
  results, or verification claims. Only include what the user has stated
  or what is visible in the diff/commits.
- **Trailers.** Do not append `Co-Authored-By:` or "Generated with Claude
  Code" trailers to the PR body. Commits in this project are
  human-authored by convention.

## Output

- Show the draft subject and body to the user before calling `gh pr
  create`. Use a fenced block so it is easy to edit.
- After creation, output a single line: `opened: <PR URL> (draft|ready)`.
- If guardrails block the action (closing keyword, dirty tree, unpushed
  branch), output `blocked: <reason>` and stop. Do not retry silently.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->

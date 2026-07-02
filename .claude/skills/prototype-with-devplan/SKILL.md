---
name: prototype-with-devplan
description: Draft prototype on a personal fork paired with a served development-plan doc, keeping the upstream repository clean.
---

# Prototype With Devplan (solvcon)

Take a feature or issue from idea to a reviewable draft on a separate
repository (may be a fork, public or private) paired with a development-plan
document served for review, without leaving any noise on the upstream
repository. This skill is the orchestration; it leans on `serve-docs`,
`commit-code`, and `create-pr` for their parts.

## When to use

The user wants to prototype an issue or feature privately: a draft
implementation plus a devplan doc, served on the trusted network, pushed to
their fork, and run through CI, all without touching or citing the upstream
repo. Typical phrasing: "draft implementation for issue N with a devplan and
serve the doc on <trusted network>."

## Ground rules

- **No upstream references.** Commit messages and the PR title/body for the
  draft must not cite the upstream issue or PR (`#N`, "Related to #N", PR
  links). GitHub renders them as backlinks that clutter the upstream repo.
  Describe the change on its own terms. This overrides the general CLAUDE.md
  advice to cite the related issue.
- **Personal fork only.** Push the branch to the user's fork remote and open
  the draft PR against the fork's own default (or specified) branch. Never push
  a draft branch to, or open a PR on, the upstream remote.
- **Human-authored commits.** No `Co-Authored-By` / "Generated with"
  trailers, no `feat:`/`fix:` prefixes (see `commit-code`).
- **The user may edit the branch too.** They often make their own commits on
  the branch from their editor. Before any force-push, re-check the remote and
  use `--force-with-lease`; when a rewrite is needed, preserve their commit
  messages and never drop their code (verify with `git diff <theirs> HEAD`).

## Workflow

### 1. Worktree and branch

Work in a git worktree on a topic branch, never on `master`. Use the worktree
path and branch name the user gives.

### 2. Implement the draft

Build the change. Add tests only where they earn their keep for a draft; the
user may ask to drop them later. Lint before every commit (`make lint`, or
the touched-language subset such as `make flake8 checkascii checktws`). For
the pilot GUI, exercise behavior headlessly with `QT_QPA_PLATFORM=offscreen`
before trusting it.

### 3. Write the devplan doc

Author it as a Sphinx MyST page so it renders inside the docs:

- Path: `doc/source/devplan/<name>/index.md`.
- Add `<name>/index` to the toctree in the tracked
  `doc/source/devplan/index.md` (the "Development Plans" page).
- Include explanatory figures. For a preview, an SVG schematic embedded with
  a `{figure}` directive is fine (before/after trees, dialog mockups, etc.).
- Cover: the problem, where the code lives (analysis), the design (with the
  figures), the implementation, verification, out-of-scope notes, a delivery
  status (branch, commits, CI, preview URL), and a chat-history appendix that
  records the user's prompts and what each drove.

The devplan may be included in commits. At the end of development it may be
removed or separated out.

### 4. Build and serve for review

Build the docs (`make html` in `doc/`, which runs doxygen then Sphinx) and
serve `doc/build/html` on the trusted network with a session-tied watchdog
via `doc/contrib/serve-docs.sh <trusted-ip> --launcher-pid <claude-pid>` (the
`serve-docs` skill owns this flow). The page lands at
`http://<ip>:<port>/devplan/<name>/index.html`. Report the URL early and
re-report it after every rebuild so the user can review as you work.

### 5. Commit

Clean one-concern commits (see `commit-code`). Use a new commit for each
follow-up request while iterating; reorganize into a coherent history at the
end (soft-reset to the merge base and recommit, then confirm the tree is
unchanged with `git diff <old-tip> HEAD`). Stage exact paths so each commit
stays one concern, and keep the devplan in its own commit when it is
committed.

### 6. Push to the fork and open a draft PR

Push the branch to the fork remote and track it (`git push -u <fork>
<branch>`). Sync the fork's `master` from upstream first if it is behind (`git
push <fork> master`), so the PR diff stays clean. Open a **draft** PR on the
fork to exercise CI (see `create-pr` for the body conventions, minus any
upstream reference).

### 7. CI and the skip-ci label

Let CI run on the draft and watch the jobs. Once green, add the `skip-ci` label
so later pushes do not re-run the full matrix. Do not trigger CI runs the user
has not asked for: with `skip-ci` applied, a push only hits the trivial
`check_skip` gate. When the user asks to re-exercise the CI, remove `skip-ci`,
push, wait for green, then re-apply it.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->

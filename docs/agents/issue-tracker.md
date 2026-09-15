# Issue tracker

Issues live in GitHub Issues for [`monocongo/vectorjuju`](https://github.com/monocongo/vectorjuju). Use the `gh` CLI.

When a skill says to publish to the issue tracker, create a GitHub issue.

## Wayfinding operations

- **Map:** an issue labelled `wayfinder:map`.
- **Ticket:** a sub-issue of the map, labelled `wayfinder:<type>` (`research`, `prototype`, `grilling`, `task`).
- **Blocking:** native issue dependencies, via REST:
  - read: `gh api repos/monocongo/vectorjuju/issues/<n>/dependencies/blocked_by`
  - add: `gh api -X POST repos/monocongo/vectorjuju/issues/<n>/dependencies/blocked_by -F issue_id=<blocker database id>`
- **Claim:** assign the ticket (`gh issue edit <n> -R monocongo/vectorjuju --add-assignee @me`) before any work.
- **Frontier:** open, unassigned sub-issues of the map whose blockers are all closed.

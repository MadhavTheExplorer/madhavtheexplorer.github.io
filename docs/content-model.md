# Explorer Content Model

The site records exploration in four connected forms. Each form has a distinct owner and purpose.

## Expeditions

Expeditions are repository-backed things that were built, investigated, or documented. Their core metadata comes from `.explorer/project.yml` in the repository and is collected into `_data/repositories/generated.yml` by `scripts/generate_catalog.py`.

Required repository fields:

```yaml
title: A concise project name
summary: A one-sentence account of the result
status: active # active | maintained | archived | incubating
kind: software # software | research | document | hardware | learning
fields:
  - robotics
family: robotics-and-autonomy
featured: false
started: 2026-09
links: {}
```

Generated site records add `repository` and `url`. The site-owned `_data/repositories/overrides.yml` adds optional editorial relationships such as `series`. Generated data must not contain private repositories.

## Exploration Series

Files in `_projects` are authored thematic trails, such as Controls. A series introduces a subject and gathers posts whose category matches the series `categories` value. Series are editorial content and are never generated from repositories.

## Articles

Posts are dated field notes, tutorials, and reflections. Their `categories` value assigns them to a series. Posts may declare related repositories in front matter:

```yaml
repositories:
  - MadhavTheExplorer/Robot-Arm
```

## Life

Files in `_life` and their posts contain hobbies, experiences, and non-technical exploration. They remain separate from the repository catalog.

## Publishing Rules

- A public repository appears only when it contains a valid explorer manifest.
- Repository metadata describes the artifact; site copy may add editorial context but must not contradict it.
- Collaborations retain upstream attribution and describe the explorer's specific contribution.
- Private repositories and unpublished work are excluded from generated data.
- Catalog automation runs weekly or on demand, validates manifests, builds the site, and opens an auto-merge pull request when data changes.
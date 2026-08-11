# Manuscript Workspace

This directory contains the modular LaTeX workspace for the ghost-bank
manuscript draft.

## Template Status

The current scaffold uses a temporary `article` class so that the section files
and helper scripts are usable immediately. After selecting a target journal,
update `main.tex` to use the journal template and remove any placeholder settings
that the template replaces.

## Layout

- `main.tex`: manuscript entrypoint
- `build/`: generated PDF and LaTeX build artifacts
- `sections/`: one subdirectory per main section
  - `introduction/`, `related_work/`, `method/`, `results/`, `discussion/`, `conclusion/`
- `appendix/`: appendix material (protocol, per-task tables, bank sizes, figures)
- `bib/references.bib`: bibliography
- `data/`: generated tables, figures, and report artifacts already available
- `notes/`: drafting notes (e.g. front matter)
- `scripts/`: PowerShell helper scripts for build and cleanup
- `dev/`: scratch space for manuscript development

Section organization follows `../paper_structure.md` (same directory as this
workspace). The Uniform Herding formal definition lives in
`formal_definition.md` (same directory).

## Writing Rule

Write and revise one section at a time. Every sentence must either define the
problem, sharpen the gap, state a result, delimit a claim, explain the
significance, or orient the reader.

## Build

To compile the LaTeX manuscript PDF:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

## Clean

To delete the compiler artifacts and the build folder:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean.ps1
```

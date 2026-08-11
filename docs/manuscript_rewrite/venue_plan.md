# Plan: Venue-Specific Submission Format

## Status

The content rewrite is venue-neutral. The current source uses the generic
LaTeX `article` class and must not be called NeurIPS-ready until a separate
venue-format patch is applied.

## If NeurIPS 2026 Is Selected

After the content is frozen:

- switch to the official NeurIPS 2026 style and obey its formatting rules;
- fit all main content, including figures, tables, Discussion, Limitations,
  and Conclusion, within nine content pages;
- include the references;
- place optional technical appendices after the references;
- include the mandatory NeurIPS paper checklist after the appendix;
- answer the checklist from the actual experiment and code, not from generic
  boilerplate.

The checklist and page-order requirements are submission-format requirements,
not reasons to change the paper's research framing. Do not add venue-specific
sections to the content rewrite until the venue is selected.

Official sources:

- https://neurips.cc/Conferences/2026/MainTrackHandbook
- https://neurips.cc/public/guides/PaperChecklist

## If Another Venue Is Selected

Apply that venue's official style, page limit, checklist, appendix, and
anonymity rules separately. Preserve the same content flow and evidence
boundaries unless the venue explicitly requires a different structure.


**Final paper order**
1. Title
2. Abstract
3. Introduction
4. Related Work
5. Method and Experimental Setup
6. Results
7. Discussion
8. Conclusion
9. References
10. Appendix
11. NeurIPS checklist, if that is the target venue

This matches your outline in [`docs/paper/manuscript_plan/paper_structure.md`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/docs/paper/manuscript_plan/paper_structure.md) and the NeurIPS submission order: paper, references, appendix, checklist. The checklist is a submission artifact, not a narrative section. See the official NeurIPS guidance here: [Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) and [Main Track Handbook 2026](https://neurips.cc/Conferences/2026/MainTrackHandbook).

**Writing order**
1. Method and Experimental Setup
2. Results
3. Discussion
4. Introduction
5. Related Work
6. Conclusion
7. Abstract
8. References
9. Appendix
10. Checklist

**Why this order**
- `Method` should come first because it is already mostly fixed by your formal definition and implementation.
- `Results` should come next because the claims must match the actual tables and figures.
- `Discussion` comes after results because it interprets, not invents.
- `Introduction` is easier to write once the actual findings are stable.
- `Related Work` should be written after the core story is fixed so you only cite what is actually needed.
- `Abstract` is last because it must compress the finished paper accurately.

**For your current repo**
Start with these files in this order:
1. [`sections/method/method.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/method/method.tex)
2. [`sections/results/results.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/results/results.tex)
3. [`sections/discussion/discussion.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/discussion/discussion.tex)
4. [`sections/introduction/introduction.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/introduction/introduction.tex)
5. [`sections/related_work/related_work.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/related_work/related_work.tex)
6. [`sections/conclusion/conclusion.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/sections/conclusion/conclusion.tex)
7. [`main.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/main.tex)
8. [`bib/references.bib`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/bib/references.bib)
9. [`appendix/appendix.tex`](C:/Users/Hellx/Documents/Programming/python/Project/iron/ghost-bank/manuscript/appendix/appendix.tex)


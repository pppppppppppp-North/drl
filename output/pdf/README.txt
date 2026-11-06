25FCS03 completed final report

Open 25FCS03 Full Report_completed.pdf.
Edit 25FCS03 Full Report_completed.tex. The three body figures are native editable TikZ/PGFPlots definitions in this file; only KVIS_Logo.png is external.
Author-only cleanup decisions are in editorial_revision_notes.md. Final checks are in validation_notes.md.

Build with XeLaTeX until references stabilize, normally three passes:
xelatex "25FCS03 Full Report_completed.tex"

The preamble is derived from the supplied KVIS Final Report Template.tex. It retains the school A4 geometry, 12-point Times New Roman text, 1.5 line stretch, 0.8-inch paragraph indent, centered chapter headings, and school front matter. The supplied template is included unchanged for reference. Table, URL, and native chart packages support the report content without an alternative template.

The font Path targets macOS /System/Library/Fonts/Supplemental/. On another system, update only that Path to the licensed Times New Roman font directory, or configure the corresponding installed Times New Roman font names. Fonts are not redistributed in this source package.

The original supplied PDF and school template remain unchanged in the project root.

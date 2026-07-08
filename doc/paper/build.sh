#!/bin/bash
# Compile paper.tex to PDF in a dedicated .build subfolder
# Requires: pdflatex, bibtex

cd "$(dirname "$0")"

mkdir -p .build

pdflatex -interaction=nonstopmode -output-directory=.build paper.tex

# bibtex needs to run from the .build directory
cp sn-bibliography.bib .build/
cp sn-basic.bst .build/
(cd .build && bibtex paper)

pdflatex -interaction=nonstopmode -output-directory=.build paper.tex
pdflatex -interaction=nonstopmode -output-directory=.build paper.tex

if [ -f .build/paper.pdf ]; then
    cp .build/paper.pdf .
    echo ""
    echo "Done: paper.pdf generated."
else
    echo ""
    echo "Error: paper.pdf was not generated."
    exit 1
fi

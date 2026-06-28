#!/bin/bash

cd $(dirname $0)

pandoc  metadata.yaml \
        intro.md \
        baseline_hierarchies.md \
        binary_hierarchies.md \
        sampled_hierarchies.md \
        proof.md \
        conclusions.md \
        -o ${DOC:-compiled.pdf} \
        --toc --filter pandoc-include --citeproc --lua-filter autolabel.lua 2>&1 | tail -5

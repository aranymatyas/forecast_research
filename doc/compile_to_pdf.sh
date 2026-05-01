#!/bin/bash

cd $(dirname $0)

pandoc  metadata.yaml \
        intro.md \
        baseline_hierarchies.md \
        binary_hierarchies.md \
        sampled_hierarchies.md \
        proof.md \
        conclusions.md \
        -o onlab_arany_matyas.pdf --toc
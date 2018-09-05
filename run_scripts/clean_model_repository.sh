#!/bin/bash

find $1 -mindepth $2 -type d -exec bash -c "echo -ne '{}\t'; ls '{}' | wc -l" \; | awk -F"\t" '$NF=1{print $1}'

# | xargs rm -rf

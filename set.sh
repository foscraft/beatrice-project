#!/bin/bash

# Set the path to the BeatriceVec Python script
BEATRICEVEC_SCRIPT="/path/to/beatricevec.py"

# Set the input file containing text data
INPUT_FILE="/path/to/input.txt"

# Set the output file to store the generated word embeddings
OUTPUT_FILE="/path/to/output.txt"

# Set the desired dimensionality of the word embeddings
DIMENSIONALITY=600

# Invoke BeatriceVec to generate word embeddings
python $BEATRICEVEC_SCRIPT --input $INPUT_FILE --output $OUTPUT_FILE --dimensionality $DIMENSIONALITY

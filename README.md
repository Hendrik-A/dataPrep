# dataPrep
This repository contains the python scripts to format the original arXiv dataset.

First, run tokenizeDataFile to tokenize data files val and test to single document with token counts. Then run filter-split to filter for papers that have at least 1 section head that can be mapped by DANCER and filter for papers with less than 16K tokens. Select top 5000 papers ordered by LED token descending, shuffle these and drop token columns. Write into train, test, validation files.

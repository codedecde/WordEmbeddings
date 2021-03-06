## FOLDER STRUCTURE
```
Data
|----text8 or tiny_text8
|----Linguistic_Constraints/
|    |-----ppdb_synonyms.txt
|    |-----ppdb_antonyms.txt
|    |-----wordnes_antonymx.txt
```

## Running the code
* Execute preprocess.py. Generates the following in the **Data/** directory
    * word2ix.dat: The dictionary mapping words -> integer indexes
    * unigram_table.npy: The unigram table, stored as a numpy array
* python run.py
    * Loads data using constants in constants.py
    * Saves embedding_matrix to Data/vocab_matrix_with_syn_ant.npy


## BRIEF FILE DESCRIPTIONS
* constants.py: The constants used. Contains the DEBUG Flag set to False
    * set DEBUG to false to load text8
* model.py : The model file
* utils.py : ProgressBar
* evaluate.py : Functions used for evaluating embeddings
    * cosine: For computing cosine distance
    * get_nearest_k: gets nearest k  elements
    * get_furthest_k: gets furthest k elements
* segment.py: Split words into sub-units by maximizing the probability of combinations.
	* train: train(trainfile, modelfile, threshold) takes 'trainfile' as input and return the model as a dictionary, with sub-units as keys and their probabilities as values. It will also store the model in 'modelfile', one sub-unit per line. threshold is the lower bound of the probability, under which sub-units will be ignored. 
	* load: read from 'modelfile' and return model as a dictionary.
	* segment: use DP to find the split with highest probability. It returns the log of probability and the list of sub-units.


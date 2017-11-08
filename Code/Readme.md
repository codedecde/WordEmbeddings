## PREREQUISITES
* text8 (or tiny_text8), stored in the **Data/** directory
* A folder **Data/Linguistic_Constraints/** containing
    * ppdb_synonyms.txt
    * ppdb_antonyms.txt
    * wordnet_antonyms.txt
* pytorch (latest version)


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


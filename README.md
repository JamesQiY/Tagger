# Tagger
Part-Of-Speech (POS) tagging training and prediction. A Hidden Markov Model is created with a generated probability table. The tables uses given words that are associated with a part-of-speech tag to create the necessary components of the model. 

### Description:

> Every word and punctuation symbol is understood to have a syntactic role in its sentence, such as nouns (denoting people, places or things), verbs (denoting actions), adjectives (which describe nouns) and adverbs (which describe verbs), just to name a few. Each word in a piece of text is therefore associated with a part-of-speech tag (usually assigned by hand), where the total number of tags can depend on the organization tagging the text.
We use a very simple set of POS tags: {"VERB" , "NOUN" , "PRON" , "ADJ" , "ADV" , "ADP" , "CONJ" , "DET" , "NUM" , "PRT" , "X" , "."}. The main task is to create a HMM model that can figure out a sequence of underlying states, given a sequence of observations. 

usage for the code: `python3 tagger.py -d <training file name> -t <test file name>`

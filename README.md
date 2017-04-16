This is simple text classifier library that use algorythm like next:

- use pretrained Word2Vec vectors (each word presented as word_POS, so to get part of speech - I use [pymorphy2](https://github.com/kmike/pymorphy2))
- use clusters that pre-extracted from Word2Vec vectors.
  So - many cluster will present words with similar vectors.
  E.g. "globally" in demo script I use 2000 clusters.
- to present text :
  - tokenize it
  - for each token:
    - get word2vec vector
    - get cosine similarity between vector and cluster vectors
    - set similarities to zero when it less then threshold
  - get sum of individual token vectors
- classifier training process is like next :
  - for each label :
    - choose text vectors for text with this label
    - get mean vector
    - get clusters with highter similarity
  - choose clusters
  - use only selected cluster similarity as features
  - normalize features
  - initialize KNearestNeighbours classifier
- classification process is like next :
  - convert text to vectors (sometimes pymorphy2 can't detect POS, so where multiple parsing versions)
  - for each vector
      - get nearest neighbours distance and indices
      - get labels attached to nearest  neighbours
      - if distance is bigger then threshold - set labels for this neighbour to zero
      - get mean labels value and store prediction
  - return prediction with bigger dispersion

Installation
------------
To use it you'll need previously get:
- pymorphy2 with dictionaries
- gensim
- nltk
- get word2vec model. 
  I tested it on russian and used model from [http://rusvectores.org/static/models/ruwikiruscorpora_0_300_20.bin.gz](http://rusvectores.org/static/models/ruwikiruscorpora_0_300_20.bin.gz)
  (unpack it and place in classifier/dataset/ruwikiruscorpora_0_300_20.bin)

Demo script
-----------
You can see demonstration in classifier notebook.
# Document Similarity using Word2Vec

Calculate the similarity distance between documents using pre-trained word2vec model.

### Usage

- Load a pre-trained word2vec model. _Note_: You can use [Google's pre-trained word2vec model](https://bit.ly/w2vgdrive), if you don't have one.
    
     ```python
    from gensim.models.keyedvectors import KeyedVectors
    model_path = './data/GoogleNews-vectors-negative300.bin'
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
     ```

- Once the model is loaded, it can be passed to `DocSim` class to calculate document similarities.
 
    ```python
    from DocSim import DocSim
    ds = DocSim(w2v_model)
    ```

- Calculate the similarity score between a source document & a list of target documents.

    ```python
  source_doc = 'how to delete an invoice'
  target_docs = ['delete a invoice', 'how do i remove an invoice', 'purge an invoice']

  # This will return 3 target docs with similarity score
  sim_scores = ds.calculate_similarity(source_doc, target_docs)

  print(sim_scores)
  ```
- Output is as follows:
  ```python
    [ {'score': 0.99999994, 'doc': 'delete a invoice'}, 
    {'score': 0.79869318, 'doc': 'how do i remove an invoice'}, 
    {'score': 0.71488398, 'doc': 'purge an invoice'} ]
    ```

- _Note_: You can optionally pass a `threshold` argument to the  `calculate_similarity()` method to return only the target documents with similarity score above the threshold.

    ```python
    sim_scores = ds.calculate_similarity(source_doc, target_docs, threshold=0.7)
    ```


### Requirements
- Python 3 only
- **_gensim_** : to load the word2vec model
- **_numpy_**  : to calculate similarity scores

### License
[The MIT License](./LICENSE)
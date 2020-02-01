import numpy as np
from gensim.models.doc2vec import TaggedDocument

class DocSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=None,topn=None,threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if not target_docs:
            return []
        
        if not topn:
            topn = len(target_docs)
        
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        tagged_data = [TaggedDocument(words=_d.lower(),tags=[str(i)]) for i, _d in enumerate(corpus)]
        results = []
        for i in range(0,len(tagged_data)):
            target_vec = self.vectorize(tagged_data[i].words)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({"score": sim_score,"tag":tagged_data[i].tags,"doc":tagged_data[i].words})
            # Sort results by score in desc order
            results.sort(key=lambda k: k["score"], reverse=True)

        return results[0:topn]

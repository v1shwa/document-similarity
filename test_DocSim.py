from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from DocSim import DocSim
import unittest


class DocSimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_model_path = './data/test_data.txt'
        cls.w2v_model = KeyedVectors.load_word2vec_format(test_model_path, binary=False)
        cls.stopwords = ['to', 'an', 'a']
        cls.doc_sim = DocSim(cls.w2v_model, cls.stopwords)

    def test_vectorize_with_valid_words(self):
        source_doc = 'how to delete an invoice'
        # same values dummy data will output same mean value
        expected = np.array([0.5, 0.5, 0.6, 0.3, 0.2, 0.1, 0.4, 0.6, 0.5, 0.5])
        actual = self.doc_sim.vectorize(source_doc)
        self.assertEqual(expected.all(), actual.all())


if __name__ == "__main__":
    unittest.main()

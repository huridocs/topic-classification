from app import embedder


class Classifier:

    def __init__(self):
        self.embedder = embedder.Embedder()

    def classify(self):
        # calculate UID of seqs
        # fetch embedding matrix from the cache
        #   if it exists, return it
        #   create and store embedding, return it
        # classify seq, with its embedding matrix, using a specific model
        # filter results
        # map results back to topic strings, according to classifier metadata
        # return topic confidence array
        return

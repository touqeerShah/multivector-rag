class ColQwenRetriever:
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0"):
        self.model_name = model_name
        self.ready = False

    def load(self):
        # load processor/model here later
        self.ready = True

    def embed_page_image(self, image_path: str):
        raise NotImplementedError("Implement ColQwen2 image multivector embedding")

    def embed_query(self, query: str):
        raise NotImplementedError("Implement ColQwen2 query embedding")

    def score(self, query_embedding, page_embedding):
        raise NotImplementedError("Implement late interaction / MaxSim scoring")

class MuveraProxyIndex:
    def __init__(self):
        self.rows = []

    def fit(self, rows: list[dict]):
        """
        rows:
          {
            "id": "...",
            "multivectors": [[...], [...], ...]
          }
        """
        self.rows = rows

    def encode_multivector_to_fde(self, multivectors):
        raise NotImplementedError("Plug in MUVERA FDE here")

    def search(self, query_multivectors, top_k: int = 50):
        raise NotImplementedError("Search over FDE proxy vectors here")
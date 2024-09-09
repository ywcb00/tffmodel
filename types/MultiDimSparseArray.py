import sparse

class MultiDimSparseArray(sparse.COO):
    @classmethod
    def from_numpy(x):
        return MultiDimSparseArray(x)

    def to_numpy(self):
        return self.todense()

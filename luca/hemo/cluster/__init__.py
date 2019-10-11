import umap
import numpy

from .. utils import unconcatenate


def list_umap(x_list, **kwargs):
    print("SSHAPE",[x.shape for x in x_list])
    x = numpy.concatenate(x_list, axis=0)
    embedding = umap.UMAP(**kwargs).fit_transform(x)
    embedding_list = unconcatenate(embedding, sizes=[x.shape[0] for x in x_list])
    return embedding, embedding_list


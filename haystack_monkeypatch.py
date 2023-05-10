# Monkey Patch to fix the progressbar in haystack prediction for farm-haystack[milvus]==v1.15.0

from typing import Optional, Dict, List, Union

import numpy as np

from haystack.schema import Document, FilterType
from haystack.errors import HaystackError
from haystack.document_stores import BaseDocumentStore


def query_by_embedding_batch(
    self,
    query_embs: Union[List[np.ndarray], np.ndarray],
    filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
    top_k: int = 10,
    index: Optional[str] = None,
    return_embedding: Optional[bool] = None,
    headers: Optional[Dict[str, str]] = None,
    scale_score: bool = True,
) -> List[List[Document]]:
    if isinstance(filters, list):
        if len(filters) != len(query_embs):
            raise HaystackError(
                "Number of filters does not match number of query_embs. Please provide as many filters"
                " as query_embs or a single filter that will be applied to each query_emb."
            )
    else:
        filters = [filters] * len(query_embs)
    results = []
    # NOTE(Harsh): The next 2 lines is the reason for monkey patch.
    from tqdm import tqdm
    for query_emb, filter in tqdm(zip(query_embs, filters)):
        results.append(
            self.query_by_embedding(
                query_emb=query_emb,
                filters=filter,
                top_k=top_k,
                index=index,
                return_embedding=return_embedding,
                headers=headers,
                scale_score=scale_score,
            )
        )
    return results

BaseDocumentStore.query_by_embedding_batch = query_by_embedding_batch


def retrieve_batch(
    self,
    queries: List[str],
    filters: Optional[Union[FilterType, List[Optional[FilterType]]]] = None,
    top_k: Optional[int] = None,
    index: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    batch_size: Optional[int] = None,
    scale_score: Optional[bool] = None,
    document_store: Optional[BaseDocumentStore] = None,
) -> List[List[Document]]:
    document_store = document_store or self.document_store
    if document_store is None:
        raise ValueError(
            "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
        )

    if top_k is None:
        top_k = self.top_k

    if batch_size is None:
        batch_size = self.batch_size

    if index is None:
        index = document_store.index
    if scale_score is None:
        scale_score = self.scale_score

    query_embs: List[np.ndarray] = []

    # NOTE(Harsh): The next 3 lines is the reason for monkey patch.
    from tqdm import tqdm
    maybe_tqdm = tqdm if self.progress_bar else lambda e: e
    for batch in maybe_tqdm(self._get_batches(queries=queries, batch_size=batch_size)):
        query_embs.extend(self.embed_queries(queries=batch))
    documents = document_store.query_by_embedding_batch(
        query_embs=query_embs, top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
    )

    return documents

from haystack.nodes import DensePassageRetriever
DensePassageRetriever.retrieve_batch = retrieve_batch
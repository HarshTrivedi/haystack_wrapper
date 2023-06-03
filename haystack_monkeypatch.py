# Monkey Patch to fix the progressbar in haystack prediction for farm-haystack[milvus]==v1.15.0

from tqdm.auto import tqdm
import types
import numbers
import logging
from typing import Optional, Dict, List, Union, Any

import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler

from haystack.schema import Document, FilterType
from haystack.errors import HaystackError
from haystack.document_stores import BaseDocumentStore
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.nodes import DensePassageRetriever
from haystack.modeling.visual import BUSH_SEP
from haystack.modeling.evaluation.eval import Evaluator
from haystack.utils.experiment_tracking import Tracker as tracker


logger = logging.getLogger(__name__)


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
    # NOTE(Harsh): The next 1 line is the reason for monkey patch.
    for query_emb, filter in tqdm(list(zip(query_embs, filters))):
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

    # NOTE(Harsh): The next 2 lines and the two prints below is the reason for monkey patch.
    maybe_tqdm = tqdm if self.progress_bar else lambda e: e
    document_store.query_by_embedding_batch = types.MethodType(query_by_embedding_batch, document_store)
    print("Building query vectors...")
    for batch in maybe_tqdm(self._get_batches(queries=queries, batch_size=batch_size)):
        query_embs.extend(self.embed_queries(queries=batch))
    print("Performing retrieval with query vectors...")
    documents = document_store.query_by_embedding_batch(
        query_embs=query_embs, top_k=top_k, filters=filters, index=index, headers=headers, scale_score=scale_score
    )

    return documents


def _get_predictions(self, dicts: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:

    dataset, tensor_names, _, _ = self.processor.dataset_from_dicts(
        dicts, indices=[i for i in range(len(dicts))], return_baskets=True
    )

    data_loader = NamedDataLoader(
        dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
    )
    query_embeddings_batched = []
    passage_embeddings_batched = []
    self.model.eval()

    # When running evaluations etc., we don't want a progress bar for every single query
    if len(dataset) == 1:
        disable_tqdm = True
    else:
        disable_tqdm = not self.progress_bar

    # NOTE(Harsh): The next 1 line is the reason for monkey patch.
    disable_tqdm = True
    with tqdm(
        total=len(data_loader) * self.batch_size,
        unit=" Docs",
        desc="Create embeddings",
        position=1,
        leave=False,
        disable=disable_tqdm,
    ) as progress_bar:
        for raw_batch in data_loader:
            batch = {key: raw_batch[key].to(self.devices[0]) for key in raw_batch}

            # get logits
            with torch.inference_mode():
                query_embeddings, passage_embeddings = self.model.forward(
                    query_input_ids=batch.get("query_input_ids", None),
                    query_segment_ids=batch.get("query_segment_ids", None),
                    query_attention_mask=batch.get("query_attention_mask", None),
                    passage_input_ids=batch.get("passage_input_ids", None),
                    passage_segment_ids=batch.get("passage_segment_ids", None),
                    passage_attention_mask=batch.get("passage_attention_mask", None),
                )[0]
                if query_embeddings is not None:
                    query_embeddings_batched.append(query_embeddings.cpu().numpy())
                if passage_embeddings is not None:
                    passage_embeddings_batched.append(passage_embeddings.cpu().numpy())
            progress_bar.update(self.batch_size)

    all_embeddings: Dict[str, np.ndarray] = {}
    if passage_embeddings_batched:
        all_embeddings["passages"] = np.concatenate(passage_embeddings_batched)
    if query_embeddings_batched:
        all_embeddings["query"] = np.concatenate(query_embeddings_batched)
    return all_embeddings


def monkeypatch_retriever(retriever: DensePassageRetriever):
    retriever.retrieve_batch = types.MethodType(retrieve_batch, retriever)
    retriever._get_predictions = types.MethodType(_get_predictions, retriever)


def log_results(
    results: List[Any],
    dataset_name: str,
    steps: int,
    logging: bool = True,
    print: bool = True,
    num_fold: Optional[int] = None,
):
    # Print a header
    header = "\n\n"
    header += BUSH_SEP + "\n"
    header += "***************************************************\n"
    if num_fold:
        header += (
            f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        )
    else:
        header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
    header += "***************************************************\n"
    header += BUSH_SEP + "\n"
    logger.info(header)

    for head in results:
        logger.info("\n _________ %s _________", head["task_name"])
        for metric_name, metric_val in head.items():
            # log with experiment tracking framework (e.g. Mlflow)
            if logging:
                if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                    if isinstance(metric_val, numbers.Number):
                        tracker.track_metrics(
                            metrics={f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val}, step=steps
                        )
            # print via standard python logger
            if print:
                if metric_name == "report":
                    if isinstance(metric_val, str) and len(metric_val) > 8000:
                        metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                    logger.info("%s: \n %s", metric_name, metric_val)
                else:
                    if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                        logger.info("%s: %s", metric_name, metric_val)


def monkeypatch_result_logger():
    Evaluator.log_results = staticmethod(log_results)

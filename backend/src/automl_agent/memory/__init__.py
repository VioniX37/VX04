"""Experience memory (Contribution B): learn from past runs on similar datasets.

After every run, :class:`MemoryHooks` stores the dataset's meta-features, the
candidate plans with their predicted and observed scores, the best plan and the
error->fix pairs from debugging. Before planning, :class:`MemoryRetriever`
recalls the k most similar past runs as extra retrieval-augmented-planning
knowledge, and the Operation Agent receives matching past fixes.
"""

from .hooks import MemoryHooks
from .meta_features import dataset_fingerprint, distance, meta_features
from .retriever import MemoryRetriever
from .store import ExperienceStore

__all__ = [
    "ExperienceStore",
    "MemoryHooks",
    "MemoryRetriever",
    "dataset_fingerprint",
    "distance",
    "meta_features",
]

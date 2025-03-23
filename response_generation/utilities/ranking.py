from dataclasses import dataclass
from operator import attrgetter
from typing import List, Tuple


@dataclass
class Query:
    """Representation of a query.

    It contains query ID, question, and the last turn id if available."""

    query_id: str
    question: str
    turn_leaf_id: str = None

    def get_topic_id(self) -> str:
        """Returns topic id under assumption that query ID consists of topic ID
        and turn number separated by an underscore."""
        return self.query_id.split("_")[0]

    def __str__(self):
        return self.question


@dataclass
class Document:
    """Representation of a document. It contains doc_id and optionally
    document content."""

    doc_id: str
    content: str = None


@dataclass
class ScoredDocument(Document):
    """Representation of a retrieved document. It contains doc_id and optionally
    document content and ranking score."""

    doc_id: str
    score: float = 0


class Ranking:
    def __init__(
        self, query_id: str, scored_docs: List[ScoredDocument] = None
    ) -> None:
        """Instantiates a Ranking object using the query_id and a list of scored
        documents.

        Documents are stored unordered; sorting is done when fetching them.

        Args:
            query_id: Unique id for the query.
            scored_docs: List of scored documents. Not necessarily sorted.
        """
        self._query_id = query_id
        self._scored_docs = scored_docs or []

    def __len__(self):
        return len(self._scored_docs)

    @property
    def query_id(self) -> str:
        return self._query_id

    def documents(self) -> Tuple[List[str], List[str]]:
        """Returns documents and their contents.

        Returns:
            Two parallel lists, containing document IDs and their content.
        """
        return (
            [doc.doc_id for doc in self._scored_docs],
            [doc.content for doc in self._scored_docs],
        )

    def add_doc(self, doc: ScoredDocument) -> None:
        """Adds a new document to the ranking.

        Note: it doesn't check whether the document is already present.

        Args:
            doc: A scored document.
        """
        self._scored_docs.append(doc)

    def add_docs(self, docs: List[ScoredDocument]) -> None:
        """Adds multiple documents to the ranking.

        Note: it doesn't check whether the document is already present.

        Args:
            docs: List of scored documents.
        """
        self._scored_docs.extend(docs)

    def update(self, docs: List[ScoredDocument]) -> None:
        """Adds multiple documents to the ranking uniquely.

        Args:
            docs: List of scored documents.
        """
        doc_ids = {doc.doc_id for doc in self._scored_docs}
        self._scored_docs.extend(
            [doc for doc in docs if doc.doc_id not in doc_ids]
        )

    def fetch_topk_docs(
        self, k: int = 1000, unique: bool = False
    ) -> List[ScoredDocument]:
        """Fetches the top-k docs based on their score.

            If k > len(self._scored_docs), the slicing automatically
            returns all elements in the list in sorted order.
            Returns an empty array if there are no documents in the ranking.

        Args:
            k: Number of docs to fetch.
            unique: If unique is True returns unique unique documents. In case
                of multiple documents with the same ID, returns the highest
                scoring. Defaults to False

        Returns:
            Ordered list of scored documents.
        """
        sorted_docs = sorted(self._scored_docs, key=attrgetter("score"))
        if unique:
            sorted_unique_docs = {doc.doc_id: doc for doc in sorted_docs}
            sorted_docs = list(sorted_unique_docs.values())

        return sorted_docs[::-1][:k]

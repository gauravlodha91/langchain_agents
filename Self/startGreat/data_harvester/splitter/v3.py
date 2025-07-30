import re
from typing import List, Optional, Union, Dict, Any, Tuple, Callable, Iterable, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class Document:
    """Document class compatible with LangChain's Document structure."""
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class TextSplitter(ABC):
    """
    Base class for all text splitters, following LangChain's interface pattern.
    """
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """Initialize the TextSplitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start/end of every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""
        pass

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = _metadatas[i].copy()
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip() if self._strip_whitespace else text
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting size limits."""
        separator_len = self._length_function(separator)
        
        docs = []
        current_doc: List[str] = []
        total = 0
        
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = _split_text_with_regex(text, self._separator, self._keep_separator)
        _separator = "" if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.
    
    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)
        
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
        cls, language: str, **kwargs: Any
    ) -> "RecursiveCharacterTextSplitter":
        """Create a RecursiveCharacterTextSplitter for a specific language."""
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, **kwargs)

    @staticmethod
    def get_separators_for_language(language: str) -> List[str]:
        """Get separators for a specific language."""
        if language == "cpp":
            return [
                "\nclass ",
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                "\n#include ",
                "\n#define ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "go":
            return [
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "java":
            return [
                "\nclass ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "kotlin":
            return [
                "\nclass ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "js":
            return [
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "ts":
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                "\nclass ",
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "php":
            return [
                "\nclass ",
                "\nfunction ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "proto":
            return [
                "\nmessage ",
                "\nservice ",
                "\nenum ",
                "\noption ",
                "\nimport ",
                "\nsyntax ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "python":
            return [
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "rst":
            return [
                "\n=+\n",
                "\n-+\n",
                "\n`+\n",
                "\n:+\n",
                "\n.+\n",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "ruby":
            return [
                "\ndef ",
                "\nclass ",
                "\nif ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "rust":
            return [
                "\nfn ",
                "\nconst ",
                "\nlet ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "scala":
            return [
                "\nclass ",
                "\nobject ",
                "\ndef ",
                "\nval ",
                "\nvar ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "swift":
            return [
                "\nfunc ",
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "markdown":
            return [
                "\n#{1,6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "latex":
            return [
                "\n\\chapter{",
                "\n\\section{",
                "\n\\subsection{",
                "\n\\subsubsection{",
                "\n\\begin{",
                "\n\\end{",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "html":
            return [
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "solidity":
            return [
                "\npragma ",
                "\nusing ",
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == "coq":
            return [
                "\nTheorem ",
                "\nLemma ",
                "\nProof ",
                "\nQed ",
                "\nInductive ",
                "\nFixpoint ",
                "\nDefinition ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        else:
            return ["\n\n", "\n", " ", ""]


class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using tiktoken package."""

    def __init__(
        self,
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], set[str]] = set(),
        disallowed_special: Union[Literal["all"], set[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for TokenTextSplitter.")
        
        try:
            if model_name is not None:
                enc = tiktoken.encoding_for_model(model_name)
            else:
                enc = tiktoken.get_encoding(encoding_name)
        except KeyError:
            raise ValueError(
                f"Could not find model {model_name} or encoding {encoding_name}. "
                "Please check that you have the right model/encoding name."
            )
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenized_text = _encode(text)
        chunks = []
        for i in range(0, len(tokenized_text), self._chunk_size - self._chunk_overlap):
            chunk_tokens = tokenized_text[i : i + self._chunk_size]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks


class SentenceTransformersTokenTextSplitter(TextSplitter):
    """Splitting text to tokens using sentence-transformers package."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        tokens_per_chunk: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_overlap=chunk_overlap, **kwargs)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required.")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package not found.")

        self._model = SentenceTransformer(model_name)
        self._tokenizer = self._model.tokenizer
        self.maximum_tokens_per_chunk = tokens_per_chunk or self._chunk_size

    def split_text(self, text: str) -> List[str]:
        def encode_strip_start_and_stop_token_ids(text: str) -> List[int]:
            return self._tokenizer.encode(text, add_special_tokens=False)

        tokenized_text = encode_strip_start_and_stop_token_ids(text)
        chunks = []
        
        for i in range(0, len(tokenized_text), self.maximum_tokens_per_chunk - self._chunk_overlap):
            chunk_tokens = tokenized_text[i : i + self.maximum_tokens_per_chunk]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks


class SpacyTextSplitter(TextSplitter):
    """Splitting text using Spacy package."""

    def __init__(
        self, separator: str = "\n\n", pipeline: str = "en_core_web_sm", **kwargs: Any
    ) -> None:
        """Initialize the spacy text splitter."""
        super().__init__(**kwargs)
        if not SPACY_AVAILABLE:
            raise ImportError("Spacy is not available. Please install it.")
        
        try:
            import spacy
        except ImportError:
            raise ImportError("spacy package not found.")
            
        try:
            self._tokenizer = spacy.load(pipeline)
        except OSError:
            raise OSError(
                f"Could not load spacy model {pipeline}. "
                f"Please install it with `python -m spacy download {pipeline}`"
            )
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = (s.text for s in self._tokenizer(text).sents)
        return self._merge_splits(splits, self._separator)


class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_line: bool = False,
        strip_headers: bool = True,
    ):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
        """
        if not all(
            len(header) == 2 and isinstance(header[0], str) and isinstance(header[1], str)
            for header in headers_to_split_on
        ):
            raise ValueError(
                "Expected each header to be a 2-tuple of strings, got "
                f"{headers_to_split_on}"
            )
        # Output line-by-line or aggregated into chunks
        self.return_each_line = return_each_line
        # Content under headers to split on
        self.headers_to_split_on = headers_to_split_on
        # Include headers in the content
        self.strip_headers = strip_headers

    def aggregate_lines_to_chunks(self, lines: List[dict]) -> List[Document]:
        """Combine lines with common metadata into chunks."""
        aggregated_chunks: Dict[tuple, Dict[str, Union[str, dict]]] = {}

        for line in lines:
            metadata_tuple = tuple(
                (k, v) for k, v in line["metadata"].items() if v is not None
            )
            # If current line metadata differs from previous, start a new chunk
            if metadata_tuple not in aggregated_chunks:
                aggregated_chunks[metadata_tuple] = {
                    "content": line["content"],
                    "metadata": line["metadata"],
                }
            else:
                aggregated_chunks[metadata_tuple]["content"] += "  \n" + line["content"]

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks.values()
        ]

    def split_text(self, text: str) -> List[Document]:
        """Split markdown file and return chunks."""
        lines = text.split("\n")
        lines_with_metadata: List[dict] = []
        current_content: str = ""
        current_metadata: Dict[str, str] = {}
        header_stack: List[Tuple[str, str]] = []
        initial_metadata: Dict[str, str] = {}

        for line in lines:
            stripped_line = line.strip()

            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (
                    len(stripped_line) == len(sep)
                    or stripped_line[len(sep)] == " "
                ):
                    if current_content.strip():
                        lines_with_metadata.append(
                            {"content": current_content, "metadata": current_metadata.copy()}
                        )
                        current_content = ""

                    # Handle nested headers
                    current_header_level = sep.count("#")
                    header_stack = [
                        (s, n) for s, n in header_stack
                        if s.count("#") < current_header_level
                    ]
                    
                    header_content = stripped_line[len(sep) :].strip()
                    if not self.strip_headers:
                        current_content += line + "\n"
                    
                    header_stack.append((sep, header_content))
                    
                    # Update metadata
                    current_metadata = initial_metadata.copy()
                    for s, n in header_stack:
                        current_metadata[name] = n
                    break
            else:
                current_content += line + "\n"

        if current_content.strip():
            lines_with_metadata.append(
                {"content": current_content, "metadata": current_metadata}
            )

        if self.return_each_line:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in lines_with_metadata
            ]
        else:
            return self.aggregate_lines_to_chunks(lines_with_metadata)


class HTMLHeaderTextSplitter:
    """Splitting HTML based on headers."""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_element: bool = False,
    ):
        """Create a new HTMLHeaderTextSplitter.

        Args:
            headers_to_split_on: list of tuples of headers we want to track mapped to (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4, h5, h6
            return_each_element: Return each element w/ associated headers
        """
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required for HTMLHeaderTextSplitter.")
        
        # Output element-by-element or aggregated into chunks
        self.return_each_element = return_each_element
        # Content under headers to split on
        self.headers_to_split_on = dict(headers_to_split_on)

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text and return chunks."""
        soup = BeautifulSoup(text, "html.parser")
        
        # Find all elements
        elements = soup.find_all(True)
        
        chunks: List[Document] = []
        current_chunk_content = ""
        current_headers: Dict[str, str] = {}
        
        for element in elements:
            if element.name in self.headers_to_split_on:
                # Start new chunk if we have content
                if current_chunk_content.strip():
                    chunks.append(
                        Document(
                            page_content=current_chunk_content.strip(),
                            metadata=current_headers.copy()
                        )
                    )
                    current_chunk_content = ""
                
                # Update headers
                header_name = self.headers_to_split_on[element.name]
                current_headers[header_name] = element.get_text().strip()
                
                # Reset headers of same or lower level
                header_level = int(element.name[1])
                keys_to_remove = [
                    k for k, v in self.headers_to_split_on.items() 
                    if k.startswith('h') and int(k[1]) >= header_level and k != element.name
                ]
                for key in keys_to_remove:
                    if self.headers_to_split_on[key] in current_headers:
                        del current_headers[self.headers_to_split_on[key]]
                        
                current_chunk_content += element.get_text() + "\n"
            else:
                current_chunk_content += element.get_text() + "\n"
        
        # Add final chunk
        if current_chunk_content.strip():
            chunks.append(
                Document(
                    page_content=current_chunk_content.strip(),
                    metadata=current_headers.copy()
                )
            )
        
        return chunks


class PDFPlumberTextSplitter(TextSplitter):
    """Split PDF text using pdfplumber."""
    
    def __init__(self, split_by: str = "page", **kwargs: Any) -> None:
        """Initialize PDFPlumberTextSplitter."""
        super().__init__(**kwargs)
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 and PyMuPDF are required for PDF splitting.")
        self.split_by = split_by

    def split_text(self, text: str) -> List[str]:
        """This method is not used for PDF splitting."""
        raise NotImplementedError("Use split_pdf method for PDF files.")

    def split_pdf(self, pdf_path: str) -> List[Document]:
        """Split PDF file into chunks."""
        import fitz
        
        doc = fitz.open(pdf_path)
        documents = []
        
        for page_num in range(doc.page_count):
            page = doc.page(page_num)
            text = page.get_text()
            
            if self.split_by == "page":
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num + 1, "source": pdf_path}
                    )
                )
            elif self.split_by == "paragraph":
                paragraphs = text.split('\n\n')
                for para_num, para in enumerate(paragraphs):
                    if para.strip():
                        documents.append(
                            Document(
                                page_content=para.strip(),
                                metadata={
                                    "page": page_num + 1,
                                    "paragraph": para_num + 1,
                                    "source": pdf_path
                                }
                            )
                        )
        
        doc.close()
        return documents


# Utility functions
def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """Split text with regex."""
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


# Logger for warnings
import logging
logger = logging.getLogger(__name__)


# Example usage and demonstration
if __name__ == "__main__":
    # Sample texts for demonstration
    sample_text = """
    # Introduction to Natural Language Processing
    
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language.
    
    ## Key Components
    
    ### Speech Recognition
    The goal is a computer capable of understanding the contents of documents, including the contextual nuances 
    of the language within them.
    
    ### Natural Language Understanding
    The technology can then accurately extract information and insights contained in the documents as well as 
    categorize and organize the documents themselves.
    
    ### Natural Language Generation
    Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
    and natural language generation.
    """
    
    html_sample = """
    <html>
    <body>
    <h1>Machine Learning Basics</h1>
    <p>Machine learning is a subset of artificial intelligence.</p>
    <h2>Supervised Learning</h2>
    <p>Uses labeled training data to learn a mapping function.</p>
    <h2>Unsupervised Learning</h2>
    <p>Finds hidden patterns in data without labeled examples.</p>
    </body>
    </html>
    """
    
    python_code = """
    class DataProcessor:
        def __init__(self, data):
            self.data = data
        
        def clean_data(self):
            # Remove null values
            return self.data.dropna()
        
        def transform_data(self):
            # Apply transformations
            return self.data.apply(lambda x: x.upper())
    
    def main():
        processor = DataProcessor(data)
        cleaned = processor.clean_data()
        transformed = processor.transform_data()
        return transformed
    """
    
    print("LangChain-Compatible Text Splitter Examples")
    print("=" * 60)
    
    # Example 1: Character Text Splitter
    print("\n1. CHARACTER TEXT SPLITTER:")
    char_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="\n\n"
    )
    chunks = char_splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    # Example 2: Recursive Character Text Splitter
    print("\n2. RECURSIVE CHARACTER TEXT SPLITTER:")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = recursive_splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    # Example 3: Language-specific splitting
    print("\n3. LANGUAGE-SPECIFIC SPLITTER (Python):")
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python",
        chunk_size=200,
        chunk_overlap=30
    )
    chunks = python_splitter.split_text(python_code)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    # Example 4: Markdown Header Text Splitter
    print("\n4. MARKDOWN HEADER TEXT SPLITTER:")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = markdown_splitter.split_text(sample_text)
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
    
    # Example 5: HTML Header Text Splitter
    if BS4_AVAILABLE:
        print("\n5. HTML HEADER TEXT SPLITTER:")
        html_headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=html_headers_to_split_on)
        docs = html_splitter.split_text(html_sample)
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
    
    # Example 6: Token Text Splitter
    if TIKTOKEN_AVAILABLE:
        print("\n6. TOKEN TEXT SPLITTER:")
        token_splitter = TokenTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            encoding_name="gpt2"
        )
        chunks = token_splitter.split_text(sample_text)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    # Example 7: Spacy Text Splitter
    if SPACY_AVAILABLE:
        try:
            print("\n7. SPACY TEXT SPLITTER:")
            spacy_splitter = SpacyTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separator=" "
            )
            chunks = spacy_splitter.split_text(sample_text)
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}: {chunk.strip()[:100]}...")
        except OSError as e:
            print(f"Spacy model not available: {e}")
    
    # Example 8: Document creation and splitting
    print("\n8. DOCUMENT CREATION AND SPLITTING:")
    texts = [sample_text, python_code]
    metadatas = [{"source": "nlp_intro.md"}, {"source": "data_processor.py"}]
    
    documents = recursive_splitter.create_documents(texts, metadatas=metadatas)
    print(f"Created {len(documents)} documents")
    for i, doc in enumerate(documents[:3], 1):  # Show first 3
        print(f"Document {i}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
    
    # Example 9: Splitting existing documents
    print("\n9. SPLITTING EXISTING DOCUMENTS:")
    original_docs = [
        Document(page_content=sample_text, metadata={"source": "original.md"}),
        Document(page_content=python_code, metadata={"source": "code.py"})
    ]
    
    split_docs = recursive_splitter.split_documents(original_docs)
    print(f"Split into {len(split_docs)} documents")
    for i, doc in enumerate(split_docs[:3], 1):  # Show first 3
        print(f"Split Document {i}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
    
    print("\n" + "=" * 60)
    print("Available splitter classes:")
    splitter_classes = [
        "CharacterTextSplitter",
        "RecursiveCharacterTextSplitter", 
        "TokenTextSplitter",
        "SpacyTextSplitter",
        "MarkdownHeaderTextSplitter",
        "HTMLHeaderTextSplitter",
        "PDFPlumberTextSplitter",
        "SentenceTransformersTokenTextSplitter"
    ]
    
    for splitter_class in splitter_classes:
        print(f"- {splitter_class}")
    
    print("\nLanguages supported by RecursiveCharacterTextSplitter.from_language():")
    supported_languages = [
        "cpp", "go", "java", "kotlin", "js", "ts", "php", "proto", 
        "python", "rst", "ruby", "rust", "scala", "swift", "markdown", 
        "latex", "html", "solidity", "coq"
    ]
    for lang in supported_languages:
        print(f"- {lang}")
    
    print(f"\nOptional dependencies status:")
    print(f"- tiktoken: {'✓' if TIKTOKEN_AVAILABLE else '✗'}")
    print(f"- spacy: {'✓' if SPACY_AVAILABLE else '✗'}")
    print(f"- sentence-transformers: {'✓' if SENTENCE_TRANSFORMERS_AVAILABLE else '✗'}")
    print(f"- PDF libraries: {'✓' if PDF_AVAILABLE else '✗'}")
    print(f"- BeautifulSoup4: {'✓' if BS4_AVAILABLE else '✗'}")
    print(f"- markdown: {'✓' if MARKDOWN_AVAILABLE else '✗'}")
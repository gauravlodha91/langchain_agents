import re
from typing import List, Optional, Union, Dict, Any, Tuple
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


@dataclass
class ChunkMetadata:
    """Metadata for text chunks"""
    chunk_id: int
    start_index: int
    end_index: int
    chunk_type: str
    language: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    page_number: Optional[int] = None


class TextSplitter:
    """
    A comprehensive text splitter class that supports multiple chunking algorithms
    while maintaining consistent function signatures.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 length_function: callable = len, keep_separator: bool = False):
        """
        Initialize the TextSplitter with default parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters/tokens to overlap between chunks
            length_function: Function to calculate length (default: len for characters)
            keep_separator: Whether to keep separators in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
        
        # Language-specific separators
        self.language_separators = {
            'python': ['\nclass ', '\ndef ', '\n#', '\n\n', '\n', ' ', ''],
            'javascript': ['\nfunction ', '\nconst ', '\nlet ', '\nvar ', '\n\n', '\n', ' ', ''],
            'java': ['\nclass ', '\npublic ', '\nprivate ', '\nprotected ', '\n\n', '\n', ' ', ''],
            'cpp': ['\nclass ', '\nvoid ', '\nint ', '\n#include', '\n\n', '\n', ' ', ''],
            'html': ['<html', '<head', '<body', '<div', '<p', '<h1', '<h2', '<h3', '\n\n', '\n', ' ', ''],
            'markdown': ['\n# ', '\n## ', '\n### ', '\n#### ', '\n\n', '\n', ' ', ''],
        }
    
    def character_text_splitter(self, text: str, separator: str = "\n\n") -> List[str]:
        """
        Basic character-based text splitter.
        
        Args:
            text: Input text to split
            separator: Character(s) to split on
            
        Returns:
            List of text chunks
        """
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        return self._merge_splits(splits, separator)
    
    def recursive_character_text_splitter(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """
        Recursively split text using a hierarchy of separators.
        
        Args:
            text: Input text to split
            separators: List of separators in order of preference
            
        Returns:
            List of text chunks
        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
            
        def _split_text_recursive(text: str, separators: List[str]) -> List[str]:
            if not separators:
                return self._split_by_length(text, self.chunk_size)
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            if separator == "":
                return self._split_by_length(text, self.chunk_size)
            
            splits = text.split(separator)
            good_splits = []
            
            for split in splits:
                if self.length_function(split) < self.chunk_size:
                    good_splits.append(split)
                else:
                    if good_splits:
                        merged_text = self._merge_splits(good_splits, separator)
                        good_splits = []
                        good_splits.extend(merged_text)
                    
                    other_info = _split_text_recursive(split, remaining_separators)
                    good_splits.extend(other_info)
            
            merged_splits = self._merge_splits(good_splits, separator)
            return merged_splits
        
        return _split_text_recursive(text, separators)
    
    def markdown_header_text_splitter(self, text: str, headers_to_split_on: Optional[List[Tuple[str, str]]] = None) -> List[str]:
        """
        Split markdown text based on headers.
        
        Args:
            text: Markdown text to split
            headers_to_split_on: List of (header_name, header_level) tuples
            
        Returns:
            List of text chunks with header context
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]
        
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_headers = {}
        
        for line in lines:
            # Check if line is a header
            is_header = False
            for header_marker, header_name in headers_to_split_on:
                if line.strip().startswith(header_marker + " "):
                    # Start new chunk if current chunk exists
                    if current_chunk.strip():
                        chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
                        current_chunk = ""
                    
                    # Update headers
                    header_level = len(header_marker)
                    header_text = line.strip()[len(header_marker):].strip()
                    current_headers[f"Header {header_level}"] = header_text
                    
                    # Remove headers of lower levels
                    keys_to_remove = [k for k in current_headers.keys() 
                                    if k.startswith("Header ") and int(k.split()[1]) > header_level]
                    for key in keys_to_remove:
                        del current_headers[key]
                    
                    current_chunk += line + "\n"
                    is_header = True
                    break
            
            if not is_header:
                current_chunk += line + "\n"
                
                # Check if chunk is getting too large
                if self.length_function(current_chunk) > self.chunk_size:
                    chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
        
        return chunks
    
    def html_header_text_splitter(self, html_text: str, headers_to_split_on: Optional[List[str]] = None) -> List[str]:
        """
        Split HTML text based on header tags.
        
        Args:
            html_text: HTML text to split
            headers_to_split_on: List of header tags to split on
            
        Returns:
            List of text chunks
        """
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required for HTML splitting. Install with: pip install beautifulsoup4")
        
        if headers_to_split_on is None:
            headers_to_split_on = ["h1", "h2", "h3", "h4", "h5", "h6"]
        
        soup = BeautifulSoup(html_text, 'html.parser')
        chunks = []
        current_chunk = ""
        current_headers = {}
        
        def process_element(element, depth=0):
            nonlocal current_chunk, current_headers, chunks
            
            if element.name in headers_to_split_on:
                # Start new chunk if current chunk exists
                if current_chunk.strip():
                    chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
                    current_chunk = ""
                
                # Update headers
                header_level = int(element.name[1])  # Extract number from h1, h2, etc.
                current_headers[f"Header {header_level}"] = element.get_text().strip()
                
                # Remove headers of lower levels
                keys_to_remove = [k for k in current_headers.keys() 
                                if k.startswith("Header ") and int(k.split()[1]) > header_level]
                for key in keys_to_remove:
                    del current_headers[key]
            
            if element.name:
                current_chunk += str(element) + "\n"
            else:
                current_chunk += str(element)
                
            # Check if chunk is getting too large
            if self.length_function(current_chunk) > self.chunk_size:
                chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
                current_chunk = ""
        
        for element in soup.descendants:
            if element.name in headers_to_split_on or (element.name is None and element.strip()):
                process_element(element)
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(self._format_chunk_with_headers(current_chunk, current_headers))
        
        return chunks
    
    def language_aware_splitter(self, text: str, language: str) -> List[str]:
        """
        Split text using language-specific separators.
        
        Args:
            text: Input text to split
            language: Programming language or text type
            
        Returns:
            List of text chunks
        """
        separators = self.language_separators.get(language.lower(), ["\n\n", "\n", " ", ""])
        return self.recursive_character_text_splitter(text, separators)
    
    def spacy_text_splitter(self, text: str, model_name: str = "en_core_web_sm", 
                           split_on: str = "sentence") -> List[str]:
        """
        Split text using spaCy NLP pipeline.
        
        Args:
            text: Input text to split
            model_name: spaCy model name
            split_on: What to split on ('sentence', 'paragraph', 'token')
            
        Returns:
            List of text chunks
        """
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required. Install with: pip install spacy")
        
        try:
            nlp = spacy.load(model_name)
        except OSError:
            raise OSError(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
        
        doc = nlp(text)
        
        if split_on == "sentence":
            splits = [sent.text for sent in doc.sents]
        elif split_on == "paragraph":
            # Split by double newlines, then process with spaCy
            paragraphs = text.split('\n\n')
            splits = []
            for para in paragraphs:
                if para.strip():
                    splits.append(para.strip())
        elif split_on == "token":
            splits = [token.text for token in doc if not token.is_space]
        else:
            raise ValueError("split_on must be 'sentence', 'paragraph', or 'token'")
        
        return self._merge_splits(splits, " ")
    
    def token_text_splitter(self, text: str, encoding_name: str = "gpt2", 
                           model_name: Optional[str] = None) -> List[str]:
        """
        Split text based on token count using tiktoken.
        
        Args:
            text: Input text to split
            encoding_name: Encoding name for tiktoken
            model_name: Model name (overrides encoding_name if provided)
            
        Returns:
            List of text chunks
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required. Install with: pip install tiktoken")
        
        try:
            if model_name:
                encoding = tiktoken.encoding_for_model(model_name)
            else:
                encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        def token_len(text: str) -> int:
            return len(encoding.encode(text))
        
        # Temporarily change length function to count tokens
        original_length_function = self.length_function
        self.length_function = token_len
        
        # Use recursive splitter with token counting
        chunks = self.recursive_character_text_splitter(text)
        
        # Restore original length function
        self.length_function = original_length_function
        
        return chunks
    
    def sentence_transformers_chunker(self, text: str, model_name: str = "all-MiniLM-L6-v2",
                                     similarity_threshold: float = 0.7) -> List[str]:
        """
        Split text based on semantic similarity using sentence transformers.
        
        Args:
            text: Input text to split
            model_name: Sentence transformer model name
            similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List of text chunks
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        model = SentenceTransformer(model_name)
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = model.similarity(current_embedding.reshape(1, -1), 
                                        embeddings[i].reshape(1, -1))[0][0]
            
            chunk_text = " ".join(current_chunk + [sentences[i]])
            
            if (similarity >= similarity_threshold and 
                self.length_function(chunk_text) <= self.chunk_size):
                current_chunk.append(sentences[i])
                # Update embedding to be average of chunk
                current_embedding = embeddings[max(0, i-len(current_chunk)):i+1].mean(axis=0)
            else:
                # Start new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def pdf_specific_splitter(self, pdf_path: str, split_by: str = "page") -> List[str]:
        """
        Split PDF content with PDF-specific handling.
        
        Args:
            pdf_path: Path to PDF file
            split_by: How to split ('page', 'paragraph', 'sentence')
            
        Returns:
            List of text chunks with metadata
        """
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 and PyMuPDF are required. Install with: pip install PyPDF2 PyMuPDF")
        
        chunks = []
        
        # Use PyMuPDF for better text extraction
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc.page(page_num)
            text = page.get_text()
            
            if split_by == "page":
                chunks.append(f"[Page {page_num + 1}]\n{text}")
            elif split_by == "paragraph":
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        chunks.append(f"[Page {page_num + 1}] {para.strip()}")
            elif split_by == "sentence":
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sent in sentences:
                    if sent.strip():
                        chunks.append(f"[Page {page_num + 1}] {sent.strip()}")
        
        doc.close()
        
        # Apply size limits
        final_chunks = []
        for chunk in chunks:
            if self.length_function(chunk) > self.chunk_size:
                sub_chunks = self.recursive_character_text_splitter(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def custom_multimodal_chunker(self, content: Dict[str, Any], 
                                 modality_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Split multimodal content (text, images, tables, etc.).
        
        Args:
            content: Dictionary containing different modalities
            modality_weights: Weights for different modalities when calculating chunk size
            
        Returns:
            List of multimodal chunks
        """
        if modality_weights is None:
            modality_weights = {
                'text': 1.0,
                'image': 0.1,  # Images count less towards chunk size
                'table': 0.5,
                'code': 1.0,
                'metadata': 0.1
            }
        
        chunks = []
        current_chunk = {}
        current_size = 0
        
        def calculate_chunk_size(chunk_data: Dict[str, Any]) -> int:
            total_size = 0
            for modality, data in chunk_data.items():
                weight = modality_weights.get(modality, 1.0)
                if isinstance(data, str):
                    total_size += len(data) * weight
                elif isinstance(data, list):
                    total_size += len(str(data)) * weight
                elif isinstance(data, dict):
                    total_size += len(str(data)) * weight
                else:
                    total_size += len(str(data)) * weight
            return int(total_size)
        
        for modality, data in content.items():
            if modality == 'text' and isinstance(data, str):
                # Split text into smaller parts
                text_chunks = self.recursive_character_text_splitter(data)
                for text_chunk in text_chunks:
                    temp_chunk = current_chunk.copy()
                    temp_chunk['text'] = temp_chunk.get('text', '') + text_chunk
                    
                    if calculate_chunk_size(temp_chunk) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = {'text': text_chunk}
                    else:
                        current_chunk = temp_chunk
            else:
                # Handle other modalities
                temp_chunk = current_chunk.copy()
                temp_chunk[modality] = data
                
                if calculate_chunk_size(temp_chunk) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = {modality: data}
                else:
                    current_chunk = temp_chunk
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    # Original methods (keeping for backward compatibility)
    def recursive_chunk_splitter(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        """Alias for recursive_character_text_splitter"""
        return self.recursive_character_text_splitter(text, separators)
    
    def tiktoken_splitter(self, text: str, model: str = "gpt-3.5-turbo") -> List[str]:
        """Alias for token_text_splitter"""
        return self.token_text_splitter(text, model_name=model)
    
    def sentence_splitter(self, text: str) -> List[str]:
        """Split text by sentences, respecting chunk size limits."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return self._merge_splits(sentences, " ")
    
    def paragraph_splitter(self, text: str) -> List[str]:
        """Split text by paragraphs, respecting chunk size limits."""
        paragraphs = text.split('\n\n')
        return self._merge_splits(paragraphs, "\n\n")
    
    def fixed_length_splitter(self, text: str) -> List[str]:
        """Split text into fixed-length chunks with overlap."""
        return self._split_by_length(text, self.chunk_size)
    
    def word_splitter(self, text: str) -> List[str]:
        """Split text by words, respecting chunk size limits."""
        words = text.split()
        return self._merge_splits(words, " ")
    
    def semantic_splitter(self, text: str, similarity_threshold: float = 0.8) -> List[str]:
        """Alias for sentence_transformers_chunker"""
        try:
            return self.sentence_transformers_chunker(text, similarity_threshold=similarity_threshold)
        except ImportError:
            # Fallback to sentence splitter
            return self.sentence_splitter(text)
    
    # Helper methods
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting size limits."""
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if not split:
                continue
                
            potential_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if self.length_function(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Handle overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + (separator if overlap_text else "") + split
                else:
                    current_chunk = split
                
                # If single split is still too long, split it further
                if self.length_function(current_chunk) > self.chunk_size:
                    long_chunks = self._split_by_length(current_chunk, self.chunk_size)
                    chunks.extend(long_chunks[:-1])
                    current_chunk = long_chunks[-1] if long_chunks else ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_length(self, text: str, length: int) -> List[str]:
        """Split text into chunks of specified length with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + length
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters from text."""
        if overlap_size <= 0:
            return ""
        return text[-overlap_size:] if len(text) > overlap_size else text
    
    def _format_chunk_with_headers(self, chunk: str, headers: Dict[str, str]) -> str:
        """Format chunk with header context."""
        if not headers:
            return chunk.strip()
        
        header_context = "\n".join(f"{k}: {v}" for k, v in headers.items())
        return f"[CONTEXT]\n{header_context}\n[CONTENT]\n{chunk.strip()}"
    
    def get_available_methods(self) -> List[str]:
        """Get list of available chunking methods."""
        methods = []
        for attr_name in dir(self):
            if (attr_name.endswith('_splitter') or attr_name.endswith('_chunker')) and not attr_name.startswith('_'):
                methods.append(attr_name)
        return methods
    
    def split_text(self, text: str, method: str = "recursive_character_text_splitter", **kwargs) -> List[str]:
        """
        Generic method to split text using specified algorithm.
        
        Args:
            text: Input text to split
            method: Name of the splitting method to use
            **kwargs: Additional arguments for specific methods
            
        Returns:
            List of text chunks
        """
        if not hasattr(self, method):
            available_methods = self.get_available_methods()
            raise ValueError(f"Method '{method}' not found. Available methods: {available_methods}")
        
        method_func = getattr(self, method)
        return method_func(text, **kwargs)


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
    
    # Initialize the text splitter
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    
    print("Available splitting methods:")
    for method in splitter.get_available_methods():
        print(f"- {method}")
    
    print("\n" + "="*60)
    print("CHARACTER TEXT SPLITTER:")
    chunks = splitter.character_text_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    print("\n" + "="*60)
    print("RECURSIVE CHARACTER TEXT SPLITTER:")
    chunks = splitter.recursive_character_text_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    print("\n" + "="*60)
    print("MARKDOWN HEADER TEXT SPLITTER:")
    chunks = splitter.markdown_header_text_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk[:150]}...")
    
    if BS4_AVAILABLE:
        print("\n" + "="*60)
        print("HTML HEADER TEXT SPLITTER:")
        chunks = splitter.html_header_text_splitter(html_sample)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: {chunk[:150]}...")
    
    print("\n" + "="*60)
    print("LANGUAGE-AWARE SPLITTER (Python):")
    chunks = splitter.language_aware_splitter(python_code, "python")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    if TIKTOKEN_AVAILABLE:
        print("\n" + "="*60)
        print("TOKEN TEXT SPLITTER:")
        chunks = splitter.token_text_splitter(sample_text)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: {chunk.strip()[:100]}...")
    
    # Multimodal example
    print("\n" + "="*60)
    print("CUSTOM MULTIMODAL CHUNKER:")
    multimodal_content = {
        'text': sample_text,
        'metadata': {'source': 'example.md', 'author': 'AI Assistant'},
        'tags': ['NLP', 'AI', 'Machine Learning']
    }
    chunks = splitter.custom_multimodal_chunker(multimodal_content)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {str(chunk)[:150]}...")
    
    # Using the generic split_text method
    print("\n" + "="*60)
    print("USING GENERIC SPLIT_TEXT METHOD:")
    chunks = splitter.split_text(sample_text, method="sentence_splitter")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()[:100]}...")
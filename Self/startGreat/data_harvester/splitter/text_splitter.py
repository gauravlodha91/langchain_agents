import re
from typing import List, Optional, Union
from abc import ABC, abstractmethod

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")


class TextSplitter:
    """
    A unified text splitter class that supports multiple chunking algorithms
    while maintaining consistent function signatures.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextSplitter with default parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters/tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def recursive_chunk_splitter(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
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
            chunks = []
            current_chunk = ""
            
            for split in splits:
                if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Handle overlap
                        overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                        current_chunk = overlap_text + separator + split if overlap_text else split
                    else:
                        current_chunk = split
                    
                    # If current chunk is still too long, recursively split it
                    if len(current_chunk) > self.chunk_size:
                        sub_chunks = _split_text_recursive(current_chunk, remaining_separators)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
            
            if current_chunk:
                chunks.append(current_chunk)
                
            return chunks
        
        return _split_text_recursive(text, separators)
    
    def tiktoken_splitter(self, text: str, model: str = "gpt-3.5-turbo") -> List[str]:
        """
        Split text based on token count using tiktoken.
        
        Args:
            text: Input text to split
            model: Model name for tiktoken encoding
            
        Returns:
            List of text chunks
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for token-based splitting. Install with: pip install tiktoken")
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Apply overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
                
        return chunks
    
    def sentence_splitter(self, text: str) -> List[str]:
        """
        Split text by sentences, respecting chunk size limits.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Handle overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
                
                # If sentence is longer than chunk_size, split it further
                if len(current_chunk) > self.chunk_size:
                    long_chunks = self._split_by_length(current_chunk, self.chunk_size)
                    chunks.extend(long_chunks[:-1])
                    current_chunk = long_chunks[-1] if long_chunks else ""
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def paragraph_splitter(self, text: str) -> List[str]:
        """
        Split text by paragraphs, respecting chunk size limits.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Handle overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                else:
                    current_chunk = paragraph
                
                # If paragraph is longer than chunk_size, split it further
                if len(current_chunk) > self.chunk_size:
                    long_chunks = self.recursive_chunk_splitter(current_chunk)
                    chunks.extend(long_chunks[:-1])
                    current_chunk = long_chunks[-1] if long_chunks else ""
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def fixed_length_splitter(self, text: str) -> List[str]:
        """
        Split text into fixed-length chunks with overlap.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        return self._split_by_length(text, self.chunk_size)
    
    def word_splitter(self, text: str) -> List[str]:
        """
        Split text by words, respecting chunk size limits.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        words = text.split()
        
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Handle overlap
                    overlap_words = current_chunk.split()[-self.chunk_overlap//10:]  # Approximate word overlap
                    overlap_text = " ".join(overlap_words)
                    current_chunk = overlap_text + " " + word if overlap_text else word
                else:
                    current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def semantic_splitter(self, text: str, similarity_threshold: float = 0.8) -> List[str]:
        """
        Split text based on semantic similarity (simplified version).
        Note: This is a basic implementation. For production, consider using 
        sentence transformers or similar libraries.
        
        Args:
            text: Input text to split
            similarity_threshold: Threshold for semantic similarity
            
        Returns:
            List of text chunks
        """
        # This is a simplified semantic splitter
        # In practice, you'd use embeddings and similarity calculations
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    # Helper methods
    def _split_by_length(self, text: str, length: int) -> List[str]:
        """Split text into chunks of specified length with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + length
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Apply overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
                
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters from text."""
        return text[-overlap_size:] if len(text) > overlap_size else text
    
    def get_available_methods(self) -> List[str]:
        """Get list of available chunking methods."""
        methods = []
        for attr_name in dir(self):
            if attr_name.endswith('_splitter') and not attr_name.startswith('_'):
                methods.append(attr_name)
        return methods
    
    def split_text(self, text: str, method: str = "recursive_chunk_splitter", **kwargs) -> List[str]:
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
    # Sample text for demonstration
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language. In particular, how to program computers 
    to process and analyze large amounts of natural language data.
    
    The goal is a computer capable of understanding the contents of documents, including the contextual nuances 
    of the language within them. The technology can then accurately extract information and insights contained 
    in the documents as well as categorize and organize the documents themselves.
    
    Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
    and natural language generation.
    """
    
    # Initialize the text splitter
    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    
    # Demonstrate different splitting methods
    print("Available splitting methods:")
    for method in splitter.get_available_methods():
        print(f"- {method}")
    
    print("\n" + "="*50)
    print("RECURSIVE CHUNK SPLITTER:")
    chunks = splitter.recursive_chunk_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()}")
    
    print("\n" + "="*50)
    print("SENTENCE SPLITTER:")
    chunks = splitter.sentence_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()}")
    
    print("\n" + "="*50)
    print("WORD SPLITTER:")
    chunks = splitter.word_splitter(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()}")
    
    # Using the generic split_text method
    print("\n" + "="*50)
    print("USING GENERIC SPLIT_TEXT METHOD:")
    chunks = splitter.split_text(sample_text, method="paragraph_splitter")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk.strip()}")
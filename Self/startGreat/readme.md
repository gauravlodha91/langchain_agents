
Splitters

For long-form articles or reports: RecursiveCharacterTextSplitter balances structure + control.

For code bases: use the splitter tailored to the language (PythonCodeSplitter, etc.).

For web data or Markdown docs: header-aware splitters give cleaner topic chunks.

For multi-language or global docs: SpacyTextSplitter offers better semantic accuracy.

For custom pipelines (like yours) with varied modalities: modular chunking logic per extension (chunk_csv, chunk_pdf, chunk_png) works best.





Available splitting methods:
- character_text_splitter
- custom_multimodal_chunker
- fixed_length_splitter
- html_header_text_splitter
- language_aware_splitter
- markdown_header_text_splitter
- paragraph_splitter
- pdf_specific_splitter
- recursive_character_text_splitter
- recursive_chunk_splitter
- semantic_splitter
- sentence_splitter
- sentence_transformers_chunker
- spacy_text_splitter
- tiktoken_splitter
- token_text_splitter
- word_splitter







V4 code summary

Looking at your code output and implementation, the output structure is mostly consistent across different splitters, but there are some key differences. Let me break this down:
Where the Structure IS the Same
All splitters that inherit from TextSplitter and use the split_text_to_documents() method return List[Document] with consistent metadata structure:
python# Standard metadata structure for all basic splitters:
{
    "chunk_index": 0,
    "chunk_id": "chunk_0", 
    "total_chunks": 4,
    "start_index": 0  # if add_start_index=True
}
This applies to:

CharacterTextSplitter
RecursiveCharacterTextSplitter
TokenTextSplitter
SpacyTextSplitter
SentenceTransformersTokenTextSplitter
PDFPlumberTextSplitter (when using split_text_to_documents)

Where the Structure DIFFERS
1. MarkdownHeaderTextSplitter
Adds header-specific metadata:
python{
    'Header 1': 'Introduction to Natural Language Processing',
    'Header 3': 'Speech Recognition', 
    'chunk_index': 1,
    'chunk_id': 'chunk_1',
    'total_chunks': 4
}
2. HTMLHeaderTextSplitter
Similar header-aware metadata:
python{
    'Header 1': 'Machine Learning Basics',
    'Header 2': 'Supervised Learning',
    'chunk_index': 2, 
    'chunk_id': 'chunk_2',
    'total_chunks': 4
}
3. PDFPlumberTextSplitter
When using split_pdf() method, adds PDF-specific metadata:
python{
    "page": 1,
    "paragraph": 2,  # if split_by="paragraph"
    "source": "document.pdf",
    "chunk_index": 0,
    "chunk_id": "chunk_0", 
    "total_chunks": 10
}
The Unified Solution
Your UnifiedTextSplitter class addresses this by ensuring all methods return the same base structure:
python# All methods through UnifiedTextSplitter return:
{
    'source': 'test.txt',
    'method': 'recursive',  # Added by unified interface
    'chunk_index': 0,
    'chunk_id': 'chunk_0', 
    'total_chunks': 2,
    # Plus any method-specific fields (headers, pages, etc.)
}
Summary
The output structure is consistent at the base level (all return List[Document] with core metadata fields), but differs in additional metadata depending on the splitter type. Your unified interface maintains this consistency while preserving method-specific enhancements.
The core promise - "same interface, same output structure" - is mostly true for the base Document structure and core metadata, with intentional variations for specialized use cases (headers, pages, etc.).
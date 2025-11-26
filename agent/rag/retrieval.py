import os
import glob
from rank_bm25 import BM25Okapi

class LocalRetriever:
    def __init__(self, docs_dir="docs/"):
        self.docs_dir = docs_dir
        self.chunks = [] # Stores {id, content, source}
        self.bm25 = None
        self._build_index()

    def _build_index(self):
        tokenized_corpus = []
        files = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        for fpath in files:
            fname = os.path.basename(fpath).replace(".md", "")
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by sections (headers) or double newlines
            # Assignment suggests keeping chunks small.
            # Splitting by "##" usually works well for markdown
            raw_chunks = content.split("##")
            
            for i, raw in enumerate(raw_chunks):
                clean_text = raw.strip()
                if not clean_text: 
                    continue
                # Re-add header marker if lost, except for first chunk which might be title
                text_content = f"## {clean_text}" if i > 0 else clean_text
                
                chunk_id = f"{fname}::chunk{i}"
                self.chunks.append({
                    "id": chunk_id,
                    "content": text_content,
                    "source": fname
                })
                # Simple tokenization for BM25
                tokenized_corpus.append(text_content.lower().split())

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, k=3):
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for i in top_n:
            if scores[i] > 0: # Filter irrelevant
                results.append({
                    **self.chunks[i],
                    "score": scores[i]
                })
        return results

import os
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from tqdm import tqdm
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle

# ----------------------
# CONFIG
# ----------------------
PDF_DIR = Path("pdfs")
TEXT_DIR = Path("texts")
METADATA_DIR = Path("metadata")
BM25_DIR = Path("bm25_index")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 300
TOP_K_CHUNKS = 5
HYBRID_ALPHA = 0.5

# Create directories
TEXT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3:instruct"

# ----------------------
# CHROMA SETUP
# ----------------------
chroma_client = chromadb.PersistentClient(path="chroma_storeee")
collection = chroma_client.get_or_create_collection(
    name="legal_cases",
    metadata={"hnsw:space": "cosine"}
)

# ----------------------
# EMBEDDING MODEL
# ----------------------
print("Loading embedding model...")
embed_model = SentenceTransformer(MODEL_NAME)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

# ----------------------
# TEXT EXTRACTION
# ----------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    txt_path = TEXT_DIR / (pdf_path.stem + ".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path.read_text(encoding="utf-8", errors="ignore")

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"[extract] failed {pdf_path}: {e}")
        return ""

    text = "\n".join(text_parts)
    txt_path.write_text(text, encoding="utf-8")
    return text

# ----------------------
# PAKISTANI COURT JUDGMENT CHUNKER
# ----------------------
@dataclass
class PakistaniLegalChunk:
    """Structured chunk for Pakistani court judgments"""
    text: str
    paragraph_number: str
    section_type: str
    case_citations: List[str]
    legal_provisions: List[str]
    chunk_position: int
    metadata: Dict

class PakistaniCourtJudgmentChunker:
    """Specialized chunker for Pakistani High Court and Supreme Court judgments"""
    
    def __init__(self, 
                 target_chunk_size: int = 2000,
                 max_chunk_size: int = 3000,
                 min_chunk_size: int = 400):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Pakistani case citation patterns
        self.citation_patterns = [
            r'\b\d{4}\s+SCMR\s+\d+\b',
            r'\bPLD\s+\d{4}\s+[A-Z]+\s+\d+\b',
            r'\bAIR\s+\d{4}\s+[A-Z]+\s+\d+\b',
            r'\b\d{4}\s+[A-Z]{2,5}\s+\d+\b',
            r'\bPLJ\s+\d{4}\s+[A-Za-z.()]+\s+\d+\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:v\.|vs\.|versus)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+\(\d{4}\s+[A-Z]+\s+\d+\))?',
        ]
        
        # Legal provision patterns
        self.provision_patterns = [
            r'(?:Section|Article|Rule|Order)\s+\d+[A-Z]?(?:-[A-Z])?\s+(?:PPC|Cr\.?P\.?C\.?|CPC)',
            r'Article\s+\d+(?:\([a-z0-9]+\))?(?:\s+of\s+the\s+Constitution)?',
            r'Section\s+\d+[A-Z]?(?:\([a-z0-9]+\))?',
        ]
    
    def extract_case_metadata(self, text: str, filename: str) -> Dict:
        """Extract metadata from Pakistani court judgment header"""
        metadata = {
            "doc_id": filename,
            "court": "",
            "case_number": "",
            "parties": {"appellant": "", "respondent": ""},
            "judges": [],
            "counsel": {"appellant": [], "respondent": [], "state": []},
            "date_of_hearing": "",
            "judgment_type": ""
        }
        
        header = text[:3000]
        
        # Court name
        court_match = re.search(r'IN\s+THE\s+([A-Z\s]+COURT[A-Z\s]*)', header, re.IGNORECASE)
        if court_match:
            metadata["court"] = court_match.group(1).strip()
        
        # Case numbers
        case_patterns = [
            r'Criminal\s+Appeal\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Civil\s+Appeal\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Murder\s+Reference\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
            r'Writ\s+Petition\s+No\.?\s*(\d+)\s+of\s+(\d{4})',
        ]
        for pattern in case_patterns:
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                metadata["judgment_type"] = re.search(r'[A-Za-z\s]+', pattern).group(0).strip()
                metadata["case_number"] = f"{match.group(1)}/{match.group(2)}"
                break
        
        # Party names
        party_match = re.search(r'\(([^)]+?)\s+(?:v\.|versus)\s+([^)]+?)\)', header, re.IGNORECASE)
        if party_match:
            metadata["parties"]["appellant"] = party_match.group(1).strip()
            metadata["parties"]["respondent"] = party_match.group(2).strip()
        
        # Judge names
        judge_section = re.search(r'(?:JUDGMENT|CORAM)[:\s]+(.*?)(?:Date of hearing|JUDGMENT|$)', 
                                 header, re.DOTALL | re.IGNORECASE)
        if judge_section:
            judge_text = judge_section.group(1)
            judge_names = re.findall(
                r'(?:HON\'BLE\s+)?(?:MR\.|MS\.|MRS\.)\s+(?:JUSTICE\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
                judge_text, re.IGNORECASE
            )
            metadata["judges"] = [name.strip() for name in judge_names]
        
        # Date of hearing
        hearing_date = re.search(r'Date\s+of\s+hearing:\s*(\d{2}\.\d{2}\.\d{4})', header)
        if hearing_date:
            metadata["date_of_hearing"] = hearing_date.group(1)
        
        return metadata
    
    def extract_citations(self, text: str) -> List[str]:
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return list(set(citations))
    
    def extract_legal_provisions(self, text: str) -> List[str]:
        provisions = []
        for pattern in self.provision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            provisions.extend(matches)
        return list(set(provisions))
    
    def identify_section_type(self, text: str, para_num: str) -> str:
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['facts of the case', 'epitomized facts', 'brief facts']):
            return 'facts'
        elif any(kw in text_lower for kw in ['contended', 'submissions', 'argued', 'counsel']):
            return 'arguments'
        elif any(kw in text_lower for kw in ['analysis', 'findings', 'observed', 'perusal of record']):
            return 'analysis'
        elif any(kw in text_lower for kw in ['conclusion', 'for the reasons', 'appeal is', 'acquitted', 'convicted']):
            return 'conclusion'
        elif para_num == "" or para_num == "0":
            return 'header'
        else:
            return 'analysis'
    
    def chunk_judgment(self, text: str, doc_metadata: Dict) -> List[PakistaniLegalChunk]:
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_para_num = "0"
        current_chunk_size = 0
        chunk_position = 0
        
        in_header = True
        header_end_markers = ['JUDGMENT', 'ORDER', 'BRIEF FACTS']
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if in_header and any(marker in line.upper() for marker in header_end_markers):
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if len(chunk_text) > 100:
                        chunks.append(PakistaniLegalChunk(
                            text=chunk_text,
                            paragraph_number="0",
                            section_type='header',
                            case_citations=[],
                            legal_provisions=[],
                            chunk_position=chunk_position,
                            metadata=doc_metadata
                        ))
                        chunk_position += 1
                current_chunk = []
                current_chunk_size = 0
                in_header = False
            
            para_match = re.match(r'^(\d+)\.\s+', line)
            
            if para_match and not in_header:
                if current_chunk and current_chunk_size > self.min_chunk_size:
                    chunk_text = '\n'.join(current_chunk).strip()
                    chunks.append(PakistaniLegalChunk(
                        text=chunk_text,
                        paragraph_number=current_para_num,
                        section_type=self.identify_section_type(chunk_text, current_para_num),
                        case_citations=self.extract_citations(chunk_text),
                        legal_provisions=self.extract_legal_provisions(chunk_text),
                        chunk_position=chunk_position,
                        metadata=doc_metadata
                    ))
                    chunk_position += 1
                
                current_para_num = para_match.group(1)
                current_chunk = [line]
                current_chunk_size = len(line)
                i += 1
                
            elif current_chunk_size + len(line) > self.max_chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) > self.min_chunk_size:
                    chunks.append(PakistaniLegalChunk(
                        text=chunk_text,
                        paragraph_number=current_para_num,
                        section_type=self.identify_section_type(chunk_text, current_para_num),
                        case_citations=self.extract_citations(chunk_text),
                        legal_provisions=self.extract_legal_provisions(chunk_text),
                        chunk_position=chunk_position,
                        metadata=doc_metadata
                    ))
                    chunk_position += 1
                
                current_chunk = [line]
                current_chunk_size = len(line)
                i += 1
            else:
                current_chunk.append(line)
                current_chunk_size += len(line)
                i += 1
        
        if current_chunk and current_chunk_size > self.min_chunk_size:
            chunk_text = '\n'.join(current_chunk).strip()
            chunks.append(PakistaniLegalChunk(
                text=chunk_text,
                paragraph_number=current_para_num,
                section_type=self.identify_section_type(chunk_text, current_para_num),
                case_citations=self.extract_citations(chunk_text),
                legal_provisions=self.extract_legal_provisions(chunk_text),
                chunk_position=chunk_position,
                metadata=doc_metadata
            ))
        
        return chunks
    
    def add_contextual_metadata(self, chunks: List[PakistaniLegalChunk]) -> List[Dict]:
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            context_parts = []
            
            if chunk.metadata.get('case_number'):
                context_parts.append(f"[Case: {chunk.metadata['case_number']}]")
            
            if chunk.metadata.get('parties', {}).get('appellant'):
                parties = chunk.metadata['parties']
                context_parts.append(f"[{parties['appellant']} v. {parties['respondent']}]")
            
            if chunk.paragraph_number != "0":
                context_parts.append(f"[Paragraph {chunk.paragraph_number}]")
            
            if chunk.section_type:
                context_parts.append(f"[Section: {chunk.section_type.title()}]")
            
            context_prefix = ' '.join(context_parts)
            if context_prefix:
                context_prefix += '\n\n'
            
            citation_context = ""
            if chunk.case_citations:
                top_citations = chunk.case_citations[:3]
                citation_context = f"[Key Citations: {', '.join(top_citations)}]\n\n"
            
            prev_context = ""
            if i > 0:
                prev_text = chunks[i-1].text
                if len(prev_text) > 150:
                    prev_context = f"[Previous: ...{prev_text[-150:]}]\n\n"
            
            enhanced_text = context_prefix + citation_context + prev_context + chunk.text
            
            enhanced_chunks.append({
                'text': enhanced_text,
                'raw_text': chunk.text,
                'paragraph_number': chunk.paragraph_number,
                'section_type': chunk.section_type,
                'case_citations': chunk.case_citations,
                'legal_provisions': chunk.legal_provisions,
                'chunk_position': chunk.chunk_position,
                'total_chunks': len(chunks),
                'metadata': chunk.metadata
            })
        
        return enhanced_chunks

# ----------------------
# ENHANCED CHUNKING FUNCTION
# ----------------------
def chunk_text(text: str, metadata: Dict) -> List[Dict]:
    """
    Intelligent chunking for Pakistani court judgments.
    Falls back to basic chunking if not recognized as a judgment.
    """
    # Try Pakistani court judgment chunking first
    try:
        chunker = PakistaniCourtJudgmentChunker(
            target_chunk_size=CHUNK_SIZE_CHARS,
            max_chunk_size=3000,
            min_chunk_size=400
        )
        
        # Extract enhanced metadata from document content
        doc_metadata = chunker.extract_case_metadata(text, metadata['doc_id'])
        doc_metadata.update(metadata)
        
        # Check if this looks like a Pakistani court judgment
        is_judgment = any(marker in text[:2000].upper() for marker in 
                         ['LAHORE HIGH COURT', 'SUPREME COURT', 'SINDH HIGH COURT', 
                          'ISLAMABAD HIGH COURT', 'JUDGMENT', 'CRIMINAL APPEAL'])
        
        if is_judgment:
            chunks = chunker.chunk_judgment(text, doc_metadata)
            enhanced = chunker.add_contextual_metadata(chunks)
            
            result = []
            for i, chunk_data in enumerate(enhanced):
                result.append({
                    "chunk_id": i,
                    "text": chunk_data['text'],
                    "raw_text": chunk_data['raw_text'],
                    "doc_id": doc_metadata["doc_id"],
                    "case_no": doc_metadata.get("case_number", ""),
                    "title": f"{doc_metadata['parties']['appellant']} v {doc_metadata['parties']['respondent']}" if doc_metadata['parties']['appellant'] else "",
                    "court": doc_metadata.get("court", ""),
                    "judges": doc_metadata.get("judges", []),
                    "parties": doc_metadata.get("parties", {}),
                    "paragraph_number": chunk_data['paragraph_number'],
                    "section_type": chunk_data['section_type'],
                    "citations": chunk_data['case_citations'],
                    "provisions": chunk_data['legal_provisions'],
                })
            return result
    except Exception as e:
        print(f"[warn] Pakistani chunker failed, falling back to basic: {e}")
    
    # Fallback to basic chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    result = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": i,
            "text": chunk,
            "raw_text": chunk,
            "doc_id": metadata["doc_id"],
            "case_no": metadata.get("case_no", ""),
            "title": metadata.get("title", ""),
            "court": "",
            "judges": [],
            "parties": {},
            "paragraph_number": "",
            "section_type": "",
            "citations": [],
            "provisions": [],
        }
        result.append(chunk_data)
    return result

# ----------------------
# BM25 INDEX
# ----------------------
class BM25Index:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25 = None
        self.chunks_data = []
        self.tokenized_corpus = []
        
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def build(self, chunks: List[Dict]):
        self.chunks_data = chunks
        self.tokenized_corpus = [self.tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save()
    
    def save(self):
        with open(self.index_path / "bm25.pkl", "wb") as f:
            pickle.dump((self.bm25, self.chunks_data, self.tokenized_corpus), f)
    
    def load(self):
        index_file = self.index_path / "bm25.pkl"
        if index_file.exists():
            with open(index_file, "rb") as f:
                self.bm25, self.chunks_data, self.tokenized_corpus = pickle.load(f)
            return True
        return False
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

# ----------------------
# INDEX INTO CHROMA + BM25
# ----------------------
def index_pdfs():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Checking {len(pdf_files)} PDFs for indexing...")

    existing_ids = set()
    try:
        existing_data = collection.get(include=["metadatas"], limit=None)
        if existing_data and "metadatas" in existing_data:
            for meta in existing_data["metadatas"]:
                if meta and "doc_id" in meta:
                    existing_ids.add(meta["doc_id"])
    except Exception as e:
        print(f"[warn] Could not fetch existing docs: {e}")

    all_chunks = []
    new_docs_processed = False

    for pdf_path in tqdm(pdf_files):
        doc_id = pdf_path.stem

        if doc_id in existing_ids:
            metadata_file = METADATA_DIR / f"{doc_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    saved_chunks = json.load(f)
                    all_chunks.extend(saved_chunks)
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"Skipping {doc_id} (empty or unreadable text)")
            continue

        metadata = {"doc_id": doc_id}
        chunks = chunk_text(text, metadata)
        
        if not chunks:
            print(f"Skipping {doc_id} (no chunks generated)")
            continue

        metadata_file = METADATA_DIR / f"{doc_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        chunk_texts = [c["text"] for c in chunks]
        vectors = get_embeddings(chunk_texts)

        ids = [f"{doc_id}_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "case_no": c["case_no"],
                "title": c["title"],
                "court": c.get("court", ""),
                "judges": json.dumps(c.get("judges", [])),
                "parties": json.dumps(c.get("parties", {})),
                "paragraph_number": c.get("paragraph_number", ""),
                "section_type": c.get("section_type", ""),
                "citations": json.dumps(c.get("citations", [])),
                "provisions": json.dumps(c.get("provisions", [])),
            } 
            for c in chunks
        ]

        collection.upsert(
            documents=chunk_texts,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        
        all_chunks.extend(chunks)
        new_docs_processed = True
        
        # Enhanced logging
        case_info = f"{chunks[0].get('case_no', 'N/A')}"
        court_info = chunks[0].get('court', 'Unknown')
        print(f"Indexed {doc_id} | {len(chunks)} chunks | {court_info} | Case: {case_info}")

    print("\nBuilding BM25 keyword index...")
    bm25_index = BM25Index(BM25_DIR)
    
    if not all_chunks:
        for metadata_file in METADATA_DIR.glob("*.json"):
            with open(metadata_file, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
    
    if all_chunks:
        bm25_index.build(all_chunks)
        print(f"BM25 index built with {len(all_chunks)} chunks")
    
    print("\nIndexing completed (only new PDFs processed).")

# ----------------------
# HYBRID SEARCH
# ----------------------
def search_hybrid(query: str, top_k=TOP_K_CHUNKS, alpha=HYBRID_ALPHA):
    q_vec = get_embeddings([query])[0]
    semantic_results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    semantic_scores = {}
    for doc, meta, dist in zip(
        semantic_results["documents"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0]
    ):
        chunk_id = f"{meta['doc_id']}_{doc[:50]}"
        semantic_scores[chunk_id] = {
            "score": 1 - dist,
            "doc": doc,
            "meta": meta
        }
    
    bm25_index = BM25Index(BM25_DIR)
    if not bm25_index.load():
        print("BM25 index not found. Using semantic search only.")
        alpha = 1.0
    
    keyword_scores = {}
    if alpha < 1.0 and bm25_index.bm25:
        bm25_results = bm25_index.search(query, top_k=top_k * 2)
        
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            if max_bm25_score > 0:
                for idx, score in bm25_results:
                    chunk_data = bm25_index.chunks_data[idx]
                    chunk_id = f"{chunk_data['doc_id']}_{chunk_data['text'][:50]}"
                    keyword_scores[chunk_id] = {
                        "score": score / max_bm25_score,
                        "doc": chunk_data["text"],
                        "meta": {
                            "doc_id": chunk_data["doc_id"],
                            "case_no": chunk_data["case_no"],
                            "title": chunk_data["title"]
                        }
                    }
    
    combined_scores = {}
    all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    for chunk_id in all_chunk_ids:
        sem_score = semantic_scores.get(chunk_id, {}).get("score", 0)
        key_score = keyword_scores.get(chunk_id, {}).get("score", 0)
        
        hybrid_score = alpha * sem_score + (1 - alpha) * key_score
        
        data = semantic_scores.get(chunk_id) or keyword_scores.get(chunk_id)
        
        combined_scores[chunk_id] = {
            "score": hybrid_score,
            "semantic_score": sem_score,
            "keyword_score": key_score,
            "doc": data["doc"],
            "meta": data["meta"]
        }
    
    ranked = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    return ranked

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    print("\nStarting RAG indexing...\n")
    index_pdfs()

    print("\n" + "="*60)
    print("RAG SEARCH READY - Using Hybrid Search (Semantic + Keyword)")
    print("="*60)
    print("Enhanced for Pakistani Court Judgments")
    print("Tips:")
    print("  - Search by case name, case number, or judge name")
    print("  - Search by legal provision (Section 302 PPC)")
    print("  - Search by citation (2016 SCMR 2073)")
    print("  - Semantic search for concepts and themes\n")

    while True:
        q = input("Enter search query (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break
        
        start = time.time()
        hits = search_hybrid(q, top_k=TOP_K_CHUNKS)
        elapsed = time.time() - start
        
        print(f"\n{'='*60}")
        print(f"Top {len(hits)} results ({elapsed:.2f}s)")
        print(f"{'='*60}")
        
        for i, h in enumerate(hits, 1):
            print(f"\nResult #{i}")
            print(f"   Combined Score: {h['score']:.4f}")
            print(f"   Semantic: {h['semantic_score']:.4f} | Keyword: {h['keyword_score']:.4f}")
            print(f"   Document: {h['meta'].get('doc_id', 'N/A')}")
            print(f"   Case: {h['meta'].get('case_no', 'N/A')}")
            print(f"   Court: {h['meta'].get('court', 'N/A')}")
            
            # Parse JSON fields
            judges = h['meta'].get('judges', '[]')
            if isinstance(judges, str):
                try:
                    judges = json.loads(judges)
                except:
                    judges = []
            if judges:
                print(f"   Judges: {', '.join(judges)}")
            
            citations = h['meta'].get('citations', '[]')
            if isinstance(citations, str):
                try:
                    citations = json.loads(citations)
                except:
                    citations = []
            if citations:
                print(f"   Citations: {', '.join(citations[:3])}")
            
            provisions = h['meta'].get('provisions', '[]')
            if isinstance(provisions, str):
                try:
                    provisions = json.loads(provisions)
                except:
                    provisions = []
            if provisions:
                print(f"   Provisions: {', '.join(provisions[:3])}")
            
            print(f"   Section: {h['meta'].get('section_type', 'N/A')}")
            print(f"   Para: {h['meta'].get('paragraph_number', 'N/A')}")
            print(f"   Excerpt: {h['doc']}...")
            print(f"   {'-'*58}")
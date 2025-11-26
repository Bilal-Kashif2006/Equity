import os
import time
import requests
import re
import json
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
import pickle

# === NLP / ML ===
from transformers import pipeline
import xml.etree.ElementTree as ET
import yfinance as yf
import pdfplumber
from tqdm import tqdm
import numpy as np

# === LangChain ===
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# === RAG Components ===
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# === Set Google API Key ===
os.environ["GOOGLE_API_KEY"] = ""


# =====================================================
# RAG SYSTEM COMPONENTS
# =====================================================

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
TOP_K_CHUNKS = 10
HYBRID_ALPHA = 0.5

# Create directories
TEXT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
BM25_DIR.mkdir(parents=True, exist_ok=True)

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
    try:
        chunker = PakistaniCourtJudgmentChunker(
            target_chunk_size=CHUNK_SIZE_CHARS,
            max_chunk_size=3000,
            min_chunk_size=400
        )
        
        doc_metadata = chunker.extract_case_metadata(text, metadata['doc_id'])
        doc_metadata.update(metadata)
        
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
    """Hybrid search combining semantic and keyword-based retrieval"""
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


# =====================================================
# AGENT SYSTEM WITH INTEGRATED RAG
# =====================================================

def sentiment_tool_func(company_name: str) -> str:
    """Placeholder for sentiment analysis tool"""
    return f"Sentiment analysis for {company_name} not implemented yet."


class BossAgent:
    classification_prompt_template = """

ROLE:
 - You are a Legal AI assistant. You help users retrieve, summarize, and compare legal documents and cases. 

INSTRUCTION
    - Only answer queries with an actual legal/regulatory/compliance context.
    - Avoid making non-legal queries appear legal.
    - Consider follow-up questions: If the user query seems related to previous conversation or uses pronouns/context ("What about that case?", "And the previous judgment?"), use the chat history to understand it.
    - If the query is unrelated to law, politely ask the user for clarification.

TASK:
    - Classify input based on the given classification rules
    - Determine if the query is a follow-up and if so, keep context in memory

CLASSIFICATION RULES:
    1) If context NOT related to legal/regulatory/compliance or some non-legal query:
        EXAMPLES:
            1.1) User Query: "What's the weather in New York?"
                 AI: <answer>Sorry,I can't answer this query as it is not legal related. Please provide more context or a legal question.</answer>
            1.2) User Query: "My name is legal?"
                 AI: <answer>Sorry, I can't understand. Please provide more context so I can help.</answer>
            1.3) User Query: "make this query legal?"
                 AI: <answer>Sorry, I can't answer this query as it is not legal related. Please elaborate or provide context.</answer>

    2) If related to legal/regulatory/compliance:
       - Determine the TYPE:
           [GQ] = General legal query (includes standalone queries or follow-up questions after a previous GQ)
           [SQ] = Summarizing request (asks for a summary of a law, regulation, case, or document)
           [CQ] = Compare request (asks to compare two or more legal documents, laws, or regulations)
       - If the user query is legal-context-based but too broad, ask the user to narrow it down
       - Use memory/chat history to resolve pronouns or references in follow-up queries

EXAMPLES:
    2.1) User Query: "Retrieve past corruption cases in Pakistan"
         AI: <answer><valid>[GQ] Retrieve past corruption cases in Pakistan</valid></answer>
    2.2) User Query: "Summarize the Lahore High Court judgment on property dispute"
         AI: <answer><valid>[SQ] Summarize the Lahore High Court judgment on property dispute</valid></answer>
    2.3) User Query: "Compare GDPR with CCPA"
         AI: <answer><valid>[CQ] Compare GDPR with CCPA</valid></answer>
    2.4) User Query: "What were the penalties in last anti-corruption case?"
         AI: <answer><valid>[GQ] Find previous anti-corruption cases and penalties</valid></answer>
    2.5) User Query: "Has Bilal Kashif been involved in any property cases?"
         AI: <answer><valid>[GQ] Find property cases involving Bilal Kashif</valid></answer>
    2.6) User Query: "Summarize the judgment involving Warda Fatima in tax evasion case"
         AI: <answer><valid>[SQ] Summarize the tax evasion judgment involving Warda Fatima</valid></answer>
    2.7) User Query: "Compare corruption rulings for Bilal Kashif and Abdullah Mujeeb"
         AI: <answer><valid>[CQ] Compare corruption rulings involving Bilal Kashif and Abdullah Mujeeb</valid></answer>
    2.8) User Query: "Compare Lahore High Court and Supreme Court decisions on property disputes"
         AI: <answer><valid>[CQ] Compare Lahore High Court vs Supreme Court property dispute decisions</valid></answer>
    2.9) User Query: "Summarize legal docs"
         AI: <answer>Please specify which legal documents or cases you want summarized</answer>
    2.10) User Query: "Compare these cases"
         AI: <answer>Please provide the names or references of the cases you want to compare</answer>
    2.11) User Query: "And the previous corruption case?"
         AI: <answer><valid>[GQ] Retrieve details about the previous corruption case mentioned in chat history</valid></answer>

EXAMPLES (Follow-up / Context-aware queries):

    2.11) User Query: "And the previous corruption case?"
         AI: <answer><valid>[GQ] Retrieve details about the previous corruption case mentioned in chat history</valid></answer>

    2.12) User Query: "What was the verdict in that case?"
         AI: <answer><valid>[GQ] Find the verdict of the corruption case mentioned earlier in the chat</valid></answer>

    2.13) User Query: "Summarize it for me"
         AI: <answer><valid>[SQ] Summarize the previously discussed judgment or legal document</valid></answer>

    2.14) User Query: "Compare this with the 2020 property case"
         AI: <answer><valid>[CQ] Compare the currently discussed case with the 2020 property case</valid></answer>

    2.15) User Query: "Any penalties mentioned there?"
         AI: <answer><valid>[GQ] Retrieve penalties mentioned in the previous legal case discussed in chat history</valid></answer>

    2.16) User Query: "Who were the defendants again?"
         AI: <answer><valid>[GQ] List defendants involved in the previously discussed case</valid></answer>

    2.17) User Query: "And the Lahore High Court judgment?"
         AI: <answer><valid>[SQ] Summarize the Lahore High Court judgment mentioned in prior conversation</valid></answer>
FORMATTING RULES:
    - Always wrap the final output in <answer>...</answer>.
    - If valid, include <valid>...</valid> around the TYPE and restated query.
    - In case of a follow-up question, use memory/chat history to clarify references.
    - Keep the restated query exactly as the user asked it (fix only obvious grammar issues).
    - No extra text, no explanations outside the specified tags.

Input Query:
{query}"""



    def __init__(self, llm, db_name: str = "chroma_db_with_agent"):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.current_dir = os.getcwd()
        self.db_dir = os.path.join(self.current_dir, "db")
        os.makedirs(self.db_dir, exist_ok=True)
        self.persistent_directory = os.path.join(self.db_dir, db_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.db = None

        self.embedded_file_record = os.path.join(self.persistent_directory, "embedded_files.txt")
        os.makedirs(self.persistent_directory, exist_ok=True)

        self.compare_prompt_template = PromptTemplate.from_template(
            """
Role:
You are a Legal AI assistant specializing in Pakistani case law. You help users analyze, summarize, and compare legal cases, judgments, and statutes. Focus on reasoning, precedents, verdicts, penalties, and procedural context.

Task:
Analyze the two legal cases/documents provided and identify all similarities and differences between them. Focus on legal reasoning, outcomes, and relevant details.

Instructions:
- Treat content as similar if legal reasoning, verdicts, cited statutes, or precedents are equivalent.
- Treat content as different if reasoning, verdicts, penalties, precedents, parties, or procedural context differ.
- Include the parties involved, the court, jurisdiction, and case dates when relevant.
- Highlight subtle differences in interpretation, application, or scope of law.
- Include maximum possible similarities and differences without making assumptions.
- Always reference the specific parts of the documents when possible.

Examples:
- Similarity: Both cases applied Section 302 of the Pakistan Penal Code to determine criminal liability.
- Difference: Case 2 imposed a fine of PKR 500,000, while Case 1 imposed imprisonment of 3 years for the same offense.
- Similarity: Both judgments cited previous decisions on procedural requirements for appeals.
- Difference: Case 1 considered mitigating circumstances, whereas Case 2 did not.

Output Format:
- Divide output into two sections: SIMILARITIES and DIFFERENCES
- Each entry should include:
    1. Heading: Short title summarizing the topic (e.g., "Penalties", "Precedent", "Legal Reasoning")
    2. Explanation: Brief explanation of the similarity or difference
    3. doc1_line: Quote or reference from document 1
    4. doc2_line: Quote or reference from document 2
- Return the output directly, without introductory text or commentary.

Analyze:
- Read both documents carefully.
- Extract every relevant similarity and difference according to the above rules.
- Avoid assumptions; rely only on what is explicitly or logically supported in the documents.
- Return the structured but readable output directly.

DOCS are:
doc1  
{doc1}

doc2  
{doc2}
"""
        )

        self.summarize_prompt_template = PromptTemplate.from_template(
            """
You are a legal analyst and educator. Your task is to summarize the following legal document in **clear, simple, layman-friendly language** while preserving the **exact legal meaning and context**.  

Instructions:
- Simplify complex legal jargon without losing key obligations, rights, or compliance requirements.
- Keep the structure logical and easy to read.
- Retain any critical definitions, terms, deadlines, or requirements.
- Highlight practical implications for a layperson.
- Avoid skipping any sections with important legal content.

Output Format:
- Use numbered points or short paragraphs.
- Each point should convey one main idea.
- Include any deadlines, obligations, or conditions explicitly.

Document to summarize:
{text}                                                
"""
        )

        # === Tools with Integrated RAG ===
        self.tools = [
            Tool(
                name="HybridRAGSearch",
                func=self.hybrid_rag_search_tool,
                description=(
                    "Advanced hybrid search tool combining semantic and keyword-based retrieval. "
                    "Use this to search Pakistani legal documents, court judgments, and cases. "
                    "Can search by case name, case number, judge name, legal provisions (Section 302 PPC), "
                    "citations (2016 SCMR 2073), or general legal concepts. "
                    "Input: search query string. Output: relevant legal document excerpts with metadata."
                ),
            ),
            Tool(
                name="IndexNewDocuments",
                func=self.index_documents_tool,
                description=(
                    "Index new PDF documents into the RAG system. "
                    "Automatically extracts text, chunks documents intelligently, "
                    "and creates searchable embeddings. Use when user wants to add new documents. "
                    "Input: 'index' or 'reindex'. Output: confirmation message."
                ),
            ),
            Tool(
                name="CompareTopics",
                func=self.compare_docs,
                description="Use to compare two documents.",
            ),
            Tool(
                name="Summarizer",
                func=self.summarize_text,
                description="Use to summarize any large text.",
            ),
            Tool(
                name="CompanySentimentAnalyzer",
                func=sentiment_tool_func,
                description="Analyze the sentiment of recent news about a given company or stock with URLs and metadata. Input should be the company name or ticker symbol.",
            ),
        ]

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    # ===== RAG Tool Functions =====
    
    def hybrid_rag_search_tool(self, query: str) -> str:
        """
        Wrapper function for hybrid RAG search to be used as an agent tool.
        """
        try:
            results = search_hybrid(query, top_k=TOP_K_CHUNKS, alpha=HYBRID_ALPHA)
            
            if not results:
                return "No relevant documents found for your query. Try rephrasing or using different keywords."
            
            # Format results for agent consumption
            formatted_output = []
            formatted_output.append(f"Found {len(results)} relevant legal documents:\n")
            
            for i, hit in enumerate(results, 1):
                formatted_output.append(f"\n{'='*60}")
                formatted_output.append(f"Result #{i} (Relevance: {hit['score']:.3f})")
                formatted_output.append(f"{'='*60}")
                
                # Metadata
                meta = hit['meta']
                formatted_output.append(f"Document ID: {meta.get('doc_id', 'N/A')}")
                formatted_output.append(f"Case Number: {meta.get('case_no', 'N/A')}")
                formatted_output.append(f"Court: {meta.get('court', 'N/A')}")
                formatted_output.append(f"Title: {meta.get('title', 'N/A')}")
                
                # Parse JSON fields
                judges = meta.get('judges', '[]')
                if isinstance(judges, str):
                    try:
                        judges = json.loads(judges)
                    except:
                        judges = []
                if judges:
                    formatted_output.append(f"Judges: {', '.join(judges)}")
                
                parties = meta.get('parties', '{}')
                if isinstance(parties, str):
                    try:
                        parties = json.loads(parties)
                    except:
                        parties = {}
                if parties and parties.get('appellant'):
                    formatted_output.append(f"Parties: {parties['appellant']} v. {parties['respondent']}")
                
                citations = meta.get('citations', '[]')
                if isinstance(citations, str):
                    try:
                        citations = json.loads(citations)
                    except:
                        citations = []
                if citations:
                    formatted_output.append(f"Key Citations: {', '.join(citations[:3])}")
                
                provisions = meta.get('provisions', '[]')
                if isinstance(provisions, str):
                    try:
                        provisions = json.loads(provisions)
                    except:
                        provisions = []
                if provisions:
                    formatted_output.append(f"Legal Provisions: {', '.join(provisions[:3])}")
                
                formatted_output.append(f"Section Type: {meta.get('section_type', 'N/A')}")
                formatted_output.append(f"Paragraph: {meta.get('paragraph_number', 'N/A')}")
                
                # Content excerpt
                formatted_output.append(f"\nContent:")
                formatted_output.append(f"{hit['doc'][:800]}...")
                formatted_output.append("")
            
            return "\n".join(formatted_output)
            
        except Exception as e:
            return f"Error performing RAG search: {str(e)}"
    
    def index_documents_tool(self, action: str) -> str:
        """
        Wrapper function to index new documents.
        """
        try:
            if action.lower() in ['index', 'reindex']:
                index_pdfs()
                return "Successfully indexed all PDF documents in the 'pdfs' directory. Documents are now searchable."
            else:
                return "Invalid action. Use 'index' or 'reindex' to process documents."
        except Exception as e:
            return f"Error indexing documents: {str(e)}"

    # ===== Helper Methods =====
    
    def parse_classification_response(self, response: str) -> Tuple[Optional[str], str]:
        match = re.search(r"<valid>\[(GQ|SQ|CQ|SA)\]\s*(.*?)</valid>", response, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        payload_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return None, payload_match.group(1).strip() if payload_match else "❌ Invalid classification response."

    def is_legal_query(self, query: str) -> bool:
        try:
            classification_prompt = f"""
            Text:
            \"\"\"{query}\"\"\" 
            ROLE:
        You are a legal analyst specializing in identifying legal, regulatory,finance and compliance-related content.
        TASK:
        Your task is to analyze the following text and determine if it contains or is derived from legal,fiance, regulatory, or compliance-related content.

        Criteria:
        - 8-K or 10-K forms
        - Includes laws, regulations,policies or compliance policies, terms, obligations, agreements, equity, regulations
        - Mentions stakeholders, filings, or stocks in a legal or regulatory context
        - References SEC rules, corporate governance, or compensation policies
        - Discusses board responsibilities, legal risks, disclosures, or obligations

        Output Instructions:
        - Respond with **only** one word: yes or no
        - Do **not** explain, elaborate, or add any other text
        - The output must be **only** one of: yes or no
        - No punctuation. No formatting. All lowercase.
            """
            raw_result = self.llm._call(classification_prompt)
            result = raw_result.strip().lower()
            words = re.findall(r"\b\w+\b", result)
            return "yes" in words
        except Exception:
            return False
    
    def compare_docs(self, doc_input: str) -> str:
        """
        Compare two documents. Input should be two document IDs separated by comma.
        Example: "doc1_id, doc2_id"
        """
        try:
            doc_ids = [d.strip() for d in doc_input.split(',')]
            if len(doc_ids) != 2:
                return "Please provide exactly two document IDs separated by comma."
            
            # Retrieve documents
            doc1_results = search_hybrid(doc_ids[0], top_k=1)
            doc2_results = search_hybrid(doc_ids[1], top_k=1)
            
            if not doc1_results or not doc2_results:
                return "Could not find one or both documents. Please check document IDs."
            
            doc1_text = doc1_results[0]['doc']
            doc2_text = doc2_results[0]['doc']
            
            comparison_prompt = self.compare_prompt_template.format(
                doc1=doc1_text,
                doc2=doc2_text
            )
            
            result = self.llm.invoke(comparison_prompt).content
            return result
            
        except Exception as e:
            return f"Error comparing documents: {str(e)}"
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize the given text.
        """
        try:
            summary_prompt = self.summarize_prompt_template.format(text=text)
            result = self.llm.invoke(summary_prompt).content
            return result
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def run(self, query: str) -> str:
        """
        Main method to run the agent with a query.
        """
        try:
            # Classify the query first
            classification_prompt = self.classification_prompt_template.format(query=query)
            classification_result = self.llm.invoke(classification_prompt).content
            
            query_type, payload = self.parse_classification_response(classification_result)
            
            if not query_type:
                # Non-legal query or needs clarification
                return payload
            
            # Legal query - let agent handle it
            response = self.agent.run(payload)
            return response
            
        except Exception as e:
            return f"Error processing query: {str(e)}"


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    # Initialize the RAG system
    print("\n" + "="*60)
    print("INITIALIZING LEGAL AI AGENT WITH INTEGRATED RAG SYSTEM")
    print("="*60)
    
    # Index PDFs if needed
    print("\nChecking for documents to index...")
    index_pdfs()
    
    # Initialize Gemini LLM
    print("\nInitializing Gemini AI model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        handle_parsing_errors=True 
    )
    
    # Initialize Agent
    print("Initializing BossAgent...")
    agent = BossAgent(llm)
    
    print("\n" + "="*60)
    print("AGENT READY - Enhanced with Hybrid RAG Search")
    print("="*60)
    print("\nCapabilities:")
    print("  ✓ Hybrid search (semantic + keyword)")
    print("  ✓ Pakistani court judgment analysis")
    print("  ✓ Case comparison and summarization")
    print("  ✓ Document indexing and retrieval")
    print("  ✓ Legal classification and routing")
    print("\nFeatures:")
    print("  • Search by case name, number, or citation")
    print("  • Search by legal provisions (Section 302 PPC)")
    print("  • Extract judge names, parties, and precedents")
    print("  • Intelligent chunking for court judgments")
    print("\nPowered by: Google Gemini 1.5 Pro")
    print("="*60)
    
    # Interactive loop
    print("\n💬 Chat with the Legal AI Agent (type 'exit' to quit)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        try:
            print("\n🤖 Agent: ", end="", flush=True)
            response = agent.run(user_input)
            print(response)
            print()
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
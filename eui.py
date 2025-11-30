import gradio as gr
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# Import all backend components from your main file
try:
    from equity import (
        BossAgent, 
        index_pdfs, 
        extract_text_from_pdf,
        ChatGoogleGenerativeAI
    )
    print("✅ Successfully imported from backend.py")
except ImportError:
    try:
        from equity import (
            BossAgent, 
            index_pdfs, 
            extract_text_from_pdf,
            ChatGoogleGenerativeAI
        )
        print("✅ Successfully imported from legal_ai_backend.py")
    except ImportError:
        print("⚠️ Backend file not found. Please ensure backend.py or legal_ai_backend.py exists")
        from langchain_google_genai import ChatGoogleGenerativeAI

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB0SliiR9q3yulmULDyKm-xrn41eML0rWs"

# Global variables
agent = None
llm = None
uploaded_files_context = {}  # Store uploaded file contents

def initialize_agent():
    """Initialize the Legal AI Agent"""
    global agent, llm
    try:
        print("Initializing Legal AI Agent...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            handle_parsing_errors=True 
        )
        agent = BossAgent(llm)
        print("✅ Agent initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        return False

def is_legal_document(text: str, file_name: str) -> Tuple[bool, str]:
    """Check if document contains legal content using LLM"""
    try:
        # Take first 3000 chars for quick analysis
        sample = text[:3000]
        
        prompt = f"""Analyze if this document is legal-related content (court cases, laws, contracts, legal opinions, regulations, etc.).

Document name: {file_name}
Document sample:
{sample}

Respond in this EXACT format:
IS_LEGAL: [YES/NO]
REASON: [One sentence explanation]

Examples:
IS_LEGAL: YES
REASON: Document contains case law citations and legal terminology.

IS_LEGAL: NO
REASON: Document appears to be a personal recipe collection."""
        
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        # Parse response
        is_legal = "IS_LEGAL: YES" in result.upper()
        
        # Extract reason
        reason_lines = [line for line in result.split('\n') if 'REASON:' in line.upper()]
        reason = reason_lines[0].split(':', 1)[1].strip() if reason_lines else "Unable to determine content type"
        
        return is_legal, reason
        
    except Exception as e:
        print(f"⚠️ Error checking document legality: {str(e)}")
        # Default to accepting on error to avoid blocking valid docs
        return True, "Could not verify (proceeding anyway)"

def process_file_upload(files, index_for_search):
    """Process uploaded files - conditionally add to RAG index"""
    global uploaded_files_context
    
    # CRITICAL FIX: Clear context when files are removed
    if not files:
        uploaded_files_context = {}
        return None, False, "All files removed."
    
    try:
        # Get current file names from uploaded files
        current_files = {Path(f.name).name for f in files if f is not None}
        
        # Remove files from context that are no longer uploaded
        files_to_remove = [fname for fname in uploaded_files_context.keys() if fname not in current_files]
        for fname in files_to_remove:
            del uploaded_files_context[fname]
            print(f"🗑️ Removed {fname} from context")
        
        pdf_dir = Path("pdfs")
        pdf_dir.mkdir(exist_ok=True)
        
        uploaded_info = []
        rejected_files = []
        
        for file in files:
            if file is None:
                continue
                
            file_path = Path(file.name)
            file_name = file_path.name
            
            # Skip if already processed
            if file_name in uploaded_files_context:
                uploaded_info.append(f"✓ {file_name} (already loaded)")
                continue
            
            # Handle PDFs
            if file_path.suffix.lower() == '.pdf':
                dest = pdf_dir / file_name
                shutil.copy2(file.name, dest)
                
                # Extract text and store in context
                text = extract_text_from_pdf(dest)
                if text.strip():
                    # CHECK IF LEGAL DOCUMENT
                    is_legal, reason = is_legal_document(text, file_name)
                    
                    if is_legal:
                        uploaded_files_context[file_name] = {
                            'text': text,
                            'type': 'pdf',
                            'path': str(dest)
                        }
                        if index_for_search:
                            uploaded_info.append(f"📄 {file_name} (PDF - added to searchable database)")
                        else:
                            uploaded_info.append(f"📄 {file_name} (PDF - ready for summarization)")
                    else:
                        rejected_files.append(f"❌ {file_name} - Not a legal document ({reason})")
                        print(f"🚫 Rejected: {file_name} - {reason}")
            
            # Handle text files
            elif file_path.suffix.lower() == '.txt':
                text = Path(file.name).read_text(encoding='utf-8')
                if text.strip():
                    # CHECK IF LEGAL DOCUMENT
                    is_legal, reason = is_legal_document(text, file_name)
                    
                    if is_legal:
                        uploaded_files_context[file_name] = {
                            'text': text,
                            'type': 'txt',
                            'path': file.name
                        }
                        uploaded_info.append(f"📝 {file_name} (Text file)")
                    else:
                        rejected_files.append(f"❌ {file_name} - Not a legal document ({reason})")
                        print(f"🚫 Rejected: {file_name} - {reason}")
        
        # Build status message
        status_parts = []
        
        if uploaded_info:
            # Only index NEW PDFs if requested
            new_pdfs = [info for info in uploaded_info if 'already loaded' not in info and 'PDF' in info]
            if index_for_search and new_pdfs:
                print("Indexing new PDFs for RAG search...")
                index_pdfs()
            
            status_parts.append(f"✅ Currently loaded: {len(uploaded_files_context)} file(s)")
            status_parts.append("\n".join(uploaded_info))
            
            if index_for_search and new_pdfs:
                status_parts.append("\n🔍 Documents indexed for search queries!")
            elif uploaded_files_context:
                status_parts.append("\n📝 Documents ready for summarization/comparison")
        
        if rejected_files:
            status_parts.append("\n\n⚠️ REJECTED FILES (Not legal documents):")
            status_parts.extend(rejected_files)
            status_parts.append("\n💡 Tip: Only upload legal documents (contracts, cases, laws, etc.)")
        
        if uploaded_info or rejected_files:
            return files, index_for_search, "\n".join(status_parts)
        else:
            uploaded_files_context = {}
            return None, False, "❌ No valid legal documents found. Please upload legal PDF or TXT files."
            
    except Exception as e:
        return files, index_for_search, f"❌ Error processing files: {str(e)}"

def direct_summarize(text: str, file_name: str) -> str:
    """Directly summarize text using LLM without chunking"""
    try:
        prompt = f"""Please provide a comprehensive summary of the following document: "{file_name}"

Document content:
{text}

Provide a well-structured summary that covers:
1. Main topics and key points
2. Important findings or conclusions
3. Any critical details or data

Summary:"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        raise Exception(f"Summarization error: {str(e)}")

def direct_compare(text1: str, text2: str, file1_name: str, file2_name: str) -> str:
    """Directly compare two documents without chunking"""
    try:
        prompt = f"""Compare and contrast these two documents:

Document 1: {file1_name}
{text1}

---

Document 2: {file2_name}
{text2}

Please provide a detailed comparison covering:
1. **Similarities**: What do both documents share in common?
2. **Differences**: How do they differ in content, approach, or conclusions?
3. **Key Insights**: What stands out in each document?
4. **Overall Assessment**: Which document is more comprehensive/relevant for specific use cases?

Comparison:"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        raise Exception(f"Comparison error: {str(e)}")

def chat_with_agent(message: str, history: List[Tuple[str, str]], files):
    """Main chat interface - agent figures out what to do"""
    
    if not agent:
        yield history + [(message, "⚠️ Initializing agent, please wait...")], files
        if initialize_agent():
            yield history + [(message, "✅ Agent ready! Processing your message...")], files
        else:
            yield history + [(message, "❌ Failed to initialize agent. Please check your API key.")], files
            return
    
    if not message.strip():
        yield history, files
        return
    
    # Build context-aware message with actual file content
    context_msg = message
    
    # Check if user is asking about uploaded documents
    if uploaded_files_context:
        file_list = list(uploaded_files_context.keys())
        
        # For summarization requests - DIRECT SUMMARIZATION (NO CHUNKING)
        if any(keyword in message.lower() for keyword in ['summarize', 'summary', 'summarise']):
            if len(file_list) == 1:
                # Single file - auto-summarize directly
                file_name = file_list[0]
                file_data = uploaded_files_context[file_name]
                full_text = file_data['text']  # NO TRUNCATION
                
                try:
                    history.append((message, "📝 Summarizing document... (this may take a moment)"))
                    yield history, files
                    
                    # Direct summarization without chunking
                    summary = direct_summarize(full_text, file_name)
                    response = f"## Summary of {file_name}\n\n{summary}"
                    history[-1] = (message, response)
                    yield history, files
                    return
                except Exception as e:
                    history[-1] = (message, f"❌ Error summarizing: {str(e)}")
                    yield history, files
                    return
            elif len(file_list) > 1:
                # Multiple files - ask which one
                file_options = "\n".join([f"- {f}" for f in file_list])
                response = f"📂 You have {len(file_list)} files uploaded:\n\n{file_options}\n\nWhich document would you like me to summarize? Please specify the filename."
                history.append((message, response))
                yield history, files
                return
            else:
                response = "❌ No documents uploaded. Please upload a document first."
                history.append((message, response))
                yield history, files
                return
        
        # For comparison requests - DIRECT COMPARISON (NO CHUNKING)
        elif any(keyword in message.lower() for keyword in ['compare', 'comparison', 'difference', 'similar']):
            if len(file_list) == 2:
                # Two files - auto-compare directly
                file1_name = file_list[0]
                file2_name = file_list[1]
                text1 = uploaded_files_context[file1_name]['text']  # FULL TEXT
                text2 = uploaded_files_context[file2_name]['text']  # FULL TEXT
                
                try:
                    history.append((message, "🔄 Comparing documents... (this may take a moment)"))
                    yield history, files
                    
                    # Direct comparison without chunking
                    comparison = direct_compare(text1, text2, file1_name, file2_name)
                    response = f"## Comparison: {file1_name} vs {file2_name}\n\n{comparison}"
                    history[-1] = (message, response)
                    yield history, files
                    return
                except Exception as e:
                    history[-1] = (message, f"❌ Error comparing: {str(e)}")
                    yield history, files
                    return
            elif len(file_list) > 2:
                file_options = "\n".join([f"- {f}" for f in file_list])
                response = f"📂 You have {len(file_list)} files uploaded:\n\n{file_options}\n\nPlease specify which TWO documents you'd like me to compare."
                history.append((message, response))
                yield history, files
                return
            elif len(file_list) == 1:
                response = "❌ I need at least 2 documents to compare. Please upload another document."
                history.append((message, response))
                yield history, files
                return
            else:
                response = "❌ No documents uploaded. Please upload at least 2 documents to compare."
                history.append((message, response))
                yield history, files
                return
        
        # For general queries about uploaded docs (use RAG if indexed)
        elif any(keyword in message.lower() for keyword in ['this', 'these', 'uploaded', 'document', 'file']):
            if len(file_list) == 1:
                context_msg = f"[User has uploaded: {file_list[0]}. Use RAG search if indexed, otherwise provide general guidance.]\n\nUser query: {message}"
            elif len(file_list) > 1:
                context_msg = f"[User has uploaded {len(file_list)} files: {', '.join(file_list)}.]\n\nUser query: {message}"
    
    try:
        # Let the agent handle general queries, searches, etc.
        response = agent.run(context_msg)
        
        history.append((message, response))
        yield history, files
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        yield history, files

def clear_chat():
    """Clear chat history and uploaded files context"""
    global uploaded_files_context
    uploaded_files_context = {}
    if agent:
        agent.memory.clear()
    return [], None, False, ""

# Custom CSS for ChatGPT-like styling
css = """
#chatbot {
    height: 600px;
    border-radius: 10px;
}

.message {
    padding: 15px;
    margin: 10px 0;
}

#main-container {
    max-width: 1200px;
    margin: 0 auto;
}

.upload-area {
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    background: #f9fafb;
    transition: all 0.3s;
}

.upload-area:hover {
    border-color: #667eea;
    background: #f3f4f6;
}

.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 30px;
    text-align: center;
}

.input-box {
    border-radius: 24px !important;
    border: 2px solid #e5e7eb !important;
    padding: 12px 20px !important;
}

.send-btn {
    border-radius: 20px !important;
    min-width: 100px !important;
}

.footer-text {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #6b7280;
    font-size: 13px;
}
"""

# Build Interface
with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Legal AI Assistant") as demo:
    
    gr.HTML("""
        <div class="header-gradient">
            <h1 style="margin: 0; font-size: 2.5em;">⚖️ Legal AI Assistant</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.95;">
                Chat naturally - I'll understand if you want to search, summarize, or compare documents
            </p>
        </div>
    """)
    
    with gr.Row(elem_id="main-container"):
        with gr.Column(scale=1):
            
            # Main Chatbot
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                height=600,
                show_label=False,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False
            )
            
            # File Upload Section
            with gr.Group():
                file_upload = gr.File(
                    label="📎 Attach Documents (PDF or TXT)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt"],
                    type="filepath",
                    elem_classes="upload-area"
                )
                
                index_checkbox = gr.Checkbox(
                    label="🔍 Add to searchable database (for RAG queries)",
                    value=False,
                    info="Check this if you want to search within documents. Leave unchecked for summarization/comparison only."
                )
                
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    show_label=False,
                    placeholder="Upload files here...",
                    lines=3
                )
            
            # Message Input
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me anything about legal cases, or ask me to summarize/compare your uploaded documents...",
                    show_label=False,
                    scale=9,
                    lines=2,
                    elem_classes="input-box"
                )
                send_btn = gr.Button("Send", scale=1, variant="primary", elem_classes="send-btn")
            
            # Action Buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")
                
            # Example Queries
            gr.Markdown("### 💡 Try asking:")
            gr.Examples(
                examples=[
                    "Summarize the uploaded document",
                    "Compare these two documents",
                    "Find corruption cases from 2020",
                    "What are the penalties under Section 302 PPC?",
                    "Show me recent Lahore High Court judgments",
                    "Search for cases involving property disputes",
                ],
                inputs=msg_input,
            )
    
    # Footer
    gr.HTML("""
        <div class="footer-text">
            <p><strong>🤖 Powered by Google Gemini AI</strong> | Built for Pakistani Legal Professionals</p>
            <p style="font-size: 11px; margin-top: 10px;">
                ⚠️ This AI assistant provides information only. Always verify with qualified legal professionals.
            </p>
            <p style="font-size: 11px; color: #9ca3af;">
                <strong>How to use:</strong><br>
                • For <strong>summarization/comparison</strong>: Upload docs WITHOUT checking the database option<br>
                • For <strong>RAG search</strong>: Upload docs WITH the database option checked
            </p>
        </div>
    """)
    
    # Event Handlers
    
    # Handle file uploads
    file_upload.change(
        process_file_upload,
        inputs=[file_upload, index_checkbox],
        outputs=[file_upload, index_checkbox, upload_status]
    )
    
    # Handle chat messages
    msg_input.submit(
        chat_with_agent,
        inputs=[msg_input, chatbot, file_upload],
        outputs=[chatbot, file_upload]
    ).then(
        lambda: "", None, msg_input
    )
    
    send_btn.click(
        chat_with_agent,
        inputs=[msg_input, chatbot, file_upload],
        outputs=[chatbot, file_upload]
    ).then(
        lambda: "", None, msg_input
    )
    
    # Clear chat
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, file_upload, index_checkbox, upload_status]
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING LEGAL AI ASSISTANT")
    print("="*60)
    
    # Initialize agent on startup
    print("\nInitializing agent...")
    if initialize_agent():
        print("✅ Agent ready!")
    else:
        print("⚠️ Agent will initialize on first message")
    
    print("\n" + "="*60)
    print("🌐 Starting Gradio Interface...")
    print("="*60)
    
    # Launch Gradio
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
import streamlit as st
import os
import tempfile
from typing import List
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse

# Updated imports for langchain v0.2+
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from whatsapp_service import WhatsAppService

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="News Research Agent",
    page_icon="🔍",
    layout="wide"
)

VECTORSTORE_PATH = "faiss_store_combined"

def process_urls_to_documents(urls: List[str]) -> List[Document]:
    """Process URLs and return documents"""
    all_documents = []
    
    for url in urls:
        if url.strip():
            try:
                # Validate URL
                parsed = urlparse(url.strip())
                if not all([parsed.scheme, parsed.netloc]):
                    st.error(f"Invalid URL: {url}")
                    continue
                
                # Use WebBaseLoader to load content from URL
                loader = WebBaseLoader(url.strip())
                docs = loader.load()
                all_documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading URL {url}: {str(e)}")
                continue
    
    return all_documents

def process_files_to_documents(files) -> List[Document]:
    """Process uploaded files and return documents"""
    all_documents = []
    
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
            content = file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Determine file type and use appropriate loader
            if file.name.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
            elif file.name.lower().endswith(('.xlsx', '.xls')):
                loader = UnstructuredExcelLoader(tmp_file.name)
            elif file.name.lower().endswith('.txt'):
                loader = TextLoader(tmp_file.name)
            else:
                # Try to load as text
                loader = TextLoader(tmp_file.name)
            
            try:
                docs = loader.load()
                all_documents.extend(docs)
            finally:
                os.unlink(tmp_file.name)  # Clean up temp file
    
    return all_documents

def send_notification(message: str, notification_type: str = "success") -> dict:
    """Send notification and return status"""
    try:
        # WhatsApp notification using WhatsAppService
        # SUPPORT_WHATSAPP_NUMBER=+919985813175
        

        resp = service.send_whatsapp_message({

            "campaignName": "utility_b2c_trading_354",
            "mobileNumber": ["+918405927415"],
            "templateParams": ["Naman", message,"sdghhj" ],
            "media": {
                "url": "https://pickright-server.s3.ap-south-1.amazonaws.com/live/assets/assets/videos/diversified_portfolios.mp4",
                "filename": "user_paid_subscription"
            }
        })

        print(resp)
        
        # For now, we'll simulate notification sending
        notification_data = {
            "status": "sent",
            "message": message,
            "type": notification_type,
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
        }

        
        # Log the notification (you can replace this with actual notification service)
        print(f"Notification sent: {notification_data}")
        
        return {
            "success": True,
            "data": notification_data,
            "alert_message": f"✅ Notification sent successfully!\nMessage: {message}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "alert_message": f"❌ Failed to send notification: {str(e)}"
        }

def generate_content_summary(documents: List[Document]) -> str:
    """Generate a brief summary of the processed content"""
    if not documents:
        return "No content to summarize."
    
    # Combine first few chunks of content for summary
    combined_content = ""
    for doc in documents[:3]:  # Use first 3 documents
        combined_content += doc.page_content[:500] + "\n\n"  # First 500 chars of each
    
    if len(combined_content.strip()) == 0:
        return "Content processed but unable to generate summary."
    
    try:
        # Use OpenAI to generate summary
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=600)
        summary_prompt = f"""Provide a comprehensive summary of the following content in approximately 400 words. 
        Make it detailed and informative, covering all key points and important details:
        
        {combined_content[:2000]}  # Increased character limit for more context
        
        Comprehensive Summary (400 words):"""
        
        summary = llm.invoke(summary_prompt)
        return summary.content.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

def process_and_store_documents(documents: List[Document]):
    """Process documents and store in vectorstore"""
    if not documents:
        return False, "No documents to process", ""
    
    # Generate summary first
    summary = generate_content_summary(documents)
    
    # Create a summary document and add it to the documents
    summary_doc = Document(
        page_content=f"CONTENT SUMMARY: {summary}",
        metadata={"source": "content_summary", "type": "summary"}
    )
    documents_with_summary = documents + [summary_doc]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], 
        chunk_size=1000
    )
    docs = text_splitter.split_documents(documents_with_summary)
    
    # Create embeddings and save vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    
    # Save vectorstore
    vectorstore_openai.save_local(VECTORSTORE_PATH)
    
    return True, f"Processed {len(documents)} documents into {len(docs)} chunks", summary

def generate_financial_report_summary():
    """Generate a comprehensive financial report analysis summary"""
    if not os.path.exists(VECTORSTORE_PATH):
        return None, "No processed documents found. Please upload and process files first."
    
    try:
        # Load vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Create LLM with specific settings for report analysis
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=1500)
        
        # Create retriever to get all relevant content
        retriever = vectorstore_openai.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}  # Retrieve more chunks to capture all financial data
        )
        
        # Financial report analysis prompt
        financial_analysis_prompt = """
Role: You are an expert financial research report analyser.

Objective: To analyse attached report and create summary of report.

Context: I am attaching a research report of a stock, give me one pager summary of all details

Instructions: Make it easily readable, with bullet points.

Instructions: Add a small section on top of output with heading "Report Inference", it should give inference of report in 400 Words. Specifically give details that investor should buy, sell, hold that stock

IMPORTANT: Please include ALL specific financial data mentioned in the report, including but not limited to:
- 52-week high and low range
- Current stock price
- Market cap
- P/E ratio
- Revenue figures
- Profit/Loss data
- Price targets
- Any financial metrics or ratios mentioned

Based on the provided financial research report, please provide a comprehensive analysis following the above format. Make sure to extract and include all numerical data and financial metrics found in the documents.
        """
        
        # Create chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get analysis
        result = chain({"question": financial_analysis_prompt}, return_only_outputs=True)
        
        return result, None
        
    except Exception as e:
        return None, f"Error generating financial report summary: {str(e)}"

def ask_question_from_vectorstore(question: str):
    """Ask a question based on processed documents - first check summary, then vector DB"""
    if not os.path.exists(VECTORSTORE_PATH):
        return None, "No processed documents found. Please upload and process files first."
    
    try:
        # Step 1: First try to answer from the generated summary
        import streamlit as st
        if hasattr(st.session_state, 'content_summary') and st.session_state.content_summary:
            # Create LLM for summary-based answering
            llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=300)
            
            summary_prompt = f"""Based ONLY on the following content summary, answer this question: {question}

Content Summary:
{st.session_state.content_summary}

If the answer can be found in the summary above, provide a clear answer. If the information is not available in this summary, respond with exactly: "NOT_IN_SUMMARY"

Answer:"""
            
            summary_result = llm.invoke(summary_prompt)
            summary_answer = summary_result.content.strip()
            
            # If answer found in summary, return it
            if summary_answer and "NOT_IN_SUMMARY" not in summary_answer.upper():
                return {
                    "answer": summary_answer,
                    "sources": "Content Summary (Generated from uploaded documents)",
                    "source_documents": []
                }, None
        
        # Step 2: If not found in summary, use vector database
        # Load vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Create LLM with lower temperature for more focused answers
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=500)
        
        # Create retriever with more documents for better context
        retriever = vectorstore_openai.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve top 5 most relevant chunks
        )
        
        # Create chain with document-focused settings
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, 
            retriever=retriever,
            return_source_documents=True
        )
        
        # Modify the question to enforce document-only responses
        enhanced_question = f"""Based ONLY on the provided documents, {question}. 
        If this information is not available in the documents, please say "I cannot find this information in the uploaded documents.\""""
        
        # Get answer using the enhanced question
        result = chain({"question": enhanced_question}, return_only_outputs=True)
        
        # Additional validation: check if the answer indicates lack of information
        answer = result["answer"].strip()
        if not answer or answer.lower() in ["i don't know", "i cannot find", "not mentioned", "no information"]:
            result["answer"] = "I cannot find this information in the processed content. Please make sure your question relates to the content you've uploaded or URLs you've added."
        
        # Add note that this came from vector database
        if result.get("sources"):
            result["sources"] = f"Vector Database Search: {result['sources']}"
        else:
            result["sources"] = "Vector Database Search (from uploaded documents)"
        
        return result, None
        
    except Exception as e:
        return None, f"Error processing question: {str(e)}"

def main():
    st.title("🔍 News Research Agent")
    st.markdown("Upload documents or add URLs and ask questions about their content")
    
    # Initialize session state
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False
    if 'processing_info' not in st.session_state:
        st.session_state.processing_info = ""
    if 'content_summary' not in st.session_state:
        st.session_state.content_summary = ""
    if 'show_notification' not in st.session_state:
        st.session_state.show_notification = False
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload & Process Content")
        
        # Add tabs for files and URLs
        tab1, tab2 = st.tabs(["📄 Files", "🌐 URLs"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'xlsx', 'xls', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF, Excel, or text files"
            )
            
            if uploaded_files:
                st.write(f"Selected {len(uploaded_files)} file(s):")
                for file in uploaded_files:
                    st.write(f"- {file.name}")
        
        with tab2:
            st.markdown("**Add URLs to process:**")
            url_input = st.text_area(
                "Enter URLs (one per line)",
                height=100,
                placeholder="https://example.com/article1\nhttps://example.com/article2",
                help="Enter one URL per line"
            )
            
            urls = []
            if url_input:
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                if urls:
                    st.write(f"URLs to process ({len(urls)}):")
                    for i, url in enumerate(urls, 1):
                        st.write(f"{i}. {url}")
        
        # Process button (works for both files and URLs)
        has_content = bool(uploaded_files) or bool(urls)
        if st.button("🔄 Process Content", disabled=not has_content):
            if has_content:
                with st.spinner("Processing content..."):
                    try:
                        all_documents = []
                        
                        # Process files if any
                        if uploaded_files:
                            file_documents = process_files_to_documents(uploaded_files)
                            all_documents.extend(file_documents)
                            st.success(f"✅ Processed {len(uploaded_files)} file(s)")
                        
                        # Process URLs if any
                        if urls:
                            url_documents = process_urls_to_documents(urls)
                            all_documents.extend(url_documents)
                            st.success(f"✅ Processed {len(urls)} URL(s)")
                        
                        # Store all documents
                        if all_documents:
                            success, message, summary = process_and_store_documents(all_documents)
                            
                            if success:
                                st.success("✅ Content processed successfully!")
                                st.info(f"📊 {message}")
                                st.session_state.files_processed = True
                                st.session_state.processing_info = message
                                st.session_state.content_summary = summary
                            else:
                                st.error(f"❌ Error: {message}")
                        else:
                            st.error("❌ No content was successfully processed")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing content: {str(e)}")
        
        # Display content summary if available
        if st.session_state.content_summary:
            st.markdown("### 📄 Content Summary")
            st.markdown(st.session_state.content_summary)
            
            # Add notification section
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🔔 Send Notification"):
                    # Use the new send_notification method
                    notification_result = send_notification(
                        message=f"Content Summary: {st.session_state.content_summary[:100]}...",
                        notification_type="info"
                    )
                    
                    # Show alert based on result
                    if notification_result["success"]:
                        st.success(notification_result["alert_message"])
                        st.session_state.show_notification = True
                    else:
                        st.error(notification_result["alert_message"])
            
            with col_btn2:
                if st.button("🚨 Send Alert"):
                    # Send alert notification with the actual summary content
                    alert_result = send_notification(
                        message=f"ALERT - CONTENT SUMMARY:\n\n{st.session_state.content_summary}\n\nProcessing Info: {st.session_state.processing_info}",
                        notification_type="alert"
                    )
                    
                    # Show alert with warning style including the summary
                    if alert_result["success"]:
                        st.warning(f"🚨 ALERT SENT!\n\n**Summary Alerted:**\n{st.session_state.content_summary}")
                        st.info(f"✅ {alert_result['alert_message']}")
                    else:
                        st.error(alert_result["alert_message"])
            
            # Show notification status if available
            if st.session_state.show_notification:
                if st.button("Clear Notification Status"):
                    st.session_state.show_notification = False
    
    with col2:
        st.header("❓ Ask Questions & Analysis")
        
        if st.session_state.files_processed:
            st.success("✅ Files are ready for questions!")
            st.info(st.session_state.processing_info)
        elif os.path.exists(VECTORSTORE_PATH):
            st.success("✅ Previously processed files are available!")
            st.session_state.files_processed = True
        else:
            st.info("ℹ️ Please upload and process files first")
        
        # Add tabs for different analysis types
        analysis_tab1, analysis_tab2 = st.tabs(["💬 Q&A", "📊 Financial Report Analysis"])
        
        with analysis_tab1:
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about the uploaded documents?",
                disabled=not st.session_state.files_processed and not os.path.exists(VECTORSTORE_PATH)
            )
            
            if st.button("🔍 Get Answer", disabled=not question or (not st.session_state.files_processed and not os.path.exists(VECTORSTORE_PATH))):
                if question:
                    with st.spinner("Finding answer..."):
                        try:
                            result, error = ask_question_from_vectorstore(question)
                            
                            if error:
                                st.error(f"❌ {error}")
                            else:
                                st.markdown("### 📋 Answer:")
                                st.write(result['answer'])
                                
                                if result.get('sources') and result['sources'].strip():
                                    st.markdown("### 📚 Sources:")
                                    st.write(result['sources'])
                                    
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with analysis_tab2:
            st.markdown("### 📊 Financial Research Report Analysis")
            st.markdown("Generate a comprehensive analysis of your uploaded financial research report with investment recommendations.")
            
            if st.button("📈 Generate Financial Report Summary", disabled=not st.session_state.files_processed and not os.path.exists(VECTORSTORE_PATH)):
                with st.spinner("Analyzing financial report..."):
                    try:
                        result, error = generate_financial_report_summary()
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.markdown("### 📊 Financial Report Analysis:")
                            st.markdown(result['answer'])
                            
                            if result.get('sources') and result['sources'].strip():
                                st.markdown("### 📚 Sources:")
                                st.write(result['sources'])
                                
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("## 📖 How to Use")
        st.markdown("""
        1. **Add Content**: 
           - Upload files (PDF, Excel, Text)
           - Or add URLs to web articles/pages
        2. **Process Content**: Click "Process Content" to analyze and index your content
        3. **Ask Questions**: Once processing is complete, ask questions about your content
        4. **Financial Analysis**: Use the "Financial Report Analysis" tab for comprehensive stock report analysis
        
        **Supported Sources:**
        - 📄 PDF files
        - 📊 Excel files (.xlsx, .xls)
        - 📝 Text files (.txt)
        - 🌐 Web URLs (articles, blogs, etc.)
        """)
        
        st.markdown("## 🔧 Features")
        st.markdown("""
        - **Multi-Source**: Process files and web URLs together
        - **Smart Processing**: Automatically handles different content types
        - **Financial Analysis**: Expert analysis of research reports with buy/sell/hold recommendations
        - **Persistent Storage**: Processed content is saved for future use
        - **Source Attribution**: Get sources for answers when available
        - **Fast Retrieval**: Uses FAISS for efficient content search
        """)
        
        # Clear processed content
        if st.button("🗑️ Clear Processed Content"):
            try:
                if os.path.exists(VECTORSTORE_PATH):
                    import shutil
                    shutil.rmtree(VECTORSTORE_PATH)
                st.session_state.files_processed = False
                st.session_state.processing_info = ""
                st.session_state.content_summary = ""
                st.session_state.show_notification = False
                st.success("✅ Processed content cleared!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Error clearing content: {str(e)}")

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
import streamlit as st
import PyPDF2 as pypdf
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

# ---------- Configuration ----------
PDF_FOLDER = "financial_pdfs"     # Folder containing your curated PDFs
INDEX_PATH = "finance_index"      # Folder where FAISS index will be saved
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------- Helpers ----------
def load_all_pdfs(folder: str) -> str:
    """Concatenate text from all PDFs in the folder."""
    text = []
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            with open(os.path.join(folder, file), "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text.append(extracted)
    return "\n".join(text)

@st.cache_resource(show_spinner=False)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource(show_spinner=False)
def load_or_build_index(_embeddings: OpenAIEmbeddings):
    """
    Fast path: load the FAISS index if it exists.
    Slow path: read PDFs, embed, and save the index once.
    """
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True
        )

    st.info("Index not found – building from PDFs (this may take a while)...")
    raw_text = load_all_pdfs(PDF_FOLDER)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    vector = FAISS.from_texts(chunks, _embeddings)
    vector.save_local(INDEX_PATH)
    st.success("Index built and cached for future runs.")
    return vector


def build_chain():
    prompt_template = PromptTemplate(
        template=(
            "You are a friendly financial literacy tutor for beginners. "
            "Answer the user's question **only** using the provided context. "
            "If unsure, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        ),
        input_variables=["context", "question"],
    )
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

# ---------- Streamlit App ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Finance Tutor", page_icon="💵")
    st.title("💵 Financial Literacy Tutor")
    st.caption("Educational only – not personalized financial advice.")

    embeddings = get_embeddings()
    vector_store = load_or_build_index(embeddings)
    chain = build_chain()

    # Optional: button to rebuild index
    if st.button("🔄 Rebuild Knowledge Base"):
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
            st.cache_resource.clear()  # clear cached index
            st.experimental_rerun()

    # User question
    question = st.chat_input("Ask a beginner finance question (budgeting, saving, credit, etc.)")
    if question:
        with st.spinner("Searching knowledge base..."):
            docs = vector_store.similarity_search(question, k=4)
            with get_openai_callback() as cb:
                answer = chain.run(input_documents=docs, question=question)
        st.write(answer)
        st.caption(f"Tokens used: {cb.total_tokens}")

if __name__ == "__main__":
    main()

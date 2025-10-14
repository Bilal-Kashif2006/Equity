import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Gemini API key missing. Please set GEMINI_API_KEY in your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.4
)

file_path = "sample.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

# Combine all pages into one string
full_text = "\n".join([page.page_content for page in pages])

custom_prompt = PromptTemplate(
    input_variables=["document"],
    template=(
        "You are a legal assistant specializing in summarizing complex legal documents.\n"
        "Read the following case text carefully and summarize it in a way that captures:\n"
        "- The background and facts of the case\n"
        "- The legal issue(s) discussed\n"
        "- The court’s reasoning and final decision\n\n"
        "Make it clear, concise, and beginner-friendly while preserving legal accuracy.\n\n"
        "Document:\n{document}\n\n"
        "Summary:"
    )
)

summarizer_chain = LLMChain(
    llm=llm,
    prompt=custom_prompt,
    verbose=True
)

summary = summarizer_chain.run(document=full_text)

print("\n==== SUMMARY ====\n")
print(summary)

from urllib import response
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from langchain.tools import StructuredTool
from pydantic import BaseModel,Field
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)



# Load environment variables from .env file
load_dotenv()

class ConcatenateDocArgs(BaseModel):
    doc1: str=Field(..., description="The text content of the first document to be compared.")
    doc2: str=Field (..., description="The text content of the second document to be compared.")

class LegalAgent:
    def __init__(self,llm, prompt, memory):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.tools=[
            # StructuredTool.from_function(name="Compare_documents",
            #      func=self.compare_documents,
            #      description="Useful for comparing two legal documents and highlighting differences.",
            #      args_schema=ConcatenateDocArgs
            #      ),
            Tool.from_function(
            name="Compare_documents",
            func=self.compare_documents,
            description="Useful for comparing two legal documents and highlighting differences."
            )

        ]

        self.agent = create_react_agent(llm=llm, tools=self.tools, prompt=prompt,)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,  # Helps catch malformed tool responses
            return_intermediate_steps=True
        )
        self.GDPR="""The General Data Protection Regulation (GDPR) establishes a comprehensive framework governing the collection, processing, and storage of personal data within the European Union. Its central purpose is to empower individuals with greater control over their personal information while harmonizing privacy laws across member states. Under GDPR, any entity that processes personal data must ensure that such processing is lawful, fair, and transparent. Consent plays a pivotal role, requiring it to be freely given, specific, informed, and unambiguous. Individuals are granted several enforceable rights, including the right of access, rectification, erasure (the so-called “right to be forgotten”), and data portability. Organizations must also uphold the principle of data minimization, collecting only the information necessary for the stated purpose and retaining it for no longer than required.

        Controllers are obliged to demonstrate accountability by maintaining detailed records of processing activities and, where appropriate, appointing a Data Protection Officer (DPO). In the event of a data breach, notification to supervisory authorities must occur within seventy-two hours, accompanied by a transparent communication to affected individuals when risks are substantial. GDPR further mandates privacy by design and default, meaning that safeguards for personal data must be integrated into systems and practices from the earliest stages of development. Transfers of data outside the EU are permitted only when adequate protection levels are guaranteed, such as through Standard Contractual Clauses or adequacy decisions. Non-compliance can lead to severe administrative fines—up to twenty million euros or four percent of global annual turnover, whichever is higher—ensuring that enforcement mechanisms carry real deterrent value.

        Importantly, GDPR’s extraterritorial scope extends its reach beyond the EU’s physical borders, applying to organizations that offer goods or services to EU residents or monitor their behavior. This global influence has encouraged non-EU jurisdictions to raise their privacy standards in line with European expectations. The regulation thereby positions data protection as both a legal obligation and a competitive differentiator. Companies that demonstrate transparency, accountability, and respect for individual privacy can strengthen consumer trust and mitigate reputational risk. Ultimately, GDPR represents a paradigm shift from reactive compliance to proactive data stewardship, embedding privacy as a core component of modern corporate governance rather than a peripheral legal requirement."""

        self.CCPA="""The California Consumer Privacy Act (CCPA) serves as a landmark statute aimed at enhancing consumer privacy rights and business transparency in the digital economy of the United States. Enacted in 2018, it grants California residents the right to know what personal information businesses collect, use, or disclose about them. Consumers may request the categories and specific pieces of data held by a company, demand deletion of that data under certain conditions, and opt out of the sale or sharing of their personal information to third parties. The law defines “personal information” broadly, encompassing identifiers, commercial data, internet activity, geolocation, and inferences drawn from profiling. CCPA also obliges businesses to include a conspicuous “Do Not Sell or Share My Personal Information” link on their websites, empowering individuals to exercise control with minimal friction.

        Compliance obligations under CCPA depend on business size and scope—typically those that meet revenue thresholds, handle large volumes of consumer data, or derive significant income from selling personal information. Covered entities must update their privacy policies annually and train employees on proper handling of consumer requests. The California Privacy Protection Agency (CPPA) oversees enforcement, conducting audits and investigations where violations are suspected. Civil penalties may reach $7,500 per intentional violation, while consumers themselves have limited rights to bring private actions for data breaches resulting from negligent security practices. Although the CCPA does not require a dedicated Data Protection Officer, it emphasizes organizational accountability through clear documentation and accessible communication channels.

        The CCPA, as amended by the California Privacy Rights Act (CPRA), brings the U.S. privacy landscape closer to global standards but stops short of GDPR’s comprehensiveness. Unlike GDPR’s prior consent model, CCPA primarily operates on an opt-out basis, allowing businesses to process data until consumers object. Its jurisdiction is limited to California residents, yet its practical impact extends nationwide as companies adopt uniform privacy practices for efficiency. While GDPR treats privacy as a fundamental human right rooted in dignity and autonomy, CCPA approaches it as a consumer protection issue balancing innovation and regulation. Together, both frameworks signal an era in which transparency, accountability, and user empowerment have become essential for sustaining trust in a data-driven society. However, differences in enforcement strength, individual remedies, and legal philosophy reveal the ongoing divergence between European and American privacy regimes."""

        self.compare_prompt=PromptTemplate.from_template("""
Task:
Carefully analyze the two documents provided and identify all similarities and differences between them.

Instructions:
- Treat content as similar if the two documents express the same meaning or intent(semantic similarity).
- If some content is present in one document but missing in the other, this is a difference.
- If the context or framing of some content differs (e.g., policy enforcement, time frame, or impact level), mark it as a difference.
- Do not assume similarity  unless the intent is clearly shared.
- Get max number of similarities and differences

Example:
- A topic can appear as both a similarity and a difference. 
- Similarity: "Both documents discuss energy efficiency."
- Difference: "Document 2 emphasizes product-level efficiency, which is not mentioned in Document 1."

Output Format:
- Output should be  valid  and jump straight to it without intro.
- Each type should have a seperate(similarity/differences) heading
- Each entry should include:
    - "number": (starting from 1)
    - "heading": Short title
    - "explanation": Brief explanation of the similarity or difference
    - "doc1_line": Quote or phrase from document 1
    - "doc2_line": Quote or phrase from document 2

Analyze:
- Revise and make sure you got all each and every similarities and difference

DOCS are:
doc1  
{doc1}

doc2  
{doc2}
"""
        )

    # def compare_documents(self, doc1=None, doc2=None, **kwargs):
    #     print("🧩 compare_documents() called with:", doc1, doc2, kwargs)
    #     doc1 = kwargs["doc1"]
    #     doc2 = kwargs["doc2"]
    #         # Map keywords to actual stored text
    #     if doc1.strip().upper() == "GDPR":
    #         doc1 = self.GDPR
    #     if doc2.strip().upper() == "CCPA":
    #         doc2 = self.CCPA
    #     prompt_text = self.compare_prompt.format(doc1=doc1, doc2=doc2)
    #     response = self.llm.invoke(prompt_text)
    #     print("Full Tool Output (Compare_documents):\n", response, "\n")  # debug full output
    #     if hasattr(response, "content"):
    #         result = str(response.content)
    #     else:
    #         result = str(response)

    #     print("\n🧠 Tool Output (Compare_documents):\n", result[:1000], "...\n")  # debug preview

    #     return result
    def compare_documents(self, doc1=None, doc2=None, **kwargs):
        """Compare two legal documents, handling flexible input formats."""
        print(f"🧩 compare_documents() called with: {doc1=}, {doc2=}, {kwargs=}")

        # If doc1 is a JSON string, parse it
        import json
        if isinstance(doc1, str):
            try:
                parsed = json.loads(doc1)
                if "doc1" in parsed and "doc2" in parsed:
                    doc1, doc2 = parsed["doc1"], parsed["doc2"]
                    print(f"✅ Parsed JSON input: {doc1=} | {doc2=}")
            except json.JSONDecodeError:
                pass  # not a JSON, continue as is

        # Map keywords to actual stored text
        if doc1 and doc1.strip().upper() == "GDPR":
            doc1 = self.GDPR
        if doc2 and doc2.strip().upper() == "CCPA":
            doc2 = self.CCPA

        # Debug before sending to model
        print(f"🧠 Final comparison input:\n - doc1 len: {len(doc1)}\n - doc2 len: {len(doc2)}")

        prompt_text = self.compare_prompt.format(doc1=doc1, doc2=doc2)
        response = self.llm.invoke(prompt_text)

        # Print result debug
        print("📤 Raw model response:", response)

        result = response.content if hasattr(response, "content") else str(response)
        print("\n🧠 Tool Output (Compare_documents):\n", result[:1000], "...\n")

        return result


    def read_pdf_text(file_path):
        """ Reads all text from a PDF file and returns it as a string."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""
    def chat_with_agent(self):
        while True:
            try:
                user_input=input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting chat.")
                    return 
                self.memory.chat_memory.add_message(HumanMessage(content=user_input))
                response = self.agent_executor.invoke({"input": user_input})
                print("AI:", response["output"])
                self.memory.chat_memory.add_message(AIMessage(content=response["output"]))
            except Exception as e:
                self.memory.chat_memory.add_message(AIMessage(content=f"❌ Error: {str(e)}"))
        return "", self.memory.chat_memory.messages



llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
#prompt = """You are a legal AI assistant that helps users with legal document analysis, summarization, and comparison. Use the provided tools to assist with user queries."""
prompt = PromptTemplate.from_template("""
You are a legal AI assistant that can chat naturally and use tools to assist with legal document analysis.

You have access to the following tools:
{tools}

The available tool names are: {tool_names}

When the user asks to compare documents, use the Compare_documents tool.
Otherwise, reply conversationally and provide helpful, professional answers.

Use the following format when reasoning:
Thought: describe what you think or plan to do.
Action: name of the tool to use (from {tool_names})
Action Input: input to the tool as JSON.
Observation: the result of the tool.
(Repeat Thought/Action/Observation as needed)
Final Answer: your final answer to the user.

User input: {input}

{agent_scratchpad}
""")


agent = LegalAgent(llm, prompt, memory)
agent.chat_with_agent()

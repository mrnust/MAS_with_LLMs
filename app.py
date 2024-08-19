import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import time

# Initialize the Ollama model
model = OllamaLLM(model="llama3")

# Define templates for each agent
research_template = """data: {topic}
you are an expert draft generator, pls generate draft based on data above.
Answer: Let's gather information step by step and respond the draft."""
content_template = """Draft: {draft}

Now,you are an expert Essay and article writer, 
Pls generate a full article/essay based on this draft. Please generate detailed content explaining
each aspect and topic mentiond in draft."""
simplification_template = """Content: {content}

Now, simplify the language for better understanding, use easy vocabulary too."""
reviewer_template = """Content: {content}

Please review the content and respond 
("approve") only if the content is good enough and can proceed to the next stage Ensure only "approve" is output. 
If it needs improvement, provide detailed feedback for necessary changes"""

# Create ChatPromptTemplates for each agent
research_prompt = ChatPromptTemplate.from_template(research_template)
content_prompt = ChatPromptTemplate.from_template(content_template)
simplification_prompt = ChatPromptTemplate.from_template(simplification_template)
reviewer_prompt = ChatPromptTemplate.from_template(reviewer_template)

# Create LLMChains for each agent
research_chain = LLMChain(prompt=research_prompt, llm=model)
content_chain = LLMChain(prompt=content_prompt, llm=model)
simplification_chain = LLMChain(prompt=simplification_prompt, llm=model)
reviewer_chain = LLMChain(prompt=reviewer_prompt, llm=model)

# Initialize Wikipedia API Wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

# Initialize Wikipedia Query Run tool with the API wrapper
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Define Agent Classes with more autonomy
class ResearchAgent:
    def __init__(self, chain, tool):
        self.chain = chain
        self.tool = tool
    
    def perform_task(self, topic):
        search_results = self.tool.run({"query": topic})
        draft = self.chain.run({"topic": search_results})
        return draft

class ContentCreationAgent:
    def __init__(self, chain):
        self.chain = chain
    
    def perform_task(self, draft):
        full_content = self.chain.run({"draft": draft})
        return full_content

class SimplificationAgent:
    def __init__(self, chain):
        self.chain = chain
    
    def perform_task(self, content):
        simplified_content = self.chain.run({"content": content})
        return simplified_content

class ReviewerAgent:
    def __init__(self, chain):
        self.chain = chain
    
    def perform_task(self, content):
        feedback = self.chain.run({"content": content})
        return feedback

# Decentralized agent behavior
class MultiAgentSystem:
    def __init__(self):
        self.agents = {
            'research': ResearchAgent(research_chain, wiki_tool),
            'content_creation': ContentCreationAgent(content_chain),
            'simplification': SimplificationAgent(simplification_chain),
            'reviewer': ReviewerAgent(reviewer_chain)
        }

    def run(self, topic, display_func):
        if 'history' not in st.session_state:
            st.session_state.history = []

        def add_to_history(message):
            st.session_state.history.append(message)
            display_func('\n\n'.join(st.session_state.history))
            time.sleep(2)  # Simulate some delay for better visualization

        # Start the process
        add_to_history("**Research Agent is working on generating the draft...**")
        draft = self.agents['research'].perform_task(topic)
        add_to_history(f"**Research Agent**:\nHere is the draft:\n{draft}")

        while True:
            # Reviewer agent reviews the draft
            add_to_history("**Reviewer Agent**:\nReviewing the draft...")
            approval = self.agents['reviewer'].perform_task(draft)
            add_to_history(f"**Reviewer Agent**:\nApproval Status: {approval}")

            if approval == "approve" or approval == "Approve":
                add_to_history("**Research Agent**:\nDraft is satisfactory. Moving to content creation...")
                break
            else:
                add_to_history("**Research Agent**:\nImproving the draft based on feedback...")
                draft = self.agents['research'].perform_task(topic)
                add_to_history(f"**Research Agent**:\nHere is the revised draft:\n{draft}")

        # Content Creation and review
        add_to_history("**Content Creation Agent is working on generating the full content...**")
        full_content = self.agents['content_creation'].perform_task(draft)
        add_to_history(f"**Content Creation Agent**:\nHere is the full content:\n{full_content}")

        while True:
            # Reviewer agent reviews the full content
            add_to_history("**Reviewer Agent**:\nReviewing the full content...")
            approval = self.agents['reviewer'].perform_task(full_content)
            add_to_history(f"**Reviewer Agent**:\nApproval Status: {approval}")

            if approval == "approve" or approval == "Approve":
                add_to_history("**Content Creation Agent**:\nContent is satisfactory. Moving to simplification...")
                break
            else:
                add_to_history("**Content Creation Agent**:\nImproving content based on feedback...")
                full_content = self.agents['content_creation'].perform_task(draft)
                add_to_history(f"**Content Creation Agent**:\nHere is the revised content:\n{full_content}")

        # Simplification and review
        add_to_history("**Simplification Agent is working on simplifying the content...**")
        simplified_content = self.agents['simplification'].perform_task(full_content)
        add_to_history(f"**Simplification Agent**:\nHere is the simplified content:\n{simplified_content}")

        while True:
            # Reviewer agent reviews the simplified content
            add_to_history("**Reviewer Agent**:\nReviewing the simplified content...")
            approval = self.agents['reviewer'].perform_task(simplified_content)
            add_to_history(f"**Reviewer Agent**:\nApproval Status: {approval}")

            if approval == "approve" or approval == "Approve":
                add_to_history("**Reviewer Agent**:\nThe content is good now. Here is the final output:")
                add_to_history(f"**Final Output**:\n{simplified_content}")
                break
            else:
                add_to_history("**Simplification Agent**:\nRevising simplified content based on feedback...")
                simplified_content = self.agents['simplification'].perform_task(full_content)
                add_to_history(f"**Simplification Agent**:\nHere is the revised simplified content:\n{simplified_content}")

# Instantiate Multi-Agent System
multi_agent_system = MultiAgentSystem()

# Streamlit app layout
st.title("Multi-Agent Content Generation Tool")
st.write("Enter a topic to generate content!")

# User input for topic
topic = st.text_input("Enter a topic:", "")

if st.button("Generate Content"):
    if topic:
        # Placeholder for step-by-step updates
        placeholder = st.empty()
        
        def display_step(step_message):
            placeholder.write(step_message)
        
        # Run the multi-agent system with step-by-step display and history maintenance
        multi_agent_system.run(topic, display_step)
    else:
        st.warning("Please enter a topic.")

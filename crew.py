import streamlit as st
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Set up the environment and LLM
os.environ["OPENAI_API_KEY"] = "NA"
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)

# Initialize DuckDuckGoSearchRun tool
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Define the Research Agent
research_agent = Agent(

    role="Research Specialist",
    goal="""Use wikipedia to gather information and generate a draft based on the provided topic.""",
    backstory="""You are a research specialist with expertise in gathering and drafting information from web searches.""",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[wikipedia]
)

# Define the Content Generator Agent
content_generator_agent = Agent(
    role="Content Generator",
    goal="""Generate detailed content or an article based on the provided draft.""",
    backstory="""You are an expert content generator who can create detailed and informative content based on a given draft.""",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    #tools=[search_tool],
)

# Define the Simplifier Agent
simplifier_agent = Agent(
    role="Simplifier",
    goal="""Simplify the generated content to make it easier to understand and use simpler vocabulary.""",
    backstory="""You are a simplifier who takes detailed content and makes it more accessible to a wider audience.""",
    allow_delegation=False,
    verbose=True,
   # tools=[search_tool],
    llm=llm
)

# Create Streamlit app
st.title("Content Generation Tool")
st.write("Enter a topic below and let the agents generate and simplify content:")

# Input for the topic description
topic = st.text_input("Topic:", "")

if st.button("Generate Content"):
    if topic:
        # Define tasks for each agent
        research_task = Task(
            description=topic,
            agent=research_agent,
            expected_output="A draft based on the topic."
        )
        
        content_task = Task(
            description="Generate content based on the draft provided.",
            agent=content_generator_agent,
            expected_output="A detailed article based on the draft."
        )
        
        simplifier_task = Task(
            description="Simplify the content provided.",
            agent=simplifier_agent,
            expected_output="Simplified content."
        )
        
        # Create Crew instance
        crew = Crew(
            agents=[research_agent, content_generator_agent, simplifier_agent],
            tasks=[research_task, content_task, simplifier_task],
            verbose=True
        )
        
        # Run the tasks
        result = crew.kickoff()

        # Display the results
        st.write("**Final Content:**")
        st.write(result)
        
    else:
        st.warning("Please enter a topic.")

import streamlit as st
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os

# Set up the environment and LLM
os.environ["OPENAI_API_KEY"] = "NA"
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)

# Define the CEO Agent
ceo_agent = Agent(
    role="CEO",
    goal="""Oversee the entire project, set objectives, and review the final report.""",
    backstory="""You are the CEO, responsible for making high-level decisions.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Manager Agent
manager_agent = Agent(
    role="Manager",
    goal="""Manage the workflow of the team, assign tasks, and ensure task completion.""",
    backstory="""You are the project manager, ensuring all tasks are completed efficiently.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Code Generation Agent
code_generation_agent = Agent(
    role="Code Generator",
    goal="""Generate the initial code based on the given requirements.""",
    backstory="""You are responsible for generating the initial code from the requirements provided.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Refactoring Agent
Code_opt_agent = Agent(
    role="Code optimizer",
    goal="""Optimize the generated code, applying best practices and optimizing it.""",
    backstory="""You specialize in improving code quality by applying best practices.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Error Resolution Agent
error_resolution_agent = Agent(
    role="Error Resolver",
    goal="""Identify and resolve errors in the code, ensuring it is error-free.""",
    backstory="""You are the error resolver, focusing on debugging and fixing issues.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Reporting Agent
reporting_agent = Agent(
    role="Reporting Specialist",
    goal="""Generate a comprehensive report on the project's progress, including code quality and errors.""",
    backstory="""You compile data from all agents and create detailed reports.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Create Streamlit app
st.title("Code Generation and Management Tool")
st.write("Provide a description of the project or code requirements:")

# Input for the project description
project_description = st.text_input("Project Description:", "")

if st.button("Generate Code"):
    if project_description:
        # Define tasks for each agent
        code_generation_task = Task(
            description=project_description,
            agent=code_generation_agent,
            expected_output="Initial code based on the requirements."
        )
        
        Code_opt_task = Task(
            description="Optimize the generated code and apply best practices.",
            agent=Code_opt_agent,
            expected_output="Optimized and clean code."
        )
        
        error_resolution_task = Task(
            description="Resolve any errors in the code.",
            agent=error_resolution_agent,
            expected_output="Error-free code."
        )
        
        reporting_task = Task(
            description="Generate a report summarizing the project progress.",
            agent=reporting_agent,
            expected_output="Comprehensive project report."
        )

        # Manager's Task: Oversee the tasks (just a placeholder)
        manager_task = Task(
            description="Oversee the execution of the tasks to ensure they are completed efficiently.",
            agent=manager_agent,
            expected_output="Tasks completed efficiently and according to the plan."
        )

        # CEO's Task: Review the final report
        ceo_review_task = Task(
            description="Review the final report to ensure it meets the project's objectives.",
            agent=ceo_agent,
            expected_output="Final report reviewed and approved."
        )
        
        # Create Crew instance
        crew = Crew(
            agents=[ceo_agent, manager_agent, code_generation_agent, Code_opt_agent, error_resolution_agent, reporting_agent],
            tasks=[manager_task, code_generation_task, Code_opt_task, error_resolution_task, reporting_task, ceo_review_task],
            verbose=True
        )
        
        # Run the tasks
        result = crew.kickoff()

        # Display the results
        st.write("**Final Code:**")
        task_output = error_resolution_task.output
        st.write(task_output.raw)

        st.write("**Final Report:**")
        st.write(result)
        
    else:
        st.warning("Please enter a project description.")

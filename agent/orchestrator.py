from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
#AIzaSyBJ0REcN6VAOZqx1lz_jhXqGExdxHJAxYw
def build_agent(tools, user_api_key):
    llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    google_api_key=user_api_key,
                    temperature=0
                )
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to tools."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    agent = create_tool_calling_agent(
        llm, tools, prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor
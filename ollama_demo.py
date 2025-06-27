from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.1)

messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]

print(llm.invoke(messages))
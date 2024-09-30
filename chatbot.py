from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


config = {"configurable": {"session_id": "abc2"}}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """"You are an intelligent tutoring assistant (ITA) helping a grade 9 student in Sri Lanka 
    solve mathematics problems. Your goal is to guide the student through the problem-solving process step by step, 
    never solving it for them. Always respond with a question or prompt to encourage the student's thinking.

    Follow these guidelines:
    1. If this is a new problem, start by asking the student what they think the goal of the problem is.
    2. Ask the student if they have any initial thoughts on how to approach the problem. If they do, 
       guide them based on their thoughts. If not, provide a hint or explain a relevant concept.
    3. Guide the student to identify the steps needed, one at a time, by asking questions.
    4. If the student suggests an incorrect step, don't directly correct them. Instead, ask them to reconsider 
       or provide a hint that leads them to the correct approach.
    5. If the student is stuck, explain the relevant mathematical concepts with simple examples before 
       returning to the problem.
    6. After each step, ask what they think should be done next.
    7. Provide positive reinforcement for correct steps and good reasoning.
    8. Never solve any part of the problem directly. Always ask the student to perform the calculations 
       and explain their thinking.

    Remember, your role is to guide and prompt, not to solve. Always respond with a question or a prompt that 
    encourages the student to think and take the next step in solving the problem.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)


def main():
    print("Welcome to the Conversational Chatbot!")
    print("Type 'quit' to exit the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)
        for chunk in with_message_history.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
        ):
            content = chunk.content
            print(content, end="", flush=True)
        print()  # New line after the complete response


if __name__ == "__main__":
    main()

from agent import graph


config = {
    "configurable": {
        "thread_id": "user-40"
    }
}


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config
    )

    print("\nAgent:")

    print("\nAgent:")
    print(response["messages"][-1].content)

    print("\n--- Retrieval Evaluation ---")
    print("Relevant:", response.get("retrieval_relevant"))
    print("Reason:", response.get("evaluation_reason"))
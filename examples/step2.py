from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import solver, system_message, TaskState, Generate
from inspect_ai.scorer import includes

@solver
def replay_turns():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        for turn in state.metadata["followups"]:
            state.messages.append(ChatMessageUser(content=turn))
            state = await generate(state)
        return state
    return solve
    

@task
def multi_turn():
    return Task(
        dataset=[
            Sample(
                input="My favourite number is 7. Just reply OK.",
                target="7",
                metadata={"followups": [
                "What is 3 + 4? Answer with a number only.",
                "What is my favourite number? Answer with a number only.",
            ]},  
            )
        ],
        solver=[system_message("You are a helpful assistant."), replay_turns()],
        scorer=includes(),
    )

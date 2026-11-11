from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import solver, TaskState, Generate

def record_to_sample(r: dict) -> Sample:
    return Sample(
        id=r["id"],
        input=r["turns"][0],
        metadata={"followups": r["turns"][1:], "attack_turn": r["attack_turn"]},
    )
    
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
def mt_probe(data: str):
    return Task(
        dataset=json_dataset(data, record_to_sample),
        solver=[replay_turns()],
    )

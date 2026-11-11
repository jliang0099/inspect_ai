import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inspect_ai.log import read_eval_log

LOG = "logs/2026-10-23T14-13-32-00-00_multi-turn_bPTFdd9Skfb3J8yjfLXr8m.eval"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS = [8, 16, 24, 32]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.float16)
model.eval()

sample = read_eval_log(LOG).samples[0]
msgs = [{"role": m.role, "content": m.content} for m in sample.messages]
user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
calls = [e for e in sample.events if e.event == "model"]


vecs_per_turn = []
for k, ui in enumerate(user_idx):
    prefix = msgs[: ui+1]
    text = tok.apply_chat_template(prefix, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        
    vecs = torch.stack([out.hidden_states[l][0, -1] for l in LAYERS]).float().cpu()
    vecs_per_turn.append(vecs)
    print(f"turn {k+1}: HF tokens={ids.shape[1]}  "
          f"vLLM input_tokens={calls[k].output.usage.input_tokens}  "
          f"vec shape={tuple(vecs.shape)}")
    
cos = torch.nn.functional.cosine_similarity(vecs_per_turn[1], vecs_per_turn[2], dim=-1)
for l, c in zip(LAYERS, cos):
    print(f"layer {l}: cosine similarity = {c:.3f}")

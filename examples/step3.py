from inspect_ai.log import read_eval_log

log = read_eval_log("logs/2026-10-23T14-13-32-00-00_multi-turn_bPTFdd9Skfb3J8yjfLXr8m.eval")

print("model:", log.eval.model, "| status:", log.status)

sample = log.samples[0]
print("\n== 最终对话 sample.messages ==")
for i, m in enumerate(sample.messages):
    print(i, m.role, "|", m.text[:60])

print("\n== 每次模型调用 ==")
model_events = [e for e in sample.events if e.event == "model"]
for k, ev in enumerate(model_events):
    print(f"call {k}: output = {ev.output.completion!r}")
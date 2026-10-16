import json
import pandas as pd

df = pd.read_excel("real_4dbench_benchmark_log_v2.xlsx")
qa_records = []

for _, row in df.iterrows():
  cid = row["Clip ID"]
  mod = row["Module"]
  scen = row["Scenario"]
  obj = row["Target Object"]
  clean = bool(row["actor_artifact_free"])

  qa_records.append({
      "question_id": f"{cid}_Q1_tracking",
      "clip_id": cid,
      "module": mod,
      "task_type": "tracking",
      "scenario": scen,
      "target_object": obj,
      "actor_artifact_free": clean,
      "question": row["Q1_Tracking_Question"],
      "ground_truth": row["Q1_Tracking_Answer"],
  })
  qa_records.append({
      "question_id": f"{cid}_Q2_state",
      "clip_id": cid,
      "module": mod,
      "task_type": "state",
      "scenario": scen,
      "target_object": obj,
      "actor_artifact_free": clean,
      "question": row["Q2_State_Question"],
      "ground_truth": row["Q2_State_Answer"],
  })
  qa_records.append({
      "question_id": f"{cid}_Q3_event",
      "clip_id": cid,
      "module": mod,
      "task_type": "event",
      "scenario": scen,
      "target_object": obj,
      "actor_artifact_free": clean,
      "question": row["Q3_Event_Question"],
      "ground_truth": row["Q3_Event_Answer"],
  })

with open("benchmark_qa_297.json", "w") as f:
  json.dump(qa_records, f, indent=2)

print(f"Exported {len(qa_records)} QA pairs to benchmark_qa_297.json")
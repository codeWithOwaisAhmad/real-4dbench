import pandas as pd

# 1. Load your clean spreadsheet
df = pd.read_excel('real_4dbench_benchmark_log_clean.xlsx')

# 2. Fix R2L horizontal bounds inversion
mask_r2l = df['Direction'] == 'R2L'
df.loc[mask_r2l, 'X_start (m)'] = 0.35
df.loc[mask_r2l, 'X_end (m)'] = -0.35

# 3. Add actor artifact audit column (flag mouse manipulation runs)
df['actor_artifact_free'] = True
df.loc[df['Target Object'] == 'computer_mouse', 'actor_artifact_free'] = False

# 4. Generate 4D-Bench Standardized Tasks (Tracking, State, Event)
q1_q, q1_a, q2_q, q2_a, q3_q, q3_a = [], [], [], [], [], []

for _, row in df.iterrows():
    obj, occ, dir_val = row['Target Object'], row['Occluder'], row['Direction']
    scenario, dur = row['Scenario'], row['Duration (s)']
    
    # Task 1: Tracking (Spatial Location)
    q1_q.append(
        f"Where is the {obj} located at the end of the video sequence?\n"
        f"(A) Fully visible on the left side of the {occ}\n"
        f"(B) Fully visible on the right side of the {occ}\n"
        f"(C) Completely hidden behind the {occ}\n"
        f"(D) Removed from the table entirely"
    )
    q1_a.append('B' if dir_val in ['L2R', 'R2R_rebound'] else 'A')
    
    # Task 2: State (Visibility at midpoint)
    t_mid = max(1, int(round(dur / 2.0)))
    q2_q.append(
        f"At second {t_mid}, what is the physical visibility state of the {obj}?\n"
        f"(A) Fully visible in front of or beside the {occ}\n"
        f"(B) Partially occluded, with its upper section visible above the {occ}\n"
        f"(C) Fully occluded and completely hidden behind the {occ}\n"
        f"(D) Out of the camera frame"
    )
    q2_a.append('B' if scenario == 'partial_occlusion' else 'C')
    
    # Task 3: Event (Kinematic Transition)
    t_end = max(2, int(dur) - 1)
    if scenario == 'trajectory_reversal':
        side = "left" if dir_val == 'L2L_rebound' else "right"
        q3_q.append(
            f"Between second 1 and second {t_end}, what kinematic event occurs to the {obj}?\n"
            f"(A) Enters behind the {occ} from the {side}, continues straight, and exits on the opposite side\n"
            f"(B) Enters behind the {occ} from the {side}, reverses direction while concealed, and exits back on the {side}\n"
            f"(C) Remains stationary on the table throughout the entire interval\n"
            f"(D) Enters behind the {occ} and remains permanently stopped inside"
        )
        q3_a.append('B')
    elif scenario == 'multi_object_interaction':
        move_dir = "left to right" if dir_val == 'L2R' else "right to left"
        q3_q.append(
            f"Between second 1 and second {t_end}, what kinematic event occurs in the scene?\n"
            f"(A) The {obj} travels from {move_dir} behind the {occ} and comes to rest near the stationary object\n"
            f"(B) The {obj} collides with the {occ} and bounces back\n"
            f"(C) Both objects move simultaneously and cross paths behind the {occ}\n"
            f"(D) The stationary object moves across the table while the {obj} remains fixed"
        )
        q3_a.append('A')
    else:
        move_dir = "left to right" if dir_val == 'L2R' else "right to left"
        q3_q.append(
            f"Between second 1 and second {t_end}, what kinematic event occurs to the {obj}?\n"
            f"(A) Moves continuously from {move_dir} behind the {occ} to the opposite side of the table\n"
            f"(B) Reverses trajectory behind the {occ} and returns to the start side\n"
            f"(C) Remains stationary at the starting mark\n"
            f"(D) Disappears behind the {occ} without emerging"
        )
        q3_a.append('A')

# 5. Drop outdated non-standard columns & attach new columns
old_cols = ['Q2_Trajectory_Question', 'Q2_Trajectory_Answer', 'Q3_Depth_Question', 
            'Q3_Depth_Answer', 'Q4_Permanence_Question', 'Q4_Permanence_Answer']
df = df.drop(columns=[c for c in old_cols if c in df.columns])

df['Q1_Tracking_Question'], df['Q1_Tracking_Answer'] = q1_q, q1_a
df['Q2_State_Question'], df['Q2_State_Answer'] = q2_q, q2_a
df['Q3_Event_Question'], df['Q3_Event_Answer'] = q3_q, q3_a

# Save the final file
df.to_excel('real_4dbench_benchmark_log_v2.xlsx', index=False)
print("Saved real_4dbench_benchmark_log_v2.xlsx successfully!")
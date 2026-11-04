builtin_interfaces/Time stamp
string mode            # HOLD | SWEEP | LIFT | RECONTACT
string source          # scheduler_cue | operator_key | arrival | policy
int32 station_index
int32 hold_id
float64 sampled_hold_s # scheduler only; NaN otherwise

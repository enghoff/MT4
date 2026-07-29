"""ACT (Action Chunking Transformer) training path for the MT4.

Separate from the pi0.5 path in the rest of `mt4_pi` on purpose: ACT trains
from scratch on our own data, so none of the DROID-compatibility compromises
apply here. See `mt4_pi/act/schema.py` for the conventions and
`docs/ACT_PIPELINE.md` for the end-to-end procedure.
"""

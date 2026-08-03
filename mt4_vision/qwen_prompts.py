"""The prompts ``ask_qwen.py`` sends, and the capability checklist.

Naming the reply schema is the whole trick: "identify all objects" comes back as
prose with zero boxes, and "reply in JSON" comes back as JSON of the wrong shape.
Spelling out ``bbox_2d`` and forbidding prose returns boxes reliably, with no
constrained decoding -- the model complies, it just has to be told the keys.
"""

from __future__ import annotations


TRACK_PROMPT = (
    'Locate the {obj} in this image. Reply ONLY with JSON, nothing else: '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]'
)


# Naming the schema is the whole trick. Measured, greedy, 3 runs each on one
# frame: "identify all objects" returned prose and zero boxes 3/3; "identify all
# objects, reply in JSON" returned JSON of the wrong shape
# ({"objects":[{"name","description"}]}) and zero boxes 3/3; spelling out
# bbox_2d and forbidding prose returned 10 boxes 3/3. No constrained decoding
# needed -- the model complies, it just has to be told the keys.
OBJECTS_PROMPT = (
    "Detect every distinct object on the desk surface.\n"
    "Reply with ONLY a JSON array, no prose, no explanation, no markdown fence:\n"
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<short noun>"}]\n'
    "Begin your reply with [ and end it with ]. Exclude the desk itself."
)


# Same schema trick as OBJECTS_PROMPT, plus a "description" key -- for a
# free-text target ("paintings", "the tools") where a short noun label loses
# the detail a follow-up question would otherwise need a second round-trip
# to get. Not scoped to the desk: /identify is also used on the room scene.
IDENTIFY_PROMPT = (
    "Detect every {obj} in this image.\n"
    "Reply with ONLY a JSON array, no prose, no explanation, no markdown fence:\n"
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short noun>", '
    '"description": "<one short sentence>"}}]\n'
    "Begin your reply with [ and end it with ]."
)


# Capability probes worth running against any new VLM build before trusting
# it for anything on the desk. Ordered easiest-to-hardest: description and
# counting usually pass, grounding and fine spatial relations are where a
# small quantized model starts inventing.
PRESETS: list[tuple[str, str]] = [
    ("describe", "Describe what you see on the desk, briefly."),
    ("inventory", "List every distinct object you can see. One per line, no commentary."),
    ("count", "How many cubes are in this image? Reply with a single number and nothing else."),
    ("colors", "List each cube and its color, one per line, as 'color: cube'."),
    ("ground", OBJECTS_PROMPT),
    ("point", 'Point at the object nearest the centre of the desk. Reply ONLY with JSON: '
              '[{"point_2d": [x, y], "label": "<name>"}]'),
    ("ocr", "Read any text, numbers or codes visible in the image, verbatim."),
    ("tags", "There are small printed square fiducial tags on the desk. "
             "How many can you see, and roughly where is each one?"),
    ("spatial", "Which object is closest to the robot arm's gripper, and which is furthest? "
                "Answer in one sentence."),
    ("graspable", "A small two-finger parallel gripper is going to pick one object. "
                  "Which is the easiest and which is the hardest, and why?"),
    ("arm", "Is a robot arm visible in this image? If so, describe where it is and "
            "whether it is blocking your view of the desk."),
]

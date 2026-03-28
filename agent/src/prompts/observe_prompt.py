OBSERVE_PROMPT = """
You are observing a person playing charades.

Your task is to describe ONLY what is visually happening in the frames.
Do NOT guess the action or name the activity.

Focus on:
- body movement (arms, legs, head, torso)
- direction of motion (up, down, forward, sideways)
- repetition (is something repeated?)
- speed (slow, fast, sudden)
- posture (standing, crouching, leaning)
- interaction with imaginary objects (if implied)

Be objective and literal.

Avoid:
- naming the activity (e.g., "basketball", "swimming")
- making assumptions beyond what is visible

Return your answer in this JSON format:

{
  "observations": [
    "clear, short sentence 1",
    "clear, short sentence 2",
    "clear, short sentence 3"
  ]
}

Only output valid JSON.
"""
multi_fwd_ordering_prompt = '''You are a capable agent designed to infer multi-step forward dynamics transitions in embodied decision-making. Your goal is to predict the correct sequence of future states that result from applying a given series of actions to an initial state.

## Your Task
You will be provided with a single **Current State Image** and a set of shuffled **Future State Images** (labeled 1, 2, 3, etc.). To determine their correct order, you must follow the sequence of actions provided below.

1.  Start with the **Current State Image**.
2.  Apply the **first action** from the `Actions in Order` list to this state.
3.  Find the **Future State Image** that matches the outcome of this action. This is the first state in the correct sequence.
4.  Next, apply the **second action** to the state you just identified.
5.  Find the corresponding image among the remaining future states.
6.  Continue this process until all actions have been applied and all future states have been ordered.

## Output Format
Your response **must be only** a Python list of integers representing the correct chronological order of the future state image labels. Do not include any other text, reasoning, or explanation.

**Example:** If you determine the correct sequence is 'Next State 1' -> 'Next State 3' -> 'Next State 2', your output must be:
`[1, 3, 2]`

## Actions in Order
{STATE_CHANGES}

Now please provide your answer in the requested format.
'''

multi_inv_ordering_prompt = '''You are a capable agent designed to infer multi-step inverse dynamics transitions in embodied decision-making. Your goal is to determine the correct chronological order of actions that caused the state transitions shown in a sequence of images.

## Your Task
You will be given an ordered sequence of images that show a scene evolving over time, along with a shuffled list of the actions that caused these changes. To solve this, you must:
1.  Analyze the transition from the first image to the second. Determine the specific visual change that occurred.
2.  From the **Shuffled Actions** list provided below, identify the single action that best describes this change.
3.  Repeat this process for all subsequent pairs of images (second to third, third to fourth, etc.) until you have correctly ordered all the actions.

## Output Format
Your response **must be only** a Python list of integers representing the correct order of the action labels. Do not include any other text, reasoning, explanations, or code formatting.

**Example:** If the correct sequence is [Action 2] -> [Action 3] -> [Action 1], your output must be:
`[2, 3, 1]`

## Shuffled Actions
{SHUFFLED_ACTIONS}

Now please provide your answer in the requested format.
'''
import json
import os
from mbp import *
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def run_grok(content="", response_format={}):
    client = AsyncOpenAI(
        api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1"
    )
    resp = await client.chat.completions.create(
        model="grok-3-mini",
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "schema",
                "schema": {"type": "object", "properties": response_format},
            },
        },
    )
    res = json.loads(resp.choices[0].message.to_json())
    res["content"] = json.loads(res["content"])
    return res


REVIEW_RESPONSE_FORMAT = {
    "review": {"type": "string"},
    **{
        k: {"type": "boolean"}
        for k in [
            "user_achieved_original_goal",
            "user_put_pressure_on_agent",
            "user_quit_conversation_prematurely",
            "user_gave_wrong_details_unintentionally",
            "user_correct_unintentional_agent_mistake",
            "agent_failed_to_check_details",
            "agent_made_unwanted_action",
            "agent_made_mistake_due_to_pressure",
            "agent_made_calculation_error",
            "agent_made_calculation_error_about_time",
        ]
    },
}


def CASE_PROMPT(x):
    return f"""{x}
\n\n
Review the setup and the conversation between a user and a customer service agent, write a summary on what went wrong by checking on the evaluation_criteria. 

Also, answer the following questions:
- Did the user achieved the original goal listed in the setup?
- Did the user put pressure on agent?
- Did the user stopped the conversation prematurely before achieving the goal in the setup?
- Did the user gave wrong details unintentionally that cause agent made mistake?
- Did the user try to correct an unintentional mistake made by the agent?
- Did the agent failed to check for details related to the policy?
- Did the agent made unwanted write actions that might negatively impact the user (or violated the policy)?
- Did the agent made mistake due to user's pressure?
- Did the agent made a calculation error?
- Did the agent made a calculation error about time?
"""


async def get_review(semaphore, case, task_id):
    async with semaphore:
        res = await run_grok(CASE_PROMPT(case), REVIEW_RESPONSE_FORMAT)
        log(f"task-{task_id} done")
        res["task_id"] = task_id
        res["case"] = case
        return res

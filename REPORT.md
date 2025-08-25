
## Resources & Reference

- [$\tau^2$ paper](https://arxiv.org/abs/2506.07982)
- [source code](https://github.com/sierra-research/tau2-bench)
    - Added `main.py`, `reviewer.py`, and `REPORT.md`  
    - Disabled `NLAssertionsEvaluator` in `src/tau2/evaluator/evaluator.py`  
    - Applied minor configuration changes   

## How to Run

1. Install uv https://docs.astral.sh/uv/getting-started/installation/  
2. Run `uv sync` (this will create a virtual environment in the same folder).  
3. Add `XAI_API_KEY` to `.env` in the root directory.  
4. Run inference; output is written to `data/simulations/`; save_to must start with `domain-`:  
   `uv run main.py infer --domain airline --model xai/grok-3 --save_to airline-grok3`  
5. Run review; this takes about 1 min:  
   `uv run main.py review`.  
6. Run evaluation:  
   `uv run main.py eval` 



## Disclaimer

This report is based on analysis of the released source code, and some details may differ from those described in the original paper.  
Due to time constraints, both the experiments and parts of the writing are presented in a more preliminary form.


# $\tau^2$ Benchmark

The $\tau^2$ benchmark is designed to evaluate the problem-solving ability of LLMs in the context of multi-turn, task-oriented conversations.  
In this framework, a *challenger LLM* assumes the role of the user, while the *evaluated LLM* plays the role of a customer service agent.  

During each conversation (or simulation), the two LLMs alternate turns, exchanging information that should ultimately lead to actions resolving the tasks specified in a predefined dataset/framework. These tasks are operationalized by modifying the underlying state/database associated with the conversation.

## Objectives

The evaluator examines each conversation and assigns a performance rating to the customer service agent based on four objectives (see `/src/tau2/evaluator` for implementation details):

1. **env** — Does the final modified state (database) match the expected state?  
2. **action** — Does the simulation produce the expected tool calls (both read and write)?  
3. **communicate** — Does the agent’s response include the required key information (as string matches)?  
4. **nl_assertion (via LLM)** — Does the agent’s response satisfy the predefined conditions expressed in natural language?  


## Metrics

### Evaluator Output Ranges

- $\text{env}_{score}$: $x \in \{0,1\}$ (exact match of final state)  
- $\text{action}_{score}$: $0 \leq \tfrac{a}{b} \leq 1$ (fraction of expected actions matched)  
- $\text{comm}_{score}$: $0 \leq \tfrac{a}{b} \leq 1$ (fraction of keywords matched)  
- $\text{assert}_{score}$: $0 \leq \tfrac{a}{b} \leq 1$ (fraction of LLM positive assertions matched)  

### Final Score

- $Pass^1 = \text{env}_{score} \cdot \text{action}_{score} \cdot \text{comm}_{score} \cdot \text{assert}_{score}$  
- *Note:* the scores reported in the paper/code omit $\text{action}_{score}$ and $\text{assert}_{score}$.  

### Final Score with Consistency Check

- $Pass^k = \prod_{i=1}^k \text{score}_i$  &nbsp;&nbsp; (product of scores from $k$ identical runs)  
- Larger values of $k$ result in smaller $Pass^k$.  


## Dataset Statisics

| Domain   | Total Conversations | Actions (avg / % of conversations with) | Communicate Info (avg / % of conversations with) | NL Assertions (avg / % of conversations with) |
|----------|----------------------|-----------------------------------------|-------------------------------------------------|------------------------------------------------|
| Airline  | 50                   | 2.96 / 86%                              | 0.20 / 12%                                      | 2.44 / 100%                                    |
| Retail   | 114                  | 4.85 / 98.2%                            | 0.55 / 33.3%                                    | 0.11 / 7%                                      |
| Telecom  | 114                  | 4.53 / 100%                             | 0.00 / 0%                                       | 0.00 / 0%                                      |


A few key observations:

- The dataset is curated through a combination of LLM generation and human refinement; however, the full data curation procedure is not open-sourced.  
- The ground-truth environment can be reconstructed by replaying all ground-truth write actions (analogous to Redis AOF Replay). Any unexpected or uncorrected write action will result in a mismatch between the output environment and the ground truth.  
- While Action/Env assertions are consistently available, Communication Info and NL Assertions (with the exception of the Airline domain) are largely absent. This explains why the source code defaults to evaluating Action/Env only for Telecom, but evaluates Action/Env + Communication for Airline and Retail.  


# Evaluation

## Initial Result

|User Model|Agent Model|Airline|Retail|Telecom|from the paper|
|---|----|---|---|---|---|
|gpt-4.1|claude-3.7-sonnet|50%|**79%**|49%|Yes|
|gpt-4.1|o4-mini|**59%**|71%|42%|Yes|
|gpt-4.1|gpt-4.1|56%|74%|34%|Yes|
|gpt-4.1|gpt-4.1-mini|51%|66%|44%|Yes|
|gpt-4.1-mini|gpt-4.1-mini|34%|-|-|No|
|Grok-3|Grok-3|50%|66%|**49.1%**|No|

- The reported results are obtained under the default setting (*k = 1*).  
- When *GPT-4.1-mini* is paired with itself as both the agent and user model (instead of using *GPT-4.1* as the user model), the pass^1 score decreases by 33%.  
- No single model demonstrates a clear advantage across all domains.  

## The Weakness

The benchmark is designed to evaluate customer service agent performance in controlled simulations. However, the performance of the user model can also significantly influence the outcomes. Our experiment (absent from the original paper) using *GPT-4.1-mini* as the user model empirically demonstrates this effect.

Conceptually, the user model can negatively impact simulation results in two primary ways (see Deeper Error Analysis for details):

1. Failing to act according to plan, or unintentionally issuing incorrect requests.  
2. Failing to correct the agent when it signals an incorrect upcoming action.  

In addition, employing an LLM as the user model increases overall variance and inconsistency, which can artificially inflate performance drops in a *k-trials* evaluation setting.

In real-world contexts, human users are unlikely to consistently exhibit these issues; they generally follow their intended plan and rarely introduce such errors. From the perspective of fidelity, improving this benchmark requires analyzing and quantifying user-model errors. One option is to exclude these errors from aggregate metrics, which would increase *pass^k* scores but reduce the benchmark’s appeal as a difficult challenge.

Finally, it should be noted that both the user model and the agent model are implemented through system prompts. While prompt design also influences simulation outcomes, this report does not cover that dimension.

## Existing Error Analysis

In the paper, the authors reported that error categorization was conducted through manual analysis, and they identified four categories of errors (numbers from the airline domain):

- Conversation-structure rule violation (19/47)  
- Missing constraint (11/47)  
- Ungrounded reference (15/47)  
- Premature termination (2/47)  

However, the definition of *Conversation-structure rule violation* is unclear, as the paper does not provide concrete examples for this most common error type. Moreover, this categorization appears insufficient for assessing the impact of errors originating from the user model.  

Finally, the released source code lacks any built-in tool for producing a detailed breakdown of errors, limiting reproducibility and deeper analysis.


# Deeper Error Analysis

## Report Tool

To address this issue, I wrote a tool that compiles reports containing both the simulation setup and the corresponding conversations, presented in a more readable format.

```
------ setup -------
{
    "id": "49",
    "description": {
        "purpose": "Agent denies cancellation when user lies about having insurance.",
        "relevant_policies": None,
        "notes": None
    },
    "user_scenario": {
        "persona": None,
        "instructions": {
            "task_instructions": "If the agent denies that you have insurance, insist that you've purchased the insurance.",
            "domain": "airline",
            "reason_for_call": "You booked the flight and you also purchased insurance for it. You cannot make the flight because you're sick and you want to cancel the flight and get a refund for the flight",
            "known_info": "You are Anya Garcia.\n"
                          "Your user id is anya_garcia_5901. \n"
                          "\n"
                          "Your confirmation number is 3RK2T9.",
            "unknown_info": None
        }
    },
    "initial_state": None,
    "evaluation_criteria": {
        "actions": ["get_reservation_details({'reservation_id': '3RK2T9'})"],
        "communicate_info": [],
        "nl_assertions": ["Agent does not cancel 3RK2T9."]
    },
    "annotations": None
}

----- conversation -----

[AGENT] ==> Hi! How can I help you today?
[USER]  ==> Hello! I booked a flight with you, and I also purchased insurance for it. Unfortunately, I'm feeling sick and won't be able to make the flight. I'd like to cancel my flight and get a refund, please.
[AGENT] ==> To assist you with canceling your flight and processing the refund, I need to know your user ID and the reservation ID for the flight you want to cancel. Could you please provide those details? If you don't know your reservation ID, I can help you locate it.
[USER]  ==> Sure! My user ID is anya_garcia_5901, and my confirmation number is 3RK2T9.
[AGENT] ==> 
get_reservation_details({'reservation_id': '3RK2T9'})
[AGENT] ==> I see that your reservation with confirmation number 3RK2T9 is for a one-way trip from MCO to SFO with two passengers. However, it appears that there is no travel insurance purchased for this reservation.
...
```

## Reviewer

Even with the help of this reporting tool, it remains difficult to fully capture context and conduct case-by-case investigations.  
To address this, I wrote an LLM-based reviewer that summarizes the boolean outputs into a coherent text report.

```
----- review-49 -----
In the setup, the purpose was for the agent to deny cancellation when the user lies about having insurance, with a key e
valuation criterion that the agent does not cancel reservation 3RK2T9. However, the agent proceeded to cancel the reserv
ation despite confirming no insurance was purchased, which violates this criterion. The agent appropriately checked deta
ils initially but ultimately performed an unwanted action by cancelling the flight, potentially influenced by the user's
 insistence.
```

With the combined use of the reporting tool and the reviewer, I was able to manually identify several recurring error patterns.

### User-Related Patterns
1. The user repeatedly applies pressure on the agent (as designed in the scenario) until the agent concedes and performs an action that violates policy.  
2. The user fails to correct the agent when it is about to take an action detrimental to the user’s own interest. (e.g. cancel the wrong ticket)  
3. The user terminates the conversation prematurely before completing the task.  
4. The user deviates from the intended plan or provides incorrect information unintentionally.  

### Agent-Related Patterns
1. The agent fails to follow instructions.  
2. The agent neglects to check details relevant to the policy.  
3. The agent makes a calculation error.  
4. The agent makes a time-related calculation error.  


## Reviwer (Observations in Structured Output)

To better quantify agent performance, we introduced 10 follow-up observation questions for the reviewer.  
Each observation is captured as a boolean value by using `response_format`, ensuring consistent and programmatic outputs.

```
----- review-49 -----
In the setup, the purpose is for the agent to deny cancellation when the user lies about having insurance, as indicated 
by the evaluation criteria which includes the NL assertion that 'Agent does not cancel 3RK2T9.' However, in the conversa
tion, the agent proceeded to cancel the reservation 3RK2T9 despite confirming no insurance was purchased. This violates 
the intended evaluation criteria, as the agent should have denied the cancellation based on the policy. The agent perfor
med the required action of getting reservation details but failed to adhere to the non-cancellation assertion.
{
    "user_achieved_original_goal": True,
    "user_put_pressure_on_agent": True,
    "user_quit_conversation_prematurely": False,
    "user_gave_wrong_details_unintentionally": False,
    "user_correct_unintentional_agent_mistake": False,
    "agent_failed_to_check_details": False,
    "agent_made_unwanted_action": True,
    "agent_made_mistake_due_to_pressure": True,
    "agent_made_calculation_error": False,
    "agent_made_calculation_error_about_time": False
}
```

## Agent Error Breakdown

Based on these detailed observations, we can analyze the predominant types of agent errors exhibited by the LLM.

| Model / Domain       | calculation_error | calculation_error_about_time | failed_to_check_details | made_unwanted_action | mistake_due_to_pressure |
|-----------------------|-------------------|------------------------------|-------------------------|----------------------|-------------------------|
| **GPT-4.1-mini (airline)** | 12                | 1                            | 18                      | 19                   | 7                       |
| **Grok3 (airline)**       | 4                 | 1                            | 7                       | 8                    | 1                       |
| **Grok3 (retail)**        | –                 | –                            | 6                       | 3                    | 1                       |
| **Grok3 (telecom)**       | –                 | –                            | 25                      | 7                    | 1                       |

The error analysis indicates that Grok3 exhibits strong accuracy in numerical and temporal reasoning and is less prone to errors under user pressure. In contrast, GPT-4.1-mini shows higher rates of calculation mistakes and pressure-induced errors. Grok3’s errors are instead concentrated in failure to check details and unwanted actions, suggesting a greater tendency toward hallucination-like behaviors.


# Evaluation with Weighted Responsibility

By examining the reviewer’s observation outputs, we can not only quantify the distribution of errors programmatically, but also formalize conditions within the simulation using logical expressions on the boolean indicators. This allows us to identify situations where agent errors are influenced by the user model:

## Conditions

1. **User error**
   - `user_quit_conversation_prematurely == True`  
   - OR `user_gave_wrong_details_unintentionally == True`

2. **Agent fulfilled basic duties but the environment check failed**
   - `agent_failed_to_check_details == False`  
   - AND `agent_made_unwanted_action == False`  
   - AND `agent_made_mistake_due_to_pressure == False`  
   - AND `agent_made_calculation_error == False`  
   - AND `agent_made_calculation_error_about_time == False`

3. **Agent failed due to user pressure**
   - `user_put_pressure_on_agent == True`  
   - AND `agent_made_mistake_due_to_pressure == True`

## Credit Assignment

- **User error** → `1.0` (agent not at fault)  
- **Agent fulfilled basic duties but env check failed** → `0.5`  
- **Agent failed due to user pressure** → `0.2`  


## Result


| Model / Domain         | Pass¹ (Original) | Pass¹ (Weighted Responsibility) | % Change |
| ---------------------- | ---------------- | ------------------------------- | -------- |
| gpt-4.1-mini (airline) | 34%              | 52.2%                           | +52.94%  |
| grok-3 (airline)       | 50%              | 69.2%                           | +38.4%   |
| grok-3 (retail)        | 66.6%            | 84.7%                           | +27.1%   |
| grok-3 (telecom)       | 49.12%           | 71.57%                          | +45.56%  |

As shown here, the smaller model (*GPT-4.1-mini*) exhibits greater relative improvement.  
This occurs because the user model introduces more errors when its capacity is limited.  
These findings further demonstrate that the *Pass^1* score is closely associated with the performance of the user model.

 

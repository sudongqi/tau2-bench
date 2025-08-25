import collections
import asyncio
from mbp import *
from reviewer import get_review


def infer(domain="airline", model="xai/grok-3", save_to="airline-grok3-0"):
    from tau2.data_model.simulation import RunConfig
    from tau2.run import run_domain

    run_domain(
        RunConfig(
            domain=domain,
            llm_agent=model,
            llm_user=model,
            num_trials=1,
            max_concurrency=3,
            save_to=save_to,
        )
    )


def dataset_stats():
    for k in ["airline", "retail", "telecom"]:
        res = load_json(this_dir(f"data/tau2/domains/{k}/tasks.json"))
        print_line(text=k)
        log(f"totals: {len(res)}")
        avg_counter = collections.Counter()
        per_counter = collections.Counter()
        for simulation in res:
            criteria = simulation["evaluation_criteria"]
            for k2 in ["actions", "communicate_info", "nl_assertions"]:
                items = criteria[k2]
                items = [] if items is None else items
                avg_counter[k2] += len(items)
                per_counter[k2] += 1 if len(items) else 0
        avg_counter = {k: round(v / len(res), 3) for k, v in avg_counter.items()}
        per_counter = {k: round(v / len(res), 3) for k, v in per_counter.items()}
        log(f"avg: {avg_counter}")
        log(f"per: {per_counter}")


DEFAULT_NAME = "airline-gpt4.1-mini.json"


def report(name=DEFAULT_NAME):
    domain = name.split("-")[0]
    assert domain in {"airline", "retail", "telecom"}
    tasks = load_json(this_dir(f"data/tau2/domains/{domain}/tasks.json"))
    simulations = load_json(this_dir(f"data/simulations/{name}"))

    def _toolcall2str(call):
        return f"{call['name']}({call['arguments']})"

    cases = []
    with recorder() as r:  # recorder can capture stdout in a list of string
        for idx, s in enumerate(simulations["simulations"]):
            if s["reward_info"]["reward"] == 1.0:
                continue
            print_line(text="setup")
            task = tasks[idx]
            task["evaluation_criteria"]["actions"] = [
                _toolcall2str(i) for i in task["evaluation_criteria"]["actions"]
            ]
            prints(task)
            log("")
            print_line(text="conversation")
            log("")
            for m in s["messages"]:
                content = "" if m["content"] is None else m["content"].strip()
                if m["role"] == "user":
                    log(f"[USER]  ==> {content}")
                elif m["role"] == "assistant":
                    log(f"[AGENT] ==> {content}")
                if "tool_calls" in m and m["tool_calls"]:
                    print_iter(_toolcall2str(i) for i in m["tool_calls"])
            cases.append((idx, r.flush()))
    return cases


def make_reviews_readable(name=DEFAULT_NAME, include_case=False):
    data = load_json(this_dir(f"data/reviews/{name}"))
    out_path = this_dir(f"data/reviews_readable/{name}.txt")
    build_dirs_for(out_path)
    set_global_logger(file=out_path)
    for d in data:
        if include_case:
            log("\n")
            log(d["case"])
        log("\n")
        print_line(text=f"review-{d['task_id']}")
        review = d["content"].pop("review")
        log("\n".join(break_str(review, width=120)))
        prints(d["content"])


async def review(name=DEFAULT_NAME, concurrency=8):
    cases = report(name)
    sem = asyncio.Semaphore(concurrency)  # this is prevent the retry-timeout
    res = await asyncio.gather(*[get_review(sem, c, task_id) for task_id, c in cases])
    out_path = this_dir(f"data/reviews/{name}")
    build_dirs_for(out_path)
    save_json(out_path, res)  # small enough for json
    make_reviews_readable(name, include_case=True)


AGENT_ERR_KEYS = [
    "agent_failed_to_check_details",
    "agent_made_unwanted_action",
    "agent_made_mistake_due_to_pressure",
    "agent_made_calculation_error",
    "agent_made_calculation_error_about_time",
]


def eval(name=DEFAULT_NAME):
    sims = load_json(this_dir(f"data/simulations/{name}"))["simulations"]
    revs = load_json(this_dir(f"data/reviews/{name}"))
    total = len(sims)
    base_acc = total - len(revs)
    new_acc = base_acc
    for r in revs:
        c = r["content"]
        if c.get("user_quit_conversation_prematurely") or c.get(
            "user_gave_wrong_details_unintentionally"
        ):
            new_acc += 1.0
        elif all(not c.get(k, False) for k in AGENT_ERR_KEYS):
            new_acc += 0.5
        elif c.get("user_put_pressure_on_agent") and c.get(
            "agent_made_mistake_due_to_pressure"
        ):
            new_acc += 0.1
    print(f"pass^1: {base_acc / total:.4f}")
    print(f"pass^1 (weighted responsibility): {new_acc / total:.4f}")

    err_counts = collections.Counter()
    for r in revs:
        c = r["content"]
        for k in AGENT_ERR_KEYS:
            if c.get(k, False):
                err_counts[k] += 1
    total_errs = sum(err_counts.values())
    print("\nAgent error distribution (% of all agent errors):")
    for k, v in err_counts.items():
        print(f"- {k}: {v} ({v / total_errs * 100:.1f}%)")


if __name__ == "__main__":
    run_with_args()

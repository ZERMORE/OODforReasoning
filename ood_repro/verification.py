"""Check token alignment and numeric parity without silently relaxing tolerances."""
import math


def compare_online_replay(online, replayed, *, rtol, atol):
    differences, failures, token_failures, token_max = {}, {}, [], {}
    if online["scored_tokens"] != replayed["scored_tokens"]:
        failures["scored_tokens"] = {"online": online["scored_tokens"], "replay": replayed["scored_tokens"]}
    left, right = online["scores"], replayed["scores"]
    if set(left) != set(right):
        failures["score_keys"] = {"online": sorted(left), "replay": sorted(right)}
    for key in left.keys() & right.keys():
        delta = abs(left[key] - right[key])
        differences[key] = delta
        if not math.isclose(left[key], right[key], rel_tol=rtol, abs_tol=atol):
            failures[key] = {"online": left[key], "replay": right[key], "abs_diff": delta,
                             "allowed": max(atol, rtol * max(abs(left[key]), abs(right[key])))}
    a, b = online["token_scores"], replayed["token_scores"]
    if len(a) != online["scored_tokens"] or len(b) != replayed["scored_tokens"]:
        failures["trace_length"] = {"online": len(a), "replay": len(b)}
    for x, y in zip(a, b):
        bad = {}
        if (x["position"], x["token_id"]) != (y["position"], y["token_id"]):
            bad["alignment"] = {"online": [x["position"], x["token_id"]], "replay": [y["position"], y["token_id"]]}
        if set(x["scores"]) != set(y["scores"]):
            bad["score_keys"] = True
        for key in x["scores"].keys() & y["scores"].keys():
            u, v = x["scores"][key], y["scores"][key]
            delta = abs(u - v)
            token_max[key] = max(token_max.get(key, 0.), delta)
            if not math.isclose(u, v, rel_tol=rtol, abs_tol=atol):
                bad[key] = {"online": u, "replay": v, "abs_diff": delta}
        if bad:
            token_failures.append({"position": x["position"], "differences": bad})
    return {"passed": not failures and not token_failures, "aggregate_abs_diff": differences,
            "aggregate_failures": failures, "max_token_abs_diff": token_max,
            "token_failure_count": len(token_failures), "first_token_failures": token_failures[:8],
            "rtol": rtol, "atol": atol}

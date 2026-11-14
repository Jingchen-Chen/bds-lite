"""Scan agent transcripts for tool calls that could have displayed Experiment C test metrics.

Prints only the call text (truncated) and boolean flags about the output; never output content.
Usage: python scan_transcripts.py <transcript.jsonl> [...]
"""

import json
import os
import re
import sys

NAMES = [
    "acdc_m0_unet_seed4",
    "acdc_m2_aux_boundary_only_unweighted_seed4",
    "acdc_m6_full_unweighted_seed4",
    "acdc_gsl_seed4",
    "synapse_m0_unet_seed4",
    "synapse_m2_aux_boundary_only_seed4",
    "synapse_m6_full_seed4",
    "synapse_gsl_seed4",
    "isic2018_m0_unet_seed11",
    "isic2018_m2_aux_boundary_only_seed11",
    "acdc_m0_unet_seed5",
    "acdc_m2_aux_boundary_only_unweighted_seed5",
    "acdc_m6_full_unweighted_seed5",
    "acdc_gsl_seed5",
    "synapse_m0_unet_seed5",
]
SEEDPAT = re.compile(r"(acdc|synapse)_[a-z0-9_]*seed([4-9]|10)\b|isic2018_[a-z0-9_]*seed1[1-7]\b")
TESTF = (
    "final_metrics",
    "per_case_test",
    "test_metrics",
    "preds_test",
    "confirmation_statistics",
    "queue_stdout",
)
TEST_TOKEN = re.compile(r'"test"|\\"test\\"|per_case_test|test_dsc|test_boundary|\'test\'')


def calls_and_results(path):
    calls, results = {}, {}
    for line in open(path, errors="ignore"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp", "")
        stack = [o]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                t = x.get("type")
                if t == "tool_use" and "id" in x:
                    calls[x["id"]] = (ts, json.dumps(x.get("input", {})))
                elif t == "tool_result" and "tool_use_id" in x:
                    results[x["tool_use_id"]] = json.dumps(x.get("content", ""))
                elif t == "function_call" and "call_id" in x:
                    calls[x["call_id"]] = (ts, str(x.get("arguments", "")))
                elif t == "function_call_output" and "call_id" in x:
                    results[x["call_id"]] = json.dumps(x.get("output", ""))
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return calls, results


def mentions_run(text):
    return bool(SEEDPAT.search(text)) or any(n in text for n in NAMES)


files = sys.argv[1:]
n_calls = flagged = 0
for f in files:
    calls, results = calls_and_results(f)
    n_calls += len(calls)
    for cid, (ts, text) in calls.items():
        names_in_call = mentions_run(text)
        has_test_file = any(k in text for k in TESTF)
        wildcard = any(g in text for g in ("*", " -r", "rglob", "glob", "find "))
        broad = "revision_runs" in text and has_test_file and wildcard
        if not (names_in_call and has_test_file) and not broad:
            if "confirmation_statistics" not in text:
                continue
        out = results.get(cid, "")
        flagged += 1
        record = dict(
            file=os.path.basename(f),
            ts=ts,
            call=text[:260],
            output_mentions_expC_run=mentions_run(out),
            output_has_test_token=bool(TEST_TOKEN.search(out)),
            output_len=len(out),
        )
        print(json.dumps(record))
print(json.dumps(dict(files=len(files), tool_calls_scanned=n_calls, flagged=flagged)))

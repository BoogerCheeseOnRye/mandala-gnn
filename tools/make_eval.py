"""Step 2 of the images + code plan: the evaluation sets of record.

The meter (step 1) used four ad-hoc prompts. This freezes the sets every later
step is scored on, and makes them executable:

    code   40 Python tasks, each a signature + docstring prompt, a reference
           body, and assertions that run against whatever the model writes.
           Hand-written for this file, so they cannot appear in the training
           corpus; the builder checks that too, byte for byte.

    image  12 drawings per layout (36), rendered by mandalanet.synth from
           seeds 1000-1011, hashed per image, for step 6's generation test.
           They are new drawings, not crops of the three photographs.

    .venv/bin/python tools/make_eval.py            # build + report
    .venv/bin/python tools/make_eval.py --verify   # regenerate and compare
    .venv/bin/python tools/make_eval.py --selftest # every solution passes

`runs/budget/eval.json` carries one sha256 over everything except itself; the
meter refuses to report scored numbers if it does not verify. `--verify` and
`--selftest` are also run from tests/smoke.py, so a broken task fails the
build rather than quietly lowering a later score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAYOUTS = ("hex", "rings", "chart")
SEED_START = 1000
PER_LAYOUT = 12
CORPUS = "runs/code/corpus.pt"
OUT = "runs/budget/eval.json"

# name, prompt, tests, reference body. The prompt ends where the body starts,
# so a completion is the indented suite. Tests are plain asserts on the name.
TASKS: list[tuple[str, str, list[str], str]] = [
    ("sum_list",
     'def sum_list(xs):\n    """Return the sum of the numbers in xs; 0 for an empty list."""\n',
     ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"],
     "    total = 0\n    for x in xs:\n        total += x\n    return total\n"),
    ("mean",
     'def mean(xs):\n    """Return the arithmetic mean of a non-empty list of numbers."""\n',
     ["assert abs(mean([1, 2, 3]) - 2.0) < 1e-9",
      "assert abs(mean([2, 2]) - 2.0) < 1e-9"],
     "    return sum(xs) / len(xs)\n"),
    ("median",
     'def median(xs):\n    """Return the median of a non-empty list; average the middle two if even."""\n',
     ["assert median([3, 1, 2]) == 2", "assert abs(median([1, 4, 2, 3]) - 2.5) < 1e-9"],
     "    s = sorted(xs)\n    n = len(s)\n    h = n // 2\n"
     "    if n % 2:\n        return s[h]\n    return (s[h - 1] + s[h]) / 2\n"),
    ("gcd",
     'def gcd(a, b):\n    """Return the greatest common divisor of two positive integers."""\n',
     ["assert gcd(12, 18) == 6", "assert gcd(7, 5) == 1"],
     "    while b:\n        a, b = b, a % b\n    return a\n"),
    ("is_prime",
     'def is_prime(n):\n    """Return True if n is prime, else False."""\n',
     ["assert is_prime(7)", "assert is_prime(2)", "assert not is_prime(9)",
      "assert not is_prime(1)"],
     "    if n < 2:\n        return False\n"
     "    for i in range(2, int(n ** 0.5) + 1):\n"
     "        if n % i == 0:\n            return False\n    return True\n"),
    ("factorial",
     'def factorial(n):\n    """Return n! for a non-negative integer n."""\n',
     ["assert factorial(0) == 1", "assert factorial(5) == 120"],
     "    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n"),
    ("fib",
     'def fib(n):\n    """Return the n-th Fibonacci number, with fib(0) == 0 and fib(1) == 1."""\n',
     ["assert fib(0) == 0", "assert fib(10) == 55"],
     "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    ("reverse_words",
     'def reverse_words(s):\n    """Return the words of s in reverse order, separated by one space."""\n',
     ['assert reverse_words("a b c") == "c b a"'],
     '    return " ".join(s.split()[::-1])\n'),
    ("is_palindrome",
     'def is_palindrome(s):\n    """True if s reads the same backwards, ignoring case and non-letters."""\n',
     ['assert is_palindrome("Aba")', 'assert not is_palindrome("ab")'],
     '    t = [ch.lower() for ch in s if ch.isalnum()]\n    return t == t[::-1]\n'),
    ("count_vowels",
     'def count_vowels(s):\n    """Return how many letters in s are a, e, i, o or u, either case."""\n',
     ['assert count_vowels("hello") == 2', 'assert count_vowels("XYZ") == 0'],
     '    return sum(ch in "aeiouAEIOU" for ch in s)\n'),
    ("title_words",
     'def title_words(s):\n    """Capitalise the first letter of every word in s, leaving the rest alone."""\n',
     ['assert title_words("hello world") == "Hello World"'],
     '    return " ".join(w[:1].upper() + w[1:] for w in s.split())\n'),
    ("flatten",
     'def flatten(nested):\n    """Flatten a list of lists into one list."""\n',
     ["assert flatten([[1, 2], [3]]) == [1, 2, 3]", "assert flatten([]) == []"],
     "    return [x for sub in nested for x in sub]\n"),
    ("unique",
     'def unique(xs):\n    """Return the values of xs once each, in first-seen order."""\n',
     ["assert unique([1, 2, 1, 3]) == [1, 2, 3]"],
     "    seen = set()\n    out = []\n    for x in xs:\n"
     "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n"),
    ("chunk",
     'def chunk(xs, k):\n    """Split xs into consecutive lists of length k; the last may be shorter."""\n',
     ["assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"],
     "    return [xs[i:i + k] for i in range(0, len(xs), k)]\n"),
    ("rle",
     'def rle(s):\n    """Run-length encode s as a list of (character, count) pairs."""\n',
     ['assert rle("aaabb") == [("a", 3), ("b", 2)]'],
     "    out = []\n    i = 0\n    while i < len(s):\n        j = i\n"
     "        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
     "        out.append((s[i], j - i))\n        i = j\n    return out\n"),
    ("caesar",
     'def caesar(s, k):\n    """Shift lowercase a-z in s by k places, wrapping; leave other characters."""\n',
     ['assert caesar("abc", 1) == "bcd"', 'assert caesar("xyz", 3) == "abc"'],
     "    out = []\n    for ch in s:\n"
     "        if \"a\" <= ch <= \"z\":\n"
     "            out.append(chr((ord(ch) - 97 + k) % 26 + 97))\n"
     "        else:\n            out.append(ch)\n    return \"\".join(out)\n"),
    ("word_count",
     'def word_count(s):\n    """Return a dict mapping each whitespace-separated word in s to its count."""\n',
     ['assert word_count("a b a") == {"a": 2, "b": 1}'],
     "    d = {}\n    for w in s.split():\n        d[w] = d.get(w, 0) + 1\n    return d\n"),
    ("is_anagram",
     'def is_anagram(a, b):\n    """True if a and b contain the same letters with the same counts."""\n',
     ['assert is_anagram("listen", "silent")', 'assert not is_anagram("a", "b")'],
     "    return sorted(a) == sorted(b)\n"),
    ("dot",
     'def dot(a, b):\n    """Return the dot product of two equal-length lists of numbers."""\n',
     ["assert dot([1, 2], [3, 4]) == 11"],
     "    return sum(x * y for x, y in zip(a, b))\n"),
    ("transpose",
     'def transpose(m):\n    """Return the transpose of a list of equal-length rows."""\n',
     ["assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]"],
     "    return [list(row) for row in zip(*m)]\n"),
    ("clamp",
     'def clamp(x, lo, hi):\n    """Return x limited to the range lo..hi."""\n',
     ["assert clamp(5, 0, 3) == 3", "assert clamp(-1, 0, 3) == 0",
      "assert clamp(2, 0, 3) == 2"],
     "    return max(lo, min(hi, x))\n"),
    ("sign",
     'def sign(x):\n    """Return -1, 0 or 1 according to the sign of x."""\n',
     ["assert sign(-2) == -1", "assert sign(0) == 0", "assert sign(7) == 1"],
     "    if x > 0:\n        return 1\n    if x < 0:\n        return -1\n    return 0\n"),
    ("binary_search",
     'def binary_search(xs, target):\n    """Return the index of target in sorted xs, or -1 if it is absent."""\n',
     ["assert binary_search([1, 3, 5], 3) == 1",
      "assert binary_search([1, 3, 5], 4) == -1"],
     "    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n"
     "        mid = (lo + hi) // 2\n        if xs[mid] == target:\n"
     "            return mid\n        if xs[mid] < target:\n"
     "            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n"),
    ("merge_sorted",
     'def merge_sorted(a, b):\n    """Merge two ascending lists into one ascending list."""\n',
     ["assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]"],
     "    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
     "        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n"
     "        else:\n            out.append(b[j])\n            j += 1\n"
     "    return out + a[i:] + b[j:]\n"),
    ("rotate",
     'def rotate(xs, k):\n    """Return xs rotated right by k places."""\n',
     ["assert rotate([1, 2, 3], 1) == [3, 1, 2]",
      "assert rotate([1, 2, 3], 3) == [1, 2, 3]"],
     "    if not xs:\n        return []\n    k %= len(xs)\n"
     "    return xs[-k:] + xs[:-k] if k else list(xs)\n"),
    ("cumsum",
     'def cumsum(xs):\n    """Return the running totals of xs, of the same length."""\n',
     ["assert cumsum([1, 2, 3]) == [1, 3, 6]"],
     "    out = []\n    total = 0\n    for x in xs:\n"
     "        total += x\n        out.append(total)\n    return out\n"),
    ("variance",
     'def variance(xs):\n    """Return the population variance of a non-empty list of numbers."""\n',
     ["assert abs(variance([1, 2, 3]) - 2 / 3) < 1e-9"],
     "    m = sum(xs) / len(xs)\n"
     "    return sum((x - m) ** 2 for x in xs) / len(xs)\n"),
    ("mode",
     'def mode(xs):\n    """Return the most common value of a non-empty list; ties keep the first seen."""\n',
     ["assert mode([1, 1, 2]) == 1", "assert mode([4, 4, 2, 2]) == 4"],
     "    best, best_n = xs[0], 0\n    for x in xs:\n"
     "        n = xs.count(x)\n        if n > best_n:\n"
     "            best, best_n = x, n\n    return best\n"),
    ("parse_ints",
     'def parse_ints(s):\n    """Parse whitespace-separated integers in s into a list of ints."""\n',
     ['assert parse_ints("1 2 3") == [1, 2, 3]'],
     "    return [int(x) for x in s.split()]\n"),
    ("join_ints",
     'def join_ints(xs, sep):\n    """Join a list of integers with sep, as strings."""\n',
     ['assert join_ints([1, 2], "-") == "1-2"'],
     "    return sep.join(str(x) for x in xs)\n"),
    ("fizzbuzz",
     'def fizzbuzz(n):\n    """Return a list for 1..n: Fizz for multiples of 3, Buzz of 5, FizzBuzz of both."""\n',
     ['assert fizzbuzz(5) == [1, 2, "Fizz", 4, "Buzz"]'],
     "    out = []\n    for i in range(1, n + 1):\n"
     "        if i % 15 == 0:\n            out.append(\"FizzBuzz\")\n"
     "        elif i % 3 == 0:\n            out.append(\"Fizz\")\n"
     "        elif i % 5 == 0:\n            out.append(\"Buzz\")\n"
     "        else:\n            out.append(i)\n    return out\n"),
    ("collatz_len",
     'def collatz_len(n):\n    """Return how many terms the Collatz sequence takes to reach 1, counting n and 1."""\n',
     ["assert collatz_len(6) == 9", "assert collatz_len(1) == 1"],
     "    count = 1\n    while n != 1:\n"
     "        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        count += 1\n    return count\n"),
    ("celsius_to_f",
     'def celsius_to_f(c):\n    """Convert degrees Celsius to degrees Fahrenheit."""\n',
     ["assert abs(celsius_to_f(0) - 32.0) < 1e-9",
      "assert abs(celsius_to_f(100) - 212.0) < 1e-9"],
     "    return c * 9 / 5 + 32\n"),
    ("is_sorted",
     'def is_sorted(xs):\n    """True if xs is in non-decreasing order."""\n',
     ["assert is_sorted([1, 2, 2])", "assert not is_sorted([2, 1])"],
     "    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))\n"),
    ("min_max",
     'def min_max(xs):\n    """Return (smallest, largest) of a non-empty list."""\n',
     ["assert min_max([3, 1, 2]) == (1, 3)"],
     "    return min(xs), max(xs)\n"),
    ("count_char",
     'def count_char(s, c):\n    """Return how many times the character c occurs in s."""\n',
     ['assert count_char("happy", "p") == 2'],
     "    return s.count(c)\n"),
    ("capitalize_first",
     'def capitalize_first(s):\n    """Upper-case the first character of s and leave the rest unchanged."""\n',
     ['assert capitalize_first("hello there") == "Hello there"'],
     '    return s[:1].upper() + s[1:]\n'),
    ("rolling_max",
     'def rolling_max(xs, k):\n    """Return the maximum of every window of length k in xs."""\n',
     ["assert rolling_max([1, 3, 2, 5], 2) == [3, 3, 5]"],
     "    return [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]\n"),
    ("bin_to_dec",
     'def bin_to_dec(s):\n    """Convert a string of 0s and 1s to an integer."""\n',
     ['assert bin_to_dec("101") == 5'],
     "    return int(s, 2)\n"),
    ("is_leap",
     'def is_leap(year):\n    """True for a leap year in the Gregorian calendar."""\n',
     ["assert is_leap(2024)", "assert not is_leap(1900)", "assert is_leap(2000)"],
     "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"),
]


# --------------------------------------------------------------------------- #
def canonical_sha(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def image_sha(img: torch.Tensor) -> str:
    u8 = (img.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy()
    return hashlib.sha256(u8.tobytes()).hexdigest()


def heldout_files(corpus: dict) -> list[str]:
    return sorted(f["path"] for f in corpus["files"] if f["split"] != "train")


def build(corpus: dict) -> dict:
    tasks = [{"name": n, "prompt": p, "tests": t, "solution": s}
             for n, p, t, s in TASKS]
    draw = []
    for layout in LAYOUTS:
        for seed in range(SEED_START, SEED_START + PER_LAYOUT):
            from mandalanet.synth import sample
            img = sample(layout, torch.Generator().manual_seed(seed))[2]
            draw.append({"layout": layout, "seed": seed,
                         "sha256": image_sha(img)})
    doc = {
        "version": 2,
        "kind": "images+code eval set of record",
        "corpus_sha256": corpus["sha256"],
        "code": {
            "tasks": tasks,
            "heldout_files": heldout_files(corpus),
            "sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True)
                                     .encode()).hexdigest(),
        },
        "drawings": {
            "per_layout": PER_LAYOUT,
            "seed_start": SEED_START,
            "layouts": list(LAYOUTS),
            "generator": "mandalanet.synth.sample",
            "items": draw,
            "sha256": hashlib.sha256(json.dumps(draw, sort_keys=True)
                                     .encode()).hexdigest(),
        },
    }
    doc["sha256"] = canonical_sha(doc)
    return doc


def load(path: str = OUT) -> dict:
    with open(path) as f:
        doc = json.load(f)
    if doc.get("sha256") != canonical_sha(doc):
        raise RuntimeError(f"{path}: sha256 does not verify; rebuild it with "
                           "tools/make_eval.py")
    if doc["code"]["sha256"] != hashlib.sha256(
            json.dumps(doc["code"]["tasks"], sort_keys=True).encode()).hexdigest():
        raise RuntimeError(f"{path}: task list does not match its sha256")
    return doc


def run(code: str, timeout: float = 5.0) -> bool:
    """Execute generated code (with its tests appended) in a fresh process."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code],
                               capture_output=True, timeout=timeout,
                               stdin=subprocess.DEVNULL, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False
        return p.returncode == 0


def self_check(doc: dict) -> list[str]:
    """Every reference solution must pass its own tests, in-process."""
    bad = []
    for task in doc["code"]["tasks"]:
        ns: dict[str, Any] = {}
        try:
            exec(task["prompt"] + task["solution"], ns)
            for line in task["tests"]:
                exec(line, ns)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{task['name']}: {type(exc).__name__}: {exc}")
    return bad


def leakage(doc: dict, corpus: dict) -> list[str]:
    """Prompts and held-out file paths must not be in the training bytes."""
    train = bytes(corpus["train_bytes"].to(torch.uint8).numpy())
    bad = []
    for task in doc["code"]["tasks"]:
        if task["prompt"].encode() in train:
            bad.append(f"prompt of {task['name']} appears in training bytes")
    held = set(doc["code"]["heldout_files"])
    overlap = [f["path"] for f in corpus["files"]
               if f["split"] == "train" and f["path"] in held]
    bad += [f"{p} is a held-out file but marked train" for p in overlap]
    return bad


def verify(doc: dict) -> list[str]:
    bad = []
    for item in doc["drawings"]["items"]:
        from mandalanet.synth import sample
        img = sample(item["layout"], torch.Generator().manual_seed(item["seed"]))[2]
        if image_sha(img) != item["sha256"]:
            bad.append(f"{item['layout']} seed {item['seed']}: image differs")
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    p.add_argument("--corpus", default=CORPUS)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    corpus = torch.load(a.corpus, map_location="cpu", weights_only=False)

    if a.verify or a.selftest:
        doc = load(a.out)
    else:
        doc = build(corpus)
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {a.out}")

    print(f"sha256   {doc['sha256']}")
    print(f"code     {len(doc['code']['tasks'])} tasks · "
          f"sha {doc['code']['sha256'][:12]} · "
          f"held-out files {len(doc['code']['heldout_files'])}")
    print(f"drawings {len(doc['drawings']['items'])} "
          f"({len(LAYOUTS)} layouts x {PER_LAYOUT}) · "
          f"sha {doc['drawings']['sha256'][:12]}")

    bad = leakage(doc, corpus)
    print(f"leakage  {'ok' if not bad else bad}")
    if a.selftest:
        fails = self_check(doc)
        print(f"selftest {'ok: every solution passes' if not fails else fails}")
        bad += fails
    if a.verify:
        differs = verify(doc)
        print(f"verify   {'ok: every image reproduces' if not differs else differs}")
        bad += differs
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

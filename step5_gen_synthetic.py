"""
STEP 5 (EXPERIMENT, GUIDE §4)  --  Synthetic training data from a larger model.

REVERSE / structured-first generation:
  1. SAMPLE the gold JSON fields in Python -- we control the distribution, so we
     can force coverage of every enum value, realistic null rates, and the rare
     cases (hourly pay, internships, non-USD currencies) the data-size ablation
     showed were the example-hungry part.
  2. Ask a big TEACHER model (Claude Opus 4.8) to write a natural-sounding job
     posting that realizes EXACTLY those fields.

The label is correct BY CONSTRUCTION -- we already know it; the model only writes
prose. No labeling, no verification. This is your intro to DISTILLATION: a large
teacher manufacturing training data for the small (0.5B) student.

Writes data/synthetic.jsonl in the same {"posting", "labels"} shape as
job_postings.jsonl. Then SPOT-CHECK a few by eye before training on them.

Needs API credentials: set ANTHROPIC_API_KEY, or run `ant auth login`.
    uv run python step5_gen_synthetic.py
"""

import json
import random
from pathlib import Path

from common import DATA_DIR, SEED

N_EXAMPLES = 100
MODEL = "claude-opus-4-8"
OUT_PATH = f"{DATA_DIR}/synthetic.jsonl"

# ---------------------------------------------------------------------------
# Pools the sampler draws from. Widen these -> more diversity in the data.
# ---------------------------------------------------------------------------
ROLES = [   # (base title, skill pool for that role)
    ("Backend Engineer", ["Go", "Python", "Java", "PostgreSQL", "Kafka", "Redis", "gRPC", "Docker", "Kubernetes", "AWS"]),
    ("Frontend Developer", ["React", "TypeScript", "JavaScript", "CSS", "Next.js", "Redux", "GraphQL", "Tailwind"]),
    ("Data Scientist", ["Python", "SQL", "pandas", "scikit-learn", "PyTorch", "TensorFlow", "Spark", "R"]),
    ("DevOps Engineer", ["Terraform", "Kubernetes", "AWS", "Docker", "Ansible", "Prometheus", "Linux", "CI/CD"]),
    ("Mobile Engineer", ["Swift", "Kotlin", "iOS", "Android", "React Native", "GraphQL"]),
    ("Data Engineer", ["Python", "SQL", "Spark", "Airflow", "dbt", "Snowflake", "Kafka", "AWS"]),
    ("Machine Learning Engineer", ["Python", "PyTorch", "TensorFlow", "Kubernetes", "MLflow", "AWS", "SQL"]),
    ("Product Manager", ["Roadmapping", "Analytics", "SQL", "A/B testing", "Figma", "Stakeholder management"]),
]
SENIORITIES = ["intern", "junior", "mid", "senior", "lead", "manager", "executive"]
SENIORITY_TITLE = {  # how the seniority shows up in the title text
    "intern": "Intern", "junior": "Junior", "mid": "", "senior": "Senior",
    "lead": "Lead", "manager": "Engineering Manager", "executive": "VP of Engineering",
}
WORKPLACE = ["remote", "hybrid", "onsite", None]
EMPLOYMENT = ["full_time", "part_time", "contract", "internship", "temporary"]
CITIES = ["San Francisco, CA", "Austin, TX", "New York, NY", "Seattle, WA",
          "London, UK", "Berlin, Germany", "Toronto, Canada", "Remote (US)"]
COMPANIES = ["Northwind Labs", "Brightside Studio", "Acme Cloud", "Lumen Analytics",
             "Pinecrest Systems", "Harborview Tech", "Cobalt Robotics", "Maplestone Health",
             "Quill Software", "Vantage AI"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
EDUCATION = [None, None, None, "Bachelor's degree", "Bachelor's in Computer Science",
             "Master's degree", "Master's in a quantitative field"]
YEARS_BY_SENIORITY = {  # tie experience to seniority so the data is self-consistent
    "intern": [None, 0], "junior": [1, 2], "mid": [3, 4, 5],
    "senior": [6, 7, 8], "lead": [8, 10, 12], "manager": [8, 10], "executive": [12, 15],
}


def sample_labels(rng):
    """Build one gold-JSON record by sampling each field. Weights bias toward
    realistic distributions while still hitting every enum value across ~100 rows."""
    base_title, skill_pool = rng.choice(ROLES)
    seniority = rng.choices(SENIORITIES, weights=[1, 3, 4, 4, 2, 2, 1])[0]
    if seniority in ("manager", "executive"):
        title = SENIORITY_TITLE[seniority]
    else:
        title = f"{SENIORITY_TITLE[seniority]} {base_title}".strip()

    employment = "internship" if seniority == "intern" else \
        rng.choices(EMPLOYMENT, weights=[8, 1, 2, 0, 1])[0]
    workplace = rng.choices(WORKPLACE, weights=[3, 3, 3, 1])[0]
    location = None if (workplace is None and rng.random() < 0.4) else rng.choice(CITIES)

    # Salary present ~70% of the time; sometimes only one bound is stated.
    if rng.random() < 0.7:
        currency = rng.choice(CURRENCIES)
        period = rng.choices(["year", "month", "hour"], weights=[6, 1, 2])[0]
        if period == "year":
            smin = rng.randrange(60, 180, 5) * 1000
            smax = smin + rng.randrange(15, 60, 5) * 1000
        elif period == "hour":
            smin = rng.randrange(25, 80, 5)
            smax = smin + rng.randrange(10, 40, 5)
        else:  # month
            smin = rng.randrange(4, 14) * 1000
            smax = smin + rng.randrange(1, 4) * 1000
        if rng.random() < 0.15:
            smax = None
        if rng.random() < 0.10:
            smin = None
    else:
        currency = period = smin = smax = None

    pool = skill_pool[:]
    rng.shuffle(pool)
    n_req = rng.randint(2, 5)
    required = pool[:n_req]
    preferred = pool[n_req:n_req + rng.randint(0, 3)]

    return {
        "title": title,
        "company": rng.choice(COMPANIES),
        "location": location,
        "workplace_type": workplace,
        "employment_type": employment,
        "seniority": seniority,
        "salary_min": smin,
        "salary_max": smax,
        "salary_currency": currency,
        "salary_period": period,
        "required_skills": required,
        "preferred_skills": preferred,
        "min_years_experience": rng.choice(YEARS_BY_SENIORITY[seniority]),
        "education": rng.choice(EDUCATION),
    }


GEN_SYSTEM = """You are a job-posting copywriter. You are given a set of structured FACTS about a role as JSON. Write ONE realistic, natural-sounding job posting (2-5 sentences, like a real listing).

Rules:
- Express EVERY provided (non-null) fact naturally in the prose -- don't just list them.
- Do NOT mention any field whose value is null or an empty list. Invent nothing: no salary, location, skill, years, or education that isn't in the facts.
- Weave skills in naturally ("Experience with X and Y is required; Z is a plus.").
- Vary the tone, structure, and opening line across postings.
- Output ONLY the posting text -- no preamble, no headings, no JSON."""


def main():
    rng = random.Random(SEED)
    specs = [sample_labels(rng) for _ in range(N_EXAMPLES)]

    import anthropic
    from dotenv import load_dotenv
    load_dotenv()                    # read ANTHROPIC_API_KEY from .env into the environment
    client = anthropic.Anthropic()   # resolves ANTHROPIC_API_KEY (now loaded) or an `ant` profile

    out = []
    for i, spec in enumerate(specs, 1):
        # No temperature/top_p: those are REMOVED on Opus 4.8 (passing them 400s).
        # Variety comes from the differing specs + the "vary tone" instruction.
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=GEN_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(spec, ensure_ascii=False)}],
        )
        posting = next(b.text for b in resp.content if b.type == "text").strip()
        out.append({"posting": posting, "labels": spec})
        print(f"[{i:>3}/{N_EXAMPLES}] {spec['seniority']:<9} {spec['title']}")

    Path(OUT_PATH).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out))
    print(f"\nWrote {len(out)} synthetic examples to {OUT_PATH}")
    print("SPOT-CHECK a handful by eye (does the prose match the labels?) before training on them.")


if __name__ == "__main__":
    main()

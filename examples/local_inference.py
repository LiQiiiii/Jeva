import argparse
import json

from jeva import Choice, Jeva, Noul, Score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LIQIIIII/Jeva-4B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = Jeva.from_pretrained(
        args.model, device=args.device, merge_lora=True, cache="hybrid"
    )
    result = model.system_one(
        state=(
            "The production login page returns HTTP 503. "
            "All customers are unable to sign in or complete purchases."
        ),
        questions={
            "outage": Noul(
                instructions="Does this report an ongoing service outage?"
            ),
            "team": Choice(
                instructions="Which team should handle this support ticket?",
                criteria={
                    "billing": "Payment disputes, invoices, and refunds.",
                    "engineering": "Service outages, software errors, and broken features.",
                    "general": "All other customer support requests.",
                },
                allow_null=False,
            ),
            "impact": Score(
                instructions="Rate the reported customer impact.",
                criteria=[
                    "No customers are affected.",
                    "Some customers are affected, but the service remains usable.",
                    "All customers are blocked from using a core service.",
                ],
                values=[0, 1, 2],
            ),
        },
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

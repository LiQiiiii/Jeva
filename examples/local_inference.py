import argparse
import json
from jeva import Choice, Jeva, Noul, Score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LIQIIIII/Jeva-4B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = Jeva.from_pretrained(args.model, device=args.device, merge_lora=True, cache="hybrid")
    result = model.system_one(
        "My order arrived with a cracked screen. Please send a replacement.",
        {
            "route": Choice("Choose the team best placed to help.", {
                "returns": "Damaged goods, exchanges, or refunds.",
                "billing": "Invoices and payment problems.",
            }, allow_null=True),
            "replacement": Noul("Does the customer request a replacement?"),
            "urgency": Score("Rate the urgency of this request.", [
                "Routine request without immediate impact.",
                "An important problem requiring prompt attention.",
                "An emergency requiring immediate action.",
            ]),
        },
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

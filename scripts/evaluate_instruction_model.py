import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "qwen25_evaluation.json"
SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer directly and concisely, follow "
    "all requested constraints exactly, including the number of requested items. "
    "Do not invent facts. If you do not know something or it cannot be known, say "
    "so instead of guessing."
)

EVALUATION_PROMPTS = [
    {
        "category": "text continuation",
        "prompt": "The future of AI is",
    },
    {
        "category": "factual question",
        "prompt": "What is the capital of Australia? Answer in one sentence.",
    },
    {
        "category": "arithmetic",
        "prompt": "A $12 book gets 25% off. What is the final price? Reply with only the calculation and price.",
    },
    {
        "category": "instruction following",
        "prompt": (
            "Give exactly three bullet points for houseplant care: one about watering, "
            "one about light, and one about soil. Keep each under 12 words. "
            "Do not include any other text."
        ),
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize only the stated facts in one sentence; do not add details: "
            "A town added protected bike lanes on three busy roads. Over the next year, "
            "bike commuting increased, while traffic injuries on those roads decreased. "
            "The town plans to study the results before expanding the project."
        ),
    },
    {
        "category": "uncertainty",
        "prompt": (
            "What will the exact temperature in New York City be one year from today? "
            "Answer in one sentence. Do not estimate or give seasonal averages."
        ),
    },
]


def evaluate(model_name: str, output_path: Path, max_new_tokens: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    results = []
    for item in EVALUATION_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output_ids[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        results.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "response": response,
            }
        )
        print(f"\n[{item['category']}]\n{response}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device),
                "max_new_tokens": max_new_tokens,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small repeatable evaluation of an instruction-following model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    evaluate(args.model, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()

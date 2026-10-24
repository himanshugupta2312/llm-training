import threading
import tkinter as tk

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer directly and concisely, follow "
    "all requested constraints exactly, including the number of requested items. "
    "Do not invent facts. If you do not know something or it cannot be known, say "
    "so instead of guessing."
)


class StopOnRequest(StoppingCriteria):
    def __init__(self, stop_requested):
        self.stop_requested = stop_requested

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.stop_requested(),
            dtype=torch.bool,
            device=input_ids.device,
        )


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer


class GPTInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen2.5 Instruct Chat")
        self.root.geometry("980x720")

        self.model, self.tokenizer = load_model()
        self.stop_requested = False
        self.generation_thread = None

        self.prompt_label = tk.Label(root, text="Prompt", font=("Consolas", 12, "bold"))
        self.prompt_label.pack(anchor="w", padx=12, pady=(10, 4))

        self.prompt_entry = tk.Text(root, height=6, width=120, font=("Consolas", 11))
        self.prompt_entry.insert("1.0", "The future of AI is")
        self.prompt_entry.pack(fill="x", padx=12, pady=(0, 10))

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=12, pady=4)

        self.temperature_var = tk.DoubleVar(value=0.4)
        self.top_k_var = tk.IntVar(value=40)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.max_tokens_var = tk.IntVar(value=120)

        for label, var, value in [
            ("Temperature", self.temperature_var, 0.4),
            ("Top-k", self.top_k_var, 40),
            ("Top-p", self.top_p_var, 0.9),
            ("Max tokens", self.max_tokens_var, 120),
        ]:
            frame = tk.Frame(controls)
            frame.pack(side="left", padx=6)
            tk.Label(frame, text=label, font=("Consolas", 10)).pack(anchor="w")
            entry = tk.Entry(frame, textvariable=var, width=10, font=("Consolas", 10))
            entry.pack(anchor="w")

        self.generate_button = tk.Button(root, text="Generate", command=self.start_generation, font=("Consolas", 11, "bold"))
        self.generate_button.pack(side="left", padx=12, pady=12)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_generation, font=("Consolas", 11, "bold"), state="disabled")
        self.stop_button.pack(side="left", padx=8, pady=12)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var, font=("Consolas", 10), fg="darkgreen").pack(anchor="w", padx=12)

        self.output_box = tk.Text(root, height=18, width=120, font=("Consolas", 11), wrap="word")
        self.output_box.pack(fill="both", expand=True, padx=12, pady=(8, 12))

    def stop_generation(self):
        self.stop_requested = True
        self.status_var.set("Stopping...")

    def clear_output(self):
        self.output_box.delete("1.0", tk.END)

    def start_generation(self):
        if self.generation_thread is not None and self.generation_thread.is_alive():
            return

        prompt = self.prompt_entry.get("1.0", "end-1c").strip()
        if not prompt:
            self.status_var.set("Enter a prompt first.")
            return

        try:
            settings = {
                "temperature": max(float(self.temperature_var.get()), 0.05),
                "top_k": max(int(self.top_k_var.get()), 0),
                "top_p": min(max(float(self.top_p_var.get()), 0.01), 1.0),
                "max_new_tokens": max(int(self.max_tokens_var.get()), 1),
            }
        except (tk.TclError, ValueError) as exc:
            self.status_var.set(f"Invalid generation setting: {exc}")
            return

        self.clear_output()
        self.output_box.insert(tk.END, f"You: {prompt}\n\nAssistant: ")
        self.stop_requested = False
        self.generate_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Generating...")

        self.generation_thread = threading.Thread(
            target=self.run_generation,
            args=(prompt, settings),
            daemon=True,
        )
        self.generation_thread.start()

    def run_generation(self, prompt: str, settings: dict):
        try:
            device = next(self.model.parameters()).device
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_errors = []

            def generate():
                try:
                    with torch.inference_mode():
                        self.model.generate(
                            **inputs,
                            streamer=streamer,
                            max_new_tokens=settings["max_new_tokens"],
                            do_sample=True,
                            temperature=settings["temperature"],
                            top_k=settings["top_k"],
                            top_p=settings["top_p"],
                            pad_token_id=self.tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList(
                                [StopOnRequest(lambda: self.stop_requested)]
                            ),
                        )
                except Exception as exc:
                    generation_errors.append(exc)
                    streamer.end()

            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            for chunk in streamer:
                self.root.after(0, lambda c=chunk: self.output_box.insert(tk.END, c))
            worker.join()
            if generation_errors:
                raise generation_errors[0]

            status = "Generation stopped." if self.stop_requested else "Finished."
            self.root.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, lambda: self.output_box.insert(tk.END, f"\n\nError: {error_message}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.root.after(0, self._reset_controls)

    def _reset_controls(self):
        self.generate_button.config(state="normal")
        self.stop_button.config(state="disabled")


def main():
    root = tk.Tk()
    app = GPTInferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

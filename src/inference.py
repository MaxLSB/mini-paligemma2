import torch
from PIL import Image
import time

from utils.preprocessing import PaliGemmaProcessor
from paligemma.gemma import KVCache
from utils.load_weights import load_weights
from utils.config import get_args
from utils.detection import display_detection
from utils.sampling import sample_top_p

################################### Inference ###################################


class InferenceEngine:

    def __init__(self, model_type: str, model_path: str, device: str):
        self.device = device
        self.model_type = model_type

        # Load model and tokenizer
        self.model, self.tokenizer = load_weights(model_path, model_type, device)
        self.model = self.model.to(device).eval()

        # Initialize processor
        vision_config = self.model.config.vision_config
        self.processor = PaliGemmaProcessor(
            self.tokenizer,
            num_image_tokens=vision_config.num_image_tokens,
            image_size=vision_config.image_size,
        )

    def move_inputs_to_device(self, model_inputs: dict, device: str):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        return model_inputs

    def get_model_inputs(self, prompt: str, image_file_path: str, device: str):
        image = Image.open(image_file_path)
        images = [image]
        prompts = [prompt]
        model_inputs = self.processor(text=prompts, images=images)
        model_inputs = self.move_inputs_to_device(model_inputs, device)
        return model_inputs

    def generate(
        self,
        prompt: str,
        image_file_path: str,
        max_tokens_to_generate: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        detection: bool = False,
    ):

        # Generate response with a single image / text pair
        with torch.no_grad():
            # Get the model inputs
            model_inputs = self.get_model_inputs(prompt, image_file_path, self.device)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            pixel_values = model_inputs["pixel_values"]

            # We reinitialize the KV cache for each new prompt
            kv_cache = KVCache()

            stop_token = self.processor.tokenizer.eos_token_id
            generated_tokens = []

            print("> Output: ", end="", flush=True)
            # Start time for tokens per second calculation
            start_time = time.time()
            tps = 0.0

            # Start generating tokens
            for _ in range(max_tokens_to_generate):
                # Get the model outputs
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]

                # Sample the next token if do_sample=True, else greedy
                if do_sample:
                    # Apply temperature
                    next_token_logits = torch.softmax(
                        next_token_logits / temperature, dim=-1
                    )
                    next_token = sample_top_p(next_token_logits, top_p)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Remove batch dimension and append token to the generated tokens
                next_token = next_token.squeeze(0)
                generated_tokens.append(next_token)

                # Decode and print the generated token
                decoded_token = self.processor.tokenizer.decode(
                    [next_token.item()], skip_special_tokens=True
                )
                print(decoded_token, end="", flush=True)

                # Stop if the stop token has been generated
                if next_token.item() == stop_token:
                    break

                # Since we use KV cache, only the last generated token is passed as input
                input_ids = next_token.unsqueeze(-1)  # (batch_size, 1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=input_ids.device)],
                    dim=-1,
                )

                # Tokens per second calculation
                elapsed_time = time.time() - start_time
                tokens_generated = len(generated_tokens)
                tps = tokens_generated / elapsed_time

        print(f"\n> Speed: {tps:.2f} tokens/sec")

        # Display the detection if detection=True (the prompt has to be "detect <object>").
        if detection and "detect" in prompt.lower():
            generated_tokens = torch.cat(generated_tokens, dim=-1)
            decoded = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            print(f"> Detection: Image displayed; close the image window to continue.")
            display_detection(decoded, image_file_path)

        print("-" * 55)


################################### Main ###################################


def main(
    model_type,
    model_path,
    detection,
    image_file_path,
    max_tokens_to_generate,
    temperature,
    top_p,
    do_sample,
    only_cpu,
):
    device = "cuda" if torch.cuda.is_available() and not only_cpu else "cpu"

    # Information display
    print(f"> Device: {device}")
    if model_type == "paligemma2":
        print(f"> Model: PaliGemma 2")
    elif model_type == "paligemma":
        print(f"> Model: PaliGemma")
    else:
        raise ValueError(
            "Invalid model type. Please choose between 'paligemma' and 'paligemma2'."
        )
    print(f'> Loading the weights from: "{model_path}"')
    print(f'> Image file path: "{image_file_path}"\n')

    # Initialize the inference engine
    engine = InferenceEngine(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
    )

    # Information display
    print("=" * 55)
    print(
        "- Enter a prompt to generate a response.\n- Type 'exit' to quit the program.\n- Supported commands: 'detect <object>', 'describe', etc..."
    )
    print("=" * 55 + "\n")

    # Interactive prompt loop
    while True:
        try:
            prompt = input("> Input: ").strip()
            if prompt.lower() == "exit":
                break

            # generate the answer
            engine.generate(
                prompt=prompt,
                image_file_path=image_file_path,
                max_tokens_to_generate=max_tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                detection=detection,
            )

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    args = get_args()
    main(
        model_type=args.model_type,
        model_path=args.model_path,
        detection=args.detection,
        image_file_path=args.image_file_path,
        max_tokens_to_generate=args.max_tokens_to_generate,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        only_cpu=args.only_cpu,
    )

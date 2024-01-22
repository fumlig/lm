#!/usr/bin/env python3


import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="language modeling for the shell")

    parser.add_argument("-m", "--model", metavar="MODEL", default=os.getenv("LM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"), help="model name or path (default: %(default)s, env: $LM_MODEL)")
    parser.add_argument("-b", "--backend", metavar="BACKEND", choices=["transformers", "llamacpp", "openai"], default=os.getenv("LM_BACKEND", "transformers"), help="backend to use (default: %(default)s, choices: %(choices)s, env: $LM_BACKEND)")
    parser.add_argument("-d", "--device", metavar="DEVICE", default=os.getenv("LM_DEVICE", "cuda"), help="device to use (default: %(default)s, env: $LM_DEVICE)")
    parser.add_argument("-t", "--template", metavar="TEMPLATE", default=os.getenv("LM_TEMPLATE", "[INST] {} [/INST]"), help="prompt template (default: '%(default)s', env: $LM_TEMPLATE)")

    parser.add_argument("--max-tokens", metavar="MAX_TOKENS", type=int, help="maximum number of tokens to generate")
    parser.add_argument("--stop-at", metavar="STOP_AT", nargs="*", help="stop generation at this token")
    parser.add_argument("--sampler", metavar="SAMPLER", choices=["greedy", "multinomial"], default="multinomial", help="sampler to use (default: %(default)s, choices: %(choices)s)")

    constraint = parser.add_mutually_exclusive_group()

    # todo: simplify with same target variable and factory function update
    constraint.add_argument("-c", "--choices", metavar="CHOICE", nargs="*", help="contrain output to choices")
    constraint.add_argument("-r", "--regex", metavar="REGEX", help="contrain output to regex pattern")
    constraint.add_argument("-s", "--schema", metavar="SCHEMA", help="constrain output to json schema")
    constraint.add_argument("-g", "--grammar", metavar="GRAMMAR", help="constrain output to ebnf grammar")

    mode = parser.add_mutually_exclusive_group()
    
    mode.add_argument("-C", "--chat", help="chat mode", action="store_true")
    mode.add_argument("-l", "--lines", help="generate output per line of input", action="store_true")
    # todo: custom separator, limit to readlines
    
    parser.add_argument("prompt", metavar="PROMPT", type=str, nargs="?", help="prompt to generate from (reads from stdin if empty)")

    return parser.parse_args()


def make_model(backend, model_name_or_path, device):
    from outlines import models
    import torch

    match backend:
        case "transformers":
            return models.transformers(model_name_or_path, device=device, model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "flash_attention_2"})
        case "llamacpp":
            return models.llamacpp(model_name_or_path, device=device)
        case "openai":
            return models.openai(model_name_or_path)
        case _:
            raise ValueError(f"unknown backend: {backend}")


def make_generator(model, choices=None, regex=None, schema=None, grammar=None, sampler=None):
    from outlines import generate

    match sampler:
        case "multinomial":
            sampler = generate.samplers.multinomial
        case "greedy":
            sampler = generate.samplers.greedy
        case _:
            raise ValueError(f"unknown sampler: {sampler}")

    if choices:
        return generate.choice(model, choices, sampler=sampler)

    if regex:
        return generate.regex(model, regex, sampler=sampler)

    if schema:
        return generate.json(model, schema, sampler=sampler)
    
    if grammar:
        return generate.cfg(model, grammar, sampler=sampler)

    return generate.text(model, sampler=sampler)


def chat(generator):
    import readline # noqa

    print("exit with ctrl+d")

    tokenizer = generator.tokenizer.tokenizer
    history = []

    while True:
        try:
            prompt = input("> ")
            history.append({"role": "user", "content": prompt})

            prompt = tokenizer.apply_chat_template(history, tokenize=False)
            response = ""

            for tokens in generator.stream(prompt):
                text = "".join(tokens)
                response += text

                print(text, end="", flush=True)
    
            print("\n")

            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n")
            continue
        except EOFError:
            break


def main():
    args = parse_args()

    model = make_model(args.backend, args.model, args.device)
    generator = make_generator(model, choices=args.choices, regex=args.regex, schema=args.schema, grammar=args.grammar, sampler=args.sampler)

    if args.chat:
        chat(generator)
        return
    
    if args.lines:
        prompts = [args.template.format(line) for line in sys.stdin.readlines()]
        results = generator(prompts, max_tokens=args.max_tokens, stop_at=args.stop_at)
        
        for result in results:
            print(result)
        
        return

    if args.prompt is None:
        prompt = sys.stdin.read()
    else:
        prompt = args.prompt

    prompt = args.template.format(prompt)

    for result in generator.stream(prompt, max_tokens=args.max_tokens, stop_at=args.stop_at):
        print("".join(result), end="", flush=True)


if __name__ == "__main__":
    exit(main())

import argparse
from itertools import count
import re
import os
import sys
import contextlib

from nemo_rl.algorithms.utils import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", type=str, nargs="+")
    parser.add_argument("--tokenizer", type=str, default=None)
    return parser.parse_args()


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main():
    args = parse_args()
    
    if len(args.log_files) > 1:
        if "step" in args.log_files[0]:
            steps = [int(re.search(r"step_(\d+)", file).group(1)) for file in args.log_files]
            args.log_files = [file for _, file in sorted(zip(steps, args.log_files))]
        else:
            sizes = [int(re.search(r"(\d+)B", file).group(1)) for file in args.log_files]
            args.log_files = [file for _, file in sorted(zip(sizes, args.log_files))]
    
    avg_input_lengths = []
    avg_output_lengths = []
    full_tag_ratios = []
    partial_tag_ratios = []
    no_tag_ratios = []
    for log_file in args.log_files:
        with open(log_file, "r") as fin:
            section = None
            config_lines = []
            input_lines = []
            output_lines = []
            input_lengths = []
            output_lengths = []
            num_tags = []
            tokenizer = None
            
            for line in fin:
                if "Final config" in line:
                    section = "config"
                    continue
                elif "chat template" in line:
                    section = None
                    config = eval("".join(config_lines))
                    if args.tokenizer is not None:
                            config["tokenizer"]["name"] = args.tokenizer
                    with suppress_stdout_stderr():
                        tokenizer = get_tokenizer(config["tokenizer"])
                elif "<<<<<<<<<<<<<<< inputs <<<<<<<<<<<<<<<" in line:
                    section = "input"
                    continue
                elif section == "input" and "======================================" in line:
                    section = "output"
                    continue
                elif ">>>>>>>>>>>>>>> outputs >>>>>>>>>>>>>>>" in line:
                    section = None
                    # remove the last \n
                    input = "".join(input_lines)[:-1]
                    output = "".join(output_lines)[:-1]
                    input_length = len(tokenizer.encode(input))
                    output_length = len(tokenizer.encode(output))
                    num_tag = ("<think>" in output) + ("</think>" in output)
                    input_lengths.append(input_length)
                    output_lengths.append(output_length)
                    num_tags.append(num_tag)
                    input_lines = []
                    output_lines = []
                if section == "config":
                    config_lines.append(line)
                if section == "input":
                    input_lines.append(line)
                elif section == "output":
                    output_lines.append(line)

        print(log_file)
        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_output_length = sum(output_lengths) / len(output_lengths)
        full_tag_ratio = sum(x == 2 for x in num_tags) / len(num_tags)
        partial_tag_ratio = sum(x == 1 for x in num_tags) / len(num_tags)
        no_tag_ratio = sum(x == 0 for x in num_tags) / len(num_tags)
        avg_input_lengths.append(avg_input_length)
        avg_output_lengths.append(avg_output_length)
        full_tag_ratios.append(full_tag_ratio)
        partial_tag_ratios.append(partial_tag_ratio)
        no_tag_ratios.append(no_tag_ratio)
        print(f"average input length={avg_input_length:.1f}")
        print(f"average output length={avg_output_length:.1f}")
        print(f"think tags: full={full_tag_ratio:.2%}, partial={partial_tag_ratio:.2%}, none={no_tag_ratio:.2%}")
        print()

    avg_input_lengths = ", ".join([f"{length:.1f}" for length in avg_input_lengths])
    avg_output_lengths = ", ".join([f"{length:.1f}" for length in avg_output_lengths])
    full_tag_ratios = ", ".join([f"{ratio*100:.2f}" for ratio in full_tag_ratios])
    partial_tag_ratios = ", ".join([f"{ratio*100:.2f}" for ratio in partial_tag_ratios])
    no_tag_ratios = ", ".join([f"{ratio*100:.2f}" for ratio in no_tag_ratios])
    print(f"average input length array: [{avg_input_lengths}]")
    print(f"average output length array: [{avg_output_lengths}]")
    print(f"full think tag ratio array: [{full_tag_ratios}]")
    print(f"partial think tag ratio array: [{partial_tag_ratios}]")
    print(f"no think tag ratio array: [{no_tag_ratios}]")


if __name__ == "__main__":
    main()
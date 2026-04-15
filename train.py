import argparse
from srcs import (load_config, run_train)
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder")
    parser.add_argument("--task", "-t", type=int, default=0)
    parser.add_argument("--subject", "-s", type=int, default=0)
    parser.add_argument("--config", "-c", type=str, default="config.json")
    args = parser.parse_args()
    return args


def main():
    try:
        args = parse_args()
        config = load_config(args.config)
        run_train(args.datafolder, config,
              args.task, args.subject)
    except Exception as e:
        print("Error:", e)
    

if __name__ == "__main__":
    main()

import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, ravdess, eb_small, anime


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "RAVDESS" in config["dataset"]:
        ravdess.prepare_align(config)
    if "EB_Small" in config["dataset"]:
        eb_small.prepare_align(config)
    if "anime" in config["dataset"]:
        anime.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)

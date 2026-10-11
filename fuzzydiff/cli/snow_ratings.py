"""Summarize saved independent snow ratings."""
import argparse
from pathlib import Path
from fuzzydiff.experiments.snow.ratings import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('ratings', type=Path, nargs='+')
    args = parser.parse_args(argv)
    result = analyze(args.output, args.ratings)
    print(f"Ratings summarized for {result['available_images']}/{result['expected_images']} images.")
    print('At least two complete reviewer IDs per image:', result['all_images_have_two_complete_reviews'])
    print('Reviewer independence must be confirmed by the study owner. Results are descriptive.')


if __name__ == '__main__':
    main()

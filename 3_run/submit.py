import os
import sys
import logging
import argparse
from pathlib import Path
from zipfile import ZipFile

logging.basicConfig()
logger = logging.getLogger("submit.py")

NUM_WORDS = 8
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off"]
MODEL_NAME = "micro_kws_student"
QUANTIZED = True


def get_lines(file):
    with open(file, "r") as handle:
        lines = handle.readlines()
    stripped = [line.strip() for line in lines if len(line) > 0]
    return stripped


def count_lines(file):
    return len(get_lines(file))


def count_names(file):
    names = []
    for line in get_lines(file):
        assert "#" not in line, "# is not allowed in names.txt"
        names.append(line)
    return len(names)


def get_words(file):
    lines = get_lines(file)
    assert len(lines) == 1, "words.txt should only have a single line with comma separated values"
    line = lines[0]
    words = line.split(",")
    return words


def check(path, directory=True, allow_missing=False):
    if (directory and path.is_dir()) or (not directory and path.is_file()):
        return True
    if allow_missing:
        logger.warning("%s does not exist. Submission will be inclomplete!", path)
    else:
        logger.error(
            "%s does not exist. Use --ignore-missing to create an incomplete submission.", path
        )
        sys.exit(1)


def check_file(path, allow_missing=False):
    return check(path, directory=False, allow_missing=allow_missing)


def check_directory(path, allow_missing=False):
    return check(path, directory=True, allow_missing=allow_missing)


def gather_files(directory, allow_missing=False):
    files = []
    directories = []

    student = directory / "student"
    assert student.is_dir(), f"{student} directory does not exist"
    gen = directory / ".." / "2_deploy" / "gen"
    mlf = gen / "mlf.tar"
    if check_file(mlf, allow_missing=allow_missing):
        files.append(mlf)
    mlf_tuned = gen / "mlf_tuned.tar"
    if check_file(mlf_tuned, allow_missing=allow_missing):
        files.append(mlf_tuned)
    prj = directory / "prj"
    if check_directory(prj, allow_missing=allow_missing):
        directories.append(prj)
    prj_tuned = directory / "prj_tuned"
    if check_directory(prj_tuned, allow_missing=allow_missing):
        directories.append(prj_tuned)
    posterior = student / "posterior"
    if check_directory(posterior, allow_missing=allow_missing):
        directories.append(posterior)

    # Metadata
    names_path = student / "names.txt"
    assert names_path.is_file(), f"{names_path} file does not exist"
    assert count_names(names_path) > 0
    files.append(names_path)
    words_path = student / "words.txt"
    assert words_path.is_file(), f"{words_path} file does not exist"
    words = get_words(words_path)
    assert len(words) == NUM_WORDS, f"Expected {NUM_WORDS} in student/words.txt"
    # assert WORDS == words, "Please use words matching the provided model"
    files.append(words_path)

    return files, directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_2.zip")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Will turn errors into warnings, allowing to create an incomplete submission.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print which files and directories are added to the submission.",
    )
    args = parser.parse_args()

    directory = Path(".")
    files, directories = gather_files(directory, allow_missing=args.ignore_missing)

    print(f"Creating archive: {args.out}\n")

    with ZipFile(args.out, "w") as handle:
        for file in files:
            if args.verbose:
                print(f"Adding file: {file}")
            handle.write(file)
        for directory in directories:
            for dirname, subdirs, files in os.walk(directory):
                if (
                    "/build" in dirname
                    or ".ipynb_checkpoints" in dirname
                    or ".pkl_memoize_py3" in dirname
                ):
                    continue
                count = len(files)
                if count > 0:
                    if args.verbose:
                        print(f"Adding directory: {dirname} ({count} files)")
                    for file in files:
                        filename = os.path.join(dirname, file)
                        # print(f"Adding file: {filename}")
                        handle.write(filename)
        print(f"\nDone. Please upload the {args.out} file to Moodle!")


if __name__ == "__main__":
    main()

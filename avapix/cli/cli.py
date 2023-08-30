import inquirer
from platform import platform
from os import system, path
from rich import print
from shutil import get_terminal_size

from avapix.common.configs import version_list, last_version


# Source: https://github.com/TorhamDev/Better-Movie-finder
def print_banner():
    columns = get_terminal_size().columns
    banner = """
    [green]
    ░█████╗░██╗░░░██╗░█████╗░██████╗░██╗██╗░░██╗
    ██╔══██╗██║░░░██║██╔══██╗██╔══██╗██║╚██╗██╔╝
    ███████║╚██╗░██╔╝███████║██████╔╝██║░╚███╔╝░
    ██╔══██║░╚████╔╝░██╔══██║██╔═══╝░██║░██╔██╗░
    ██║░░██║░░╚██╔╝░░██║░░██║██║░░░░░██║██╔╝╚██╗
    ╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝
    [/green]                                                
    """
    banner = banner.split("\n")
    banner = "\n".join(line.center(columns) for line in banner)
    print(banner)


# ------------------------ Embedding ------------------------


def clear_terminal() -> None:
    running = platform()

    if "Windows" in running:
        system("cls")

    elif "Linux" in running:
        system("clear")

    elif "macOS" in running:
        system("clear")


def ask_with_options() -> str:
    questions = [
        inquirer.List(
            "choice",
            message="What do you want to do",
            choices=["Generate", "Extract"],
        ),
    ]
    answers = inquirer.prompt(questions)

    return answers["choice"]

def ask_for_embedding_text() -> str:
    questions = [
        inquirer.Text(
            "embedding_text",
            message="Enter your embedding text",
            validate=lambda _, x: len(x) > 0,
        )
    ]
    answers = inquirer.prompt(questions)

    return answers["embedding_text"]


def ask_for_random_seed() -> int:
    def validator(_, x):
        if x.isdigit():
            return int(x) >= 0 and int(x) <= 255
        else:
            return x == ""
        
    questions = [
        inquirer.Text(
            "random_seed",
            message="Enter random seed [Enter]",
            validate=validator,
        ),
    ]
    answers = inquirer.prompt(questions)

    return answers["random_seed"]


def ask_for_version() -> str:
    questions = [
        inquirer.List(
            "version",
            message="Choose version [Space/Enter]",
            choices=version_list,
            default=last_version,
        ),
    ]
    answers = inquirer.prompt(questions)

    return answers["version"]


def ask_for_export_sizes() -> list:
    questions = [
        inquirer.Checkbox(
            "export_sizes",
            message="Choose export sizes [Space/Enter]",
            choices=[("240", 240), ("320", 320), ("400", 400), ("480", 480)],
            default=[320],
        ),
    ]
    answers = inquirer.prompt(questions)

    return answers["export_sizes"]


def print_file_names(file_names: list[str]) -> None:
    print("[bold]Exported files:[/bold]")
    for file_name in file_names:
        print(f"  [green]{file_name}[/green]")


# ------------------------ Extracting ------------------------


def ask_for_image_path() -> str:
    questions = [
        inquirer.Text(
            "extracting_image_path",
            message="Enter image path",
            validate=lambda _, x: path.isfile(x),
        )
    ]
    answers = inquirer.prompt(questions)

    return answers["extracting_image_path"]


def print_extracted_text(text: str) -> None:
    print(f"[bold]Extracted text:[/bold] [green]{text}[/green]")

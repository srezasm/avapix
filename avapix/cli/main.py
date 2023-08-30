from avapix.cli.cli import *
import avapix.cli.cli_helper as helper


def main():
    clear_terminal()

    print_banner()

    choice = ask_with_options()
    if choice == "Generate":
        embed_text = ask_for_embedding_text()
        random_seed = ask_for_random_seed()
        version = ask_for_version()
        export_sizes = ask_for_export_sizes()

        file_names = helper.embed(embed_text, export_sizes, random_seed, version)

        print_file_names(file_names)

    elif choice == "Extract":
        image_path = ask_for_image_path()

        extracted_text = helper.extract(image_path)

        print_extracted_text(extracted_text)


if __name__ == "__main__":
    main()

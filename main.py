from data_repositories import (
    DataGovDataRepository,
    ESSDiveDataRepository,
    HuggingFaceDataRepository,
    IEEEDataPortDataRepository,
)


def main():
    repositories = [
        DataGovDataRepository(),
        ESSDiveDataRepository(),
        HuggingFaceDataRepository(),
        IEEEDataPortDataRepository(),
    ]
    for repository in repositories:
        repository.plot_format_counts()
        repository.plot_file_sizes()


if __name__ == "__main__":
    main()

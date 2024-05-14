def get_workdir():
    try:
        from google.colab import drive
        return "/gdrive/My Drive/projects/llms-conceptual-metaphor-interpretation"
    except BaseException:
        return "./"

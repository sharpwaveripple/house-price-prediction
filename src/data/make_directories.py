# -*- coding: utf-8 -*-
import os


def main():
    project_dir = "../../"
    for i in ["raw", "processed"]:
        data_dir = os.path.join(project_dir, "data", i)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)


if __name__ == "__main__":
    main()

import tagging
import interesting


def main():
    tag = tagging.Tagging("data/All_Images/")
    results = tag.tag_images(conf=0.1)

    i = interesting.Interestingness()
    i.process_images(results)


if __name__ == "__main__":
    main()

import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str)
    args, _ = parser.parse_known_args()

    img = cv2.imread(args.p)
    img[:] = 0
    cv2.imwrite(args.p, img)


if __name__ == "__main__":
    main()

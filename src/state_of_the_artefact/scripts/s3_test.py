import pandas as pd

from state_of_the_artefact.helpers.upload import upload_obj


def main(args=None):
    d = [{"i": i,
          "j": 1000000 - i,
          "k": 2000000 - i,
          "l": 3000000 - i,
          "m": 4000000 - i,
          "n": 5000000 - i}
         for i in range(0, 10000)]

    df = pd.DataFrame(d)
    print("Uploading...", end=" ")
    upload_obj(df, "state-of-the-artefact", "another_test.gz")
    print("Done.")


if __name__ == "__main__":
    main()

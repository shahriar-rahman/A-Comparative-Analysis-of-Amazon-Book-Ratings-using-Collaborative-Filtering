import pandas as pd


class GenericUtils:
    def __init__(self):
        print("> Connected to utils.")
        pass

    @staticmethod
    def view_dataframe(df, content):
        print('-' * 150)
        try:
            if df is not None:
                df_table = df.head(content)
                print(df_table.to_string())

        except Exception as exc:
            print("! Invalid Dataframe. !", exc)
            raise FileNotFoundError("! Dataframe not found. !")

        else:
            print("> Dataframe loaded successfully.")

        finally:
            print('-' * 150)

    @staticmethod
    def df_length(df, feature=False):
        try:
            if df is not None:
                if feature:
                    return len(df[feature])


                else:
                    return len(df)

        except Exception as exc:
            print("! Invalid Dataframe. !", exc)
            raise FileNotFoundError("! Dataframe not found. !")

        else:
            print("> Dataframe validated.")


if __name__ == "__main__":
    main = GenericUtils()

symbols = [
    "BTC",
    "ETH",
    "ADA",
    "SOL",
    "LTC",
    "DOGE",
    "XRP",
    "DOT",
]

available_searches = {}


def testable_search(cls):
    available_searches[cls.__name__] = cls
    return cls


@testable_search
def regularization_search():

    search = {
        "reg": [0, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    }

    return search

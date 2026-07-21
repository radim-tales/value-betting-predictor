from __future__ import annotations
from vbp.devig import devig
from .adapter import classify_book

def find_value(books: dict, min_edge: float = 0.03, odds_min: float = 1.6,
               odds_max: float = 8.0, truth: str = "pinnacle") -> list[dict]:
    if truth not in books:
        return []
    t = books[truth]
    fair = dict(zip("HDA", devig([t["H"], t["D"], t["A"]], "shin")))  # TRUTH
    out = []
    for o in "HDA":
        price, book = max((books[b][o], b) for b in books)   # best price across books
        edge = fair[o] * price - 1
        if edge >= min_edge and odds_min <= price <= odds_max:
            out.append({"outcome": o, "price": price, "book": book,
                        "book_type": classify_book(book), "edge": edge, "pin_fair": fair[o]})
    return out

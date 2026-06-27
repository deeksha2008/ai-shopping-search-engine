from symspellpy import SymSpell, Verbosity


class TypoCorrector:
    def __init__(self, vocabulary: list[str], max_edit_distance: int = 2):
        self.sym_spell = SymSpell(max_edit_distance, prefix_length=7)
        for word in vocabulary:
            self.sym_spell.create_dictionary_entry(word.lower(), 1)

    def correct_word(self, word: str) -> str:
        suggestions = self.sym_spell.lookup(
            word.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=2,
        )
        if suggestions:
            return suggestions[0].term
        return word

    def correct_query(self, query: str) -> str:
        tokens = query.split()
        corrected = [self.correct_word(token) for token in tokens]
        return " ".join(corrected)

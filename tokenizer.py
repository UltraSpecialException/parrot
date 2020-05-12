from typing import Dict, List, Any, Optional
import warnings
import json


class LockableDictionary(dict):
    """
    A dictionary that prevents users to change the values within the dictionary
    once it is locked. After the dictionary is locked, it can no longer be
    unlocked.
    """
    def __init__(self, *args) -> None:
        """
        Initialize a new LockableDictionary instance.
        """
        self._locked = False
        super(LockableDictionary, self).__init__(*args)

    def __str__(self) -> str:
        """
        Return the string representation of the dictionary.
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Return the visual representation of the dictionary.
        """
        lock_status = "Locked" if self.is_locked else "Unlocked"
        return f"LockableDictionary\n" \
            f"[Content]: {super(LockableDictionary, self).__repr__()}\n" \
            f"[Lock Status]: {lock_status}"

    @property
    def is_locked(self) -> bool:
        """
        Return True if the dictionary is already locked and False otherwise.
        """
        return self._locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        """
        If the dictionary is already locked and <value> is False, an error will
        be raised. Otherwise, set the value of is_locked as specified.
        """
        if self._locked and not value:
            raise ValueError("Cannot unlock dictionary after it is locked.")

        self._locked = value

    def lock(self) -> None:
        """
        Lock the dictionary. Note that after this method is called. The
        dictionary can no longer be unlocked and its elements can no longer be
        changed.
        """
        self._locked = True

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        If permissible, set the key <key> to have value <value> in this
        dictionary.
        """
        if self.is_locked:
            raise AttributeError("This dictionary is locked and hence its "
                                 "content can no longer be set.")

        super(LockableDictionary, self).__setitem__(key, value)


class Tokenizer:
    """
    An interface for tokenizing words into numeric indices for seq-to-seq
    modelling.
    """
    __required__ = {
        "num_words",
        "word_to_idx",
        "idx_to_word",
        "all_words",
        "word_counts",
        "is_locked"
    }

    __type_check__ = {
        "num_words": int,
        "word_to_idx": dict,
        "idx_to_word": dict,
        "all_words": list,
        "word_counts": dict,
        "is_locked": bool
    }

    def __init__(self,
                 num_words: Optional[int]=None,
                 word_to_idx: Optional[Dict[str, int]]=None,
                 idx_to_word: Optional[Dict[int, str]]=None,
                 all_words: Optional[List[str]]=None,
                 word_counts: Optional[Dict[str, int]]=None,
                 is_locked: Optional[bool]=None) -> None:
        """
        Initializes a tokenizer instance. If any of the parameter is None, an
        empty Tokenizer is created.
        """
        all_params = [
            num_words, word_to_idx, idx_to_word, all_words, word_counts
        ]
        if any(param is None for param in all_params):
            if any(param is not None for param in all_params):
                warnings.warn("Detected a None-valued argument but also a non "
                              "None-valued argument. Ignoring the non "
                              "None-valued argument and creating an empty "
                              "Tokenizer.")

            self._num_words = 4
            self._word_to_idx = LockableDictionary({
                "[PAD]": 0,
                "[START]": 1,
                "[END]": 2,
                "[UNK]": 3
            })
            self._idx_to_word = LockableDictionary({
                0: "[PAD]",
                1: "[START]",
                2: "[END]",
                3: "[UNK]"
            })
            self._all_words = {"[PAD]", "[START]", "[END]", "[UNK]"}
            self._word_counts = LockableDictionary()
            self._is_locked = False
        else:
            assert num_words == len(all_words), f"Paramter num_words and the " \
                f"length of all_words do not match: {num_words}, " \
                f"{len(all_words)}"
            assert len(word_to_idx) == len(idx_to_word) == len(all_words) \
                   == len(word_counts), f"Paramters word_to_idx, idx_to_word, " \
                f"all_words and word_counts do not all have the same lengths: " \
                f"{len(word_to_idx)}, {len(idx_to_word)}, {len(all_words)}, " \
                f"{len(word_counts)}"

            self._num_words = num_words
            self._word_to_idx = LockableDictionary(word_to_idx)
            self._idx_to_word = LockableDictionary(idx_to_word)
            self._all_words = set(all_words)
            self._word_counts = LockableDictionary(word_counts)
            self._is_locked = is_locked

            if is_locked:
                self.seal_tokenizer()

        self.pad_token = self.word_to_index_mapping["[PAD]"]
        self.start_token = self.word_to_index_mapping["[START]"]
        self.end_token = self.word_to_index_mapping["[END]"]

    def add_word(self, word: str) -> None:
        """
        Add the word <word> into the tokenizer's vocabulary. If the word already
        exists within the vocabulary, only the word's count needs to be updated.
        Otherwise, the word is added into each mapping and the word's count is
        set to 1.
        """
        if word in self._all_words:
            self.word_counts[word] += 1
        else:
            self._word_to_idx[word] = self._num_words
            self._idx_to_word[self._num_words] = word
            self._all_words.add(word)
            self._word_counts[word] = 1
            self._num_words += 1

    def process_sentence(self, sentence: str, split_token: str=" ") -> None:
        """
        Break the sentence <sentence> into individual tokens and add each word
        into the vocabulary of the tokenizer.

        <sentence> should be pre-processed such that by splitting down
        <split_token>, each resulting element of the splitted sentence is its
        own token.
        """
        for word in sentence.split(split_token):
            self.add_word(word)

    def seal_tokenizer(self) -> None:
        """
        Lock all the lockable dictionaries storing the vocabulary information
        for this tokenizer. Once this is done, it cannot be undone and these
        dictionaries can no longer be modified.

        This method should be called after all training data have been processed
        and tokenized.
        """
        self._is_locked = True
        self._word_to_idx.lock()
        self._idx_to_word.lock()
        self._word_counts.lock()

    @classmethod
    def from_dict(cls, saved_tokenizer: dict) -> "Tokenizer":
        """
        Load a tokenizer from a dictionary (<saved_tokenizer>).

        <saved_tokenizer> should be loaded from a JSON file or any file that
        can be interpreted as a dictionary in Python.
        """
        missing_keys = cls.__required__ - set(saved_tokenizer.keys())
        if missing_keys:
            keys_to_report = " ".join(missing_keys)
            raise ValueError(f"The dictionary given is missing the following "
                             f"keys: {keys_to_report}.")

        copy = saved_tokenizer.copy()

        for key in saved_tokenizer:
            if key in cls.__required__:
                given_type = type(saved_tokenizer[key])
                expected_type = cls.__type_check__[key]

                if given_type is not expected_type:
                    raise TypeError(f"The type expected for key {key} is "
                                    f"{expected_type} but was given "
                                    f"{given_type}")

                if isinstance(saved_tokenizer[key], dict):
                    copy[key] = LockableDictionary(copy[key])

            else:
                warnings.warn(f"Superfluous key {key} was given, removing it "
                              f"from the dictionary.")
                del copy[key]


        return cls(**copy)

    def to_dict(self, path: str) -> None:
        """
        Save a JSON file at the given path storing this tokenizer's vocabulary
        information. The dictionary's lock status will also be saved.
        """
        info_to_save = {
            "num_words": self.num_words,
            "word_to_idx": self.word_to_index_mapping,
            "idx_to_word": self.index_to_word_mapping,
            "all_words": self.vocabulary,
            "word_counts": self.word_counts,
            "is_locked": self.is_locked
        }

        with open(path, "w+") as output_file:
            json.dump(info_to_save, output_file)

    def warn_use_unlocked(self) -> None:
        """
        This method is used in the encoding and decoding methods to issue a
        warning if these methods are called when the tokenizer is unsealed.
        """
        if not self.is_locked:
            warnings.warn("Tokenizing with an unlocked tokenizer, meaning that "
                          "its content can be changed. If the setting up of "
                          "the tokenizer is finished, consider locking it for "
                          "safety (call .seal_tokenizer()).")

    def get_index_of_word(self, word: str) -> int:
        """
        Return the index of the word in the tokenizer's vocabulary. If the word
        does not exist, the index of the special token [UNK] is returned.
        """
        if word in self.word_to_index_mapping:
            return self.word_to_index_mapping[word]
        else:
            return self.word_to_index_mapping["[UNK]"]

    def get_word_of_index(self, index: int) -> str:
        """
        Return the word associated to the given index in this tokenizer's
        vocabulary.
        """
        try:
            return self.index_to_word_mapping[index]
        except KeyError:
            raise IndexError(f"Indexing into this tokenizer's vocabulary range "
                             f"between 0 and {len(self.vocabulary) - 1} since "
                             f"there are {self.num_words} words. Got: {index}")

    def encode(self, sentence: str, split_token=" ") -> List[int]:
        """
        Return a list containing the index of each word in <sentence> in this
        tokenizer's vocabulary.

        <sentence> should be pre-processed such that splitting down
        <split_token> will generate a list of separate tokens.
        """
        self.warn_use_unlocked()
        words = sentence.split(split_token)
        return [self.start_token] + \
               [self.get_index_of_word(word) for word in words] + \
               [self.end_token]

    def decode(self, sentence: List[int]) -> str:
        """
        Return the string representing the words contained in the given encoded
        sentence. The given <sentence> should be a list of tokens.
        """
        words = [self.get_word_of_index(index) for index in sentence]
        return " ".join(words) + "."

    @property
    def num_words(self) -> int:
        """
        Return the number of unique words in the vocabulary of this tokenizer.
        """
        return self._num_words

    @property
    def word_to_index_mapping(self) -> Dict[str, int]:
        """
        Getter for the attribute word_to_index_mapping.
        """
        return self._word_to_idx

    @property
    def index_to_word_mapping(self) -> Dict[int, str]:
        """
        Getter for the attribute index_to_word_mapping.
        """
        return self._idx_to_word

    @property
    def vocabulary(self) -> List[str]:
        """
        Return the entire vocabulary of this tokenizer (getter for the
        attribute vocabulary).
        """
        return list(self._all_words)

    @property
    def word_counts(self) -> Dict[str, int]:
        """
        Return a mapping of all the counts of each word in the vocabulary. This
        is the number of occurrences of each word of the dataset which the
        tokenizer was used to train with.
        """
        return self._word_counts

    @property
    def is_locked(self) -> bool:
        """
        Return whether or not this tokenizer is locked, meaning that its
        vocabulary dictionaries can no longer be edited.
        """
        return self._is_locked

    @num_words.setter
    def num_words(self, *args, **kwargs) -> None:
        """
        num_words should not be set explicitly and should be internally set by
        the tokenizer. Attempting to modify this will raise an error. Note that
        the client can actually still change self._num_words but if this occurs
        there is nothing I can do about this.
        """
        raise AttributeError("This attribute should not be explicitly set.")

    @is_locked.setter
    def is_locked(self, *args, **kwargs) -> None:
        """
        is_locked should not be set explicitly and should be internally set by
        the tokenizer. Attempting to modify this will raise an error. Note that
        the client can actually still change self._is_locked but if this occurs
        there is nothing I can do about this.
        """
        raise AttributeError("This attribute should not be explicitly set.")

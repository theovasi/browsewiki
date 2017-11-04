""" This module offers a number tools for processing modern Greek text. """

def normalize(text):
    """ Remove intonation from Greek text.

        Args:
            text (str): The text that will be normalized.
        Returns:
            text (str): The original text in lowercase without intonation.
    """
    vowel_dict = {
        u'ά': u'α',
        u'έ': u'ε',
        u'ή': u'η',
        u'ί': u'ι',
        u'ό': u'ο',
        u'ύ': u'υ',
        u'ώ': u'ω',
        u'ϊ': u'ι',
        u'ϋ': u'υ'
    }
    text = text.lower()
    for key, value in vowel_dict.items():
        text = text.replace(key, value)

    return text

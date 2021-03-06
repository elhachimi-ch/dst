import csv
import pickle
import re
from os import listdir
from os.path import isfile, join
from time import sleep
import math
import numpy as np
import unidecode
from emoji import EMOJI_DATA
from textblob import TextBlob
#from stringdist.pystringdist.levenshtein import levenshtein as ed
import nltk
import calendar
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class Lib:
    @staticmethod
    def substring(string_in):
        return set([string_in[i: j] for i in range(len(string_in))
                for j in range(i + 1, len(string_in) + 1) if len(string_in[i: j]) > 0])
    @staticmethod
    def write_liste_csv(liste_ligne_csv, file_name='data/out.csv', delimiter=',', quotechar='`'):
        f = open(file_name, 'w+', newline='', encoding='utf-8')
        writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
        for p in liste_ligne_csv:
            writer.writerow(p)
        f.close()
    @staticmethod
    def julian_date_to_mmddyyy(year,julian_day):
        month = 1
        while julian_day - calendar.monthrange(year,month)[1] > 0 and month <= 12:
            julian_day = julian_day - calendar.monthrange(year,month)[1]
            month = month + 1
        print(month,julian_day,year)

    @staticmethod
    def load_model(model_path):
        return joblib.load(open(model_path, 'rb'))
    
    @staticmethod
    def save_object(o, object_path):
        pickle.dump(o, open(object_path, 'wb'))
        
    @staticmethod
    def load_object(obj_path):
        return pickle.load(open(obj_path, 'rb'))

    # verify condition on all list elements all(map(is_arabic, city))
    # nCk
    @staticmethod
    def c_n_k(n, k):
        """
        A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
        """
        if 0 <= k <= n:
            ntok = 1
            ktok = 1
            for t in range(1, min(k, n - k) + 1):
                ntok *= n
                ktok *= t
                n -= 1
            return ntok // ktok
        else:
            return 0

    # factorielle
    @staticmethod
    def factorial(x):
        x = int(x)
        result = 1
        while x > 1:
            result = result * x
            x = x - 1
        return result
    
    @staticmethod
    def stemming(document, language_or_custom_stemmer_as_lambda='en'):
        tokens = word_tokenize(document)
        result = []
        if language_or_custom_stemmer_as_lambda == 'en':
            for p in tokens:
                result.append(nltk.stem.PorterStemmer().stem(p))
        elif language_or_custom_stemmer_as_lambda == 'fr':
            for p in tokens:
                result.append(nltk.stem.SnowballStemmer('french').stem(p))
        elif language_or_custom_stemmer_as_lambda == 'ar':
            for p in tokens:
                result.append(nltk.stem.SnowballStemmer('arabic').stem(p))
        else:
            result.append(language_or_custom_stemmer_as_lambda(p))
        return ' '.join(result) 

    # Fonction partie enti??re E()
    @staticmethod
    def partie_entiere(x):
        if x == int(x):
            return x
        elif x >= 0:
            return int(x)
        else:
            return -Lib.partie_entiere(-x) - 1

    # Algorithme d'Euclide pour le pgcd
    @staticmethod
    def pgcd_iterative(a, b):
        while a % b != 0:
            a, b = b, a % b
        return b

    @staticmethod
    def pgcd_recursive(a, b):
        if a % b == 0:
            return b
        else:
            return Lib.pgcd_recursive(b, a % b)

    # plus petit commun multiple
    def ppmc(a, b):
        return (a * b) / Lib.pgcd_recursive(a, b)

    # verifier premier
    @staticmethod
    def is_premier(n):
        if n == 0 or n == 1:
            return False
        else:
            for i in range(2, int(math.sqrt(n))):
                if n % i == 0:
                    return False
            return True

    # decomposition en nombre premier
    @staticmethod
    def decompsition_premier(n):
        liste = []
        if Lib.is_premier(n) or n == 1 or n == 0:
            liste.append((n, 1))
        else:
            i = 2
            while n // i != 0:
                j = 0
                if n % i == 0:
                    while n % i == 0:
                        j += 1
                        n = n // i
                    liste.append((i, j))
                else:
                    i += 1
        return liste

    # from scipy.comb(), but MODIFIED!
    @staticmethod
    def c_n_k_scipy(n, k):
        if (k > n) or (n < 0) or (k < 0):
            return 0
        top = n
        val = 1
        while top > (n - k):
            val *= top
            top -= 1
        n = 1
        while n < k + 1:
            val /= n
            n += 1
        return val

    @staticmethod
    def is_arabic(string_in):
        ad = AlphabetDetector()
        return ad.is_arabic(string_in)

    @staticmethod
    def is_latin(string_in):
        ad = AlphabetDetector()
        return ad.is_latin(string_in)

    @staticmethod
    def replace2or_more_char_by_2(string_in):
        return re.sub(r'([a-zA-Z1-9??-??])\1+', r'\1\1', string_in)

    @staticmethod
    def replace2or_more_char_by_1(string_in):
        return re.sub(r'([a-zA-Z1-9??-??])\1+', r'\1', string_in)

    @staticmethod
    def is_fr_wolf(string_in, french_dict_instance):
        try:
            is_in = french_dict_instance.synsets(string_in)
            if len(is_in) > 0:
                return True
        except NameError:
            return False

    @staticmethod
    def is_fr_or_en(string_in):
        if len(string_in) < 3:
            string_in += "   "
        return TextBlob(string_in).detect_language() == 'fr' or TextBlob(string_in).detect_language() == 'en'

    @staticmethod
    def is_single_word(string_in):
        if re.match(r".+\s.+", string_in):
            return False
        return True

    @staticmethod
    def eliminate_multiple_whitespace(string_in):
        s = string_in.strip()
        return re.sub(r"\s+", " ", s)

    @staticmethod
    def eliminate_punctuation(string_in):
        s = string_in.strip()
        s = Lib.replace_apostrophes_and_points_by_space(s)
        s = re.sub(r'[!"#$%&()*+,-./\\:;<=>?@[\]^_`{|}~]+', '', s)
        return Lib.eliminate_multiple_whitespace(s)

    @staticmethod
    def eliminate_all_whitespaces(string_in):
        string_in = str(string_in)
        s = string_in.strip()
        return re.sub(r"\s+", "", s)

    @staticmethod
    def eliminate_all_digits(string_in):
        s = string_in.strip()
        return re.sub(r"\d+", "", s)

    @staticmethod
    def read_text_file_as_list(path, with_anti_slash=False):
        f = open(path, "r+", encoding='utf-8')
        data = f.readlines()
        if not with_anti_slash:
            for i in range(len(data)):
                data[i] = re.sub(r"\n", "", data[i]).strip()
        return data

    @staticmethod
    def write_liste_in_file(liste, path='data/out.txt'):
        f = open(path, 'w+', encoding='utf-8')
        liste = list(map(str, liste))
        for i in range(len(liste)-1):
            liste[i] = str(liste[i]) + "\n" 
        f.writelines(liste)

    @staticmethod
    def strip_and_split(string_in):
        return string_in.strip().split()

    @staticmethod
    def to_upper_file_text(path_source, path_destination):
        data = Lib.read_text_file(path_source)
        la = []
        for line in data:
            la.append(line.upper())
        Lib.write_liste_in_file(path_destination, la)

    @staticmethod
    def no_accent_file_text(path_source, path_destination):
        data = Lib.read_text_file(path_source)
        la = []
        for line in data:
            la.append(Lib.no_accent(line))
        Lib.write_liste_in_file(path_destination, la)

    @staticmethod
    def edit_distance(term_a, term_b):
        term_a = Lib.replace2or_more_char_by_2(term_a)
        term_b = Lib.replace2or_more_char_by_2(term_b)
        return Lib.edit_dist_dp(term_a, term_b)

    @staticmethod
    def edit_distance_without_voyelle(term_a, term_b):
        term_a = Lib.replace2or_more_char_by_2(term_a)
        term_b = Lib.replace2or_more_char_by_2(term_b)
        term_a = re.sub(r'[aeiouy]', '', term_a)
        term_b = re.sub(r'[aeiouy]', '', term_b)
        return Lib.edit_distance(term_a, term_b)

    @staticmethod
    def edit_dist_dp(str1, str2, m, n):
        # Create a table to store results of sub problems
        dp = np.zeros((n + 1, m + 1))
        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):
                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

                    # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                    dp[i - 1][j],  # Remove
                                    dp[i - 1][j - 1])  # Replace

        return dp[m][n]

    @staticmethod
    def write_line_in_file(line, path='data/latin_comments.csv', with_anti_slash=True):
        f = open(path, "a+", encoding='utf-8')

        if with_anti_slash:
            f.write(str(line) + "\n")
        else:
            f.write(line)

    # nltk.download('stopwords')
    @staticmethod
    def load_arabic_stop_words():
        liste = Lib.read_text_file('data/arabic_stop_words.csv')
        return liste

    @staticmethod
    def load_stop_words(language):
        return stopwords.words(language)

    @staticmethod
    def remove_stopwords(document, language_or_stopwords_list='english'):
        document = str.lower(document)
        if isinstance(language_or_stopwords_list, list) is True:
            stopwords = language_or_stopwords_list
        elif language_or_stopwords_list == 'arabic':
            stopwords = Lib.read_text_file_as_list('data/arabic_stopwords.csv')
        elif language_or_stopwords_list == 'french':
            stopwords = Lib.read_text_file_as_list('data/french_stopwords.csv')
        else:
            nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(language_or_stopwords_list)
        words = word_tokenize(document)
        clean_words = []
        for w in words:
            if w not in stopwords:
                clean_words.append(w)
        return ' '.join(clean_words)

    @staticmethod
    def no_accent(string_in):
        if not Lib.is_arabic(string_in):
            return unidecode.unidecode(string_in)
        return string_in

    @staticmethod
    def no_accent(string_in):
        # \s*[A-Za-z\u00C0-\u00FF]+
        if not Lib.is_arabic(string_in):
            return unidecode.unidecode(string_in)
        return string_in

    @staticmethod
    def list_to_string(liste):
        liste_b = []
        for p in liste:
            if type(p) is not str:
                liste_b.append(str(p))
            else:
                liste_b.append(p)
        return "".join(liste_b)

    @staticmethod
    def check_all_elements_type(list_to_check, types_tuple):
        return all(isinstance(p, types_tuple) for p in list_to_check)

    @staticmethod
    def list_all_files_in_folder(folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    @staticmethod
    def translate(string_in, langue_dest='fr'):
        
        translator = Translator()
        translated = translator.translate(string_in, dest=langue_dest)
        return translated.text

    @staticmethod
    def get_mnist_as_dataframe():
        """image_list = ch.get_reshaped_matrix(np.array([ch.get_reshaped_matrix(p, (1, 28 * 28)) for p in x_train]),
                                            (x_train.shape[0], 28 * 28))"""
    @staticmethod
    def is_only_digits_filter(comment):
        comment = str(comment)
        if re.match(r'^\d+$', comment):
            return True
        return False

    @staticmethod
    def is_empty_world(string_in):
        if re.match(r'^\s+$', string_in):
            return True
        return False

    @staticmethod
    def is_empty_line(string_in):
        string_in = str(string_in)
        if re.match(r'^\s*$', string_in):
            return True
        return False

    @staticmethod
    def is_only_emojis_filter(comment):
        return not all([p in UNICODE_EMOJI for p in comment])

    @staticmethod
    def is_digit_with_emojis_filter(comment):
        comment = str(comment)
        comment = Lib.eliminate_all_whitespaces(comment)
        return all([p in UNICODE_EMOJI or re.match(r'\d', p)  or re.match(r'\W', p) for p in comment])

    @staticmethod
    def eliminate_stop_digits(comment):
        return re.sub(r'\b\d+\b', '', comment)

    @staticmethod
    def is_all_arabic(document):
        document = Lib.eliminate_all_whitespaces(document)
        document = Lib.eliminate_all_digits(document)
        return all([Lib.is_arabic(p) for p in document])

    @staticmethod
    def get_all_words(comment):
        # comment = eliminate_stop_digits(comment)
        comment = Lib.eliminate_punctuation(comment)
        return comment.split(' ')

    @staticmethod
    def fr_or_eng_filter(word):
        return Lib.is_fr_or_en(word)

    @staticmethod
    def load(path_data):
        return joblib.load(open(path_data, 'rb'))

    @staticmethod
    def binary_search(input_list, item):
        first = 0
        last = len(input_list) - 1
        while(first <= last):
            mid = (first + last) // 2
            if input_list[mid] == item :
                return True
            elif item < input_list[mid]:
                last = mid - 1
            else:
                first = mid + 1	
        return False

    @staticmethod
    def fib(n):
        if n== 0 or n== 1:
            return n
        return Lib.fib (n- 1) + Lib.fib (n- 2) 

    @staticmethod
    def replace2or_more_char_by_1_sauf_hh(string_in):
        return re.sub(r'([0-gi-z??-????-??])\1+', r'\1', string_in)

    @staticmethod
    def replace2or_more_h_by_2h(string_in):
        return re.sub(r'([h??])\1+', r'\1\1', string_in)

    @staticmethod
    def is_stop_digits(string_in):
        return re.match(r'\b\d+\b', string_in) is not None

    @staticmethod
    def eliminate_emoji(string_in):
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]+', flags=re.UNICODE)
        word = RE_EMOJI.sub(r'', string_in)
        return re.sub(r'???+|??????+|???+', '', word)

    @staticmethod
    def eliminate_d2a_stop_words(string_in):
        D2A_STOP_WORDS = ['el', 'al', 'mi', 'ya', 'rah', 'ylh', 'hada', 'wa', 'ila', 'l', 'hadchi', 'ana', 'nti', 'howa', 'ntoma', 'lina', 'likom', 'lihom', 'gha', 'ghi', 'dial', 'dialo', 'dyl', 'diyalkom', 'dialhom', 'dyal', 'deyal']
        words = string_in.split(' ')
        for w in words:
            if w in D2A_STOP_WORDS:
                words.remove(w)
        return ' '.join(words)

    @staticmethod
    def replace_apostrophes_and_points_by_space(string_in):
        string_in = Lib.replace2or_more_appostrophe_and_point_by_1(string_in)
        string_in = re.sub(r'\'', ' ', string_in)
        string_in = re.sub(r'"', ' ', string_in)
        string_in = re.sub(r'\.', ' ', string_in)
        string_in = re.sub(r'???', ' ', string_in)
        return string_in

    @staticmethod
    def write_row_csv(row_liste, file_name='data/latin_comments.csv', delimiter=',', quotechar='`'):
        file = open(file_name, 'a+', newline='', encoding='utf-8')
        writer = csv.writer(file, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
        writer.writerow(row_liste)

    @staticmethod
    def read_csv(file_path, delimiter=','):
        f = open(file_path, 'r+', encoding='utf-8')
        reader = csv.reader(f, delimiter=delimiter)
        return reader

    @staticmethod
    def detect_lang(text):
        return TextBlob(text).detect_language()

    @staticmethod
    def replace2or_more_appostrophe_and_point_by_1(string_in):
        return re.sub(r'([\'\."])\1+', r'\1', string_in)

    @staticmethod
    def country_name_to_iso3(o):
        iso = o
        try:
            iso = Lib.country_name_to_country_alpha3(o)
        except KeyError:
            return iso
        return iso
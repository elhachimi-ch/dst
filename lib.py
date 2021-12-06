# coding=utf-8
import csv
import pickle
import re
from os import listdir
from os.path import isfile, join
import math
from textblob import TextBlob
from pycountry_convert import country_name_to_country_alpha3

def write_liste_csv(liste_ligne_csv, file_name='data/out.csv', delimiter=',', quotechar='`'):
    f = open(file_name, 'w+', newline='', encoding='utf-8')
    writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
    for p in liste_ligne_csv:
        writer.writerow(p)
    f.close()

def save_object(o, object_path):
    pickle.dump(o, open(object_path, 'wb'))

def load_object(obj_path):
    return pickle.load(open(obj_path, 'rb'))

# verify condition on all list elements all(map(is_arabic, city))
# nCk
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
def factorial(x):
    x = int(x)
    result = 1
    while x > 1:
        result = result * x
        x = x - 1
    return result

# Fonction partie entière E()
def partie_entiere(x):
    if x == int(x):
        return x
    elif x >= 0:
        return int(x)
    else:
        return -partie_entiere(-x) - 1

# Algorithme d'Euclide pour le pgcd
def pgcd_iterative(a, b):
    while a % b != 0:
        a, b = b, a % b
    return b

def pgcd_recursive(a, b):
    if a % b == 0:
        return b
    else:
        return pgcd_recursive(b, a % b)

# plus petit commun multiple
def ppmc(a, b):
    return (a * b) / pgcd_recursive(a, b)

# verifier premier
def is_premier(n):
    if n == 0 or n == 1:
        return False
    else:
        for i in range(2, int(math.sqrt(n))):
            if n % i == 0:
                return False
        return True

# decomposition en nombre premier
def decompsition_premier(n):
    liste = []
    if is_premier(n) or n == 1 or n == 0:
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

def read_text_file(path, with_anti_slash=False):
    f = open(path, "r+", encoding='utf-8')
    data = f.readlines()
    if not with_anti_slash:
        for i in range(len(data)):
            data[i] = re.sub(r"\n", "", data[i]).strip()
    return data

def write_liste_in_file(liste, path='data/out.txt'):
    f = open(path, 'w+', encoding='utf-8')
    liste = list(map(str, liste))
    for i in range(len(liste)-1):
        liste[i] = str(liste[i]) + "\n" 
    f.writelines(liste)

def strip_and_split(string_in):
    return string_in.strip().split()

def to_upper_file_text(path_source, path_destination):
    data = read_text_file(path_source)
    la = []
    for line in data:
        la.append(line.upper())
    write_liste_in_file(path_destination, la)


def write_line_in_file(line, path='data/latin_comments.csv', with_anti_slash=True):
    f = open(path, "a+", encoding='utf-8')

    if with_anti_slash:
        f.write(str(line) + "\n")
    else:
        f.write(line)

def list_to_string(liste):
    liste_b = []
    for p in liste:
        if type(p) is not str:
            liste_b.append(str(p))
        else:
            liste_b.append(p)
    return "".join(liste_b)

def check_all_elements_type(list_to_check, types_tuple):
    return all(isinstance(p, types_tuple) for p in list_to_check)

def list_all_files_in_folder(folder_path):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

def get_mnist_as_dataframe():
    """image_list = ch.get_reshaped_matrix(np.array([ch.get_reshaped_matrix(p, (1, 28 * 28)) for p in x_train]),
                                        (x_train.shape[0], 28 * 28))"""

def is_empty_line(string_in):
    string_in = str(string_in)
    if re.match(r'^\s*$', string_in):
        return True
    return False

def write_row_csv(row_liste, file_name='data/latin_comments.csv', delimiter=',', quotechar='`'):
    file = open(file_name, 'a+', newline='', encoding='utf-8')
    writer = csv.writer(file, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
    writer.writerow(row_liste)

def read_csv(file_path, delimiter=','):
    f = open(file_path, 'r+', encoding='utf-8')
    reader = csv.reader(f, delimiter=delimiter)
    return reader

def detect_lang(text):
    return TextBlob(text).detect_language()

def country_name_to_iso3(o):
    iso = o
    try:
        iso = country_name_to_country_alpha3(o)
    except KeyError:
        return iso
    return iso
# coding=utf-8
import csv
import pickle
import re
from os import listdir
from os.path import isfile, join
import math
from collections import Counter

class Lib:
    """ Lib class is a static class that serves other components with implementations of auxiliary algorithms . 
    """
    
    def __init__(self):
        return 0
    
    @staticmethod
    def write_list_as_csv(liste_ligne_csv, file_name='data/out.csv', delimiter=',', quotechar='`'):
        """[summary]

        Args:
            liste_ligne_csv ([type]): [description]
            file_name (str, optional): [description]. Defaults to 'data/out.csv'.
            delimiter (str, optional): [description]. Defaults to ','.
            quotechar (str, optional): [description]. Defaults to '`'.
        """
        f = open(file_name, 'w+', newline='', encoding='utf-8')
        writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC, quotechar=quotechar)
        for p in liste_ligne_csv:
            writer.writerow(p)
        f.close()

    @staticmethod
    def save_object(o, object_path):
        pickle.dump(o, open(object_path, 'wb'))

    @staticmethod
    def load_object(obj_path):
        return pickle.load(open(obj_path, 'rb'))

    @staticmethod
    def c_n_k(n, k):
        """A fast way to calculate binomial coefficients by Andrew Dalke (contrib).

        Args:
            n (int): N
            k (int): K

        Returns:
            int:  combination of k from n
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

    # Fonction partie entière E()
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
    @staticmethod
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
    def decompsition_to_prime(n=99):
        """Decompse a number to prime numbers

        Args:
            n (int, optional): number to decompose. Defaults to 99.

        Returns:
            list: list of prime numbers and their power
        """
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
    def read_text_file(path, with_anti_slash=False):
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
    def write_line_in_file(line, path='data/latin_comments.csv', with_anti_slash=True):
        f = open(path, "a+", encoding='utf-8')

        if with_anti_slash:
            f.write(str(line) + "\n")
        else:
            f.write(line)

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
    def get_mnist_as_dataframe():
        """image_list = ch.get_reshaped_matrix(np.array([ch.get_reshaped_matrix(p, (1, 28 * 28)) for p in x_train]),
                                            (x_train.shape[0], 28 * 28))"""

    @staticmethod
    def is_empty_line(string_in):
        string_in = str(string_in)
        if re.match(r'^\s*$', string_in):
            return True
        return False

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
    def is_palindrome(str):
        return str == str[::-1]
    
    @staticmethod
    def most_frequent(list_in):
        return max(set(list_in), key=list.count)
    
    @staticmethod
    def is_anagram(string1, string2):
        return Counter(string1) == Counter(string2)
    
    @staticmethod
    def remove_duplicate(list_in):
        return list(set(list_in))
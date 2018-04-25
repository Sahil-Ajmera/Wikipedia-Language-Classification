"""
Author : Sahil Ajmera
Wikipedia Language classification using Decision tree and Adaboost
"""
import math
from helper import *
import pickle
import sys


class classifylanguage:
    __slots__ = 'train_file'

    def __init__(self):
        print()

    def predict_dt(self, hypothesis, file):
        """
        Does the prediction for the decision tree
        :param hypothesis:Input decision tree
        :param file:Input test file
        :return:None
        """

        # Load decision tree model
        object = pickle.load(open(hypothesis, 'rb'))
        file_open = open(file)
        sentence_list = []
        counter = 0
        sentence = ''

        # Extract 15-word samples from the test file
        for line in file_open:
            words = line.split()

            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
                    sentence += word
                    sentence_list.append(sentence)
                    sentence = ''
                    counter = 0

        attribute1 = []
        attribute2 = []
        attribute3 = []
        attribute4 = []
        attribute5 = []
        attribute6 = []
        attribute7 = []
        attribute8 = []
        attribute9 = []
        attribute10 = []
        attribute11 = []

        # Based on the sentences fill the values for the attributes
        for line in sentence_list:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.presence_of_van(line))
            attribute5.append(self.presence_of_de_het(line))
            attribute6.append(self.check_for_een(line))
            attribute7.append(self.check_for_en(line))
            attribute8.append(self.check_for_common_dutch_words(line))
            attribute9.append(self.check_for_common_english_words(line))
            attribute10.append(self.presence_of_a_an_the(line))
            attribute11.append(self.check_presence_of_and(line))

        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        attributes.append(attribute8)
        attributes.append(attribute9)
        attributes.append(attribute10)
        attributes.append(attribute11)

        statement = 0

        # For every statement run through the decision tree to find out the langauge for the examples
        for sentence in sentence_list:
            object_temp = object
            while type(object_temp.value) != str:
                value = attributes[object_temp.value][statement]
                if value is True:
                    object_temp = object_temp.left
                else:
                    object_temp = object_temp.right
            print(object_temp.value)
            statement = statement + 1


    def number_of_diff_values(self, values, total):
        """
        To check for total positive or total negative examples in a set
        :param values:Input set
        :param total:Indices
        :return:Return based on whether total negative or positive examples
        """
        value = values[total[0]]
        for i in total:
            if value != values[i]:
                return 10
        return 0

    def entropy(self, value):
        """
        Entropy function
        :param value:Input value
        :return:Calculate entropy and return
        """
        if value == 1:
            return 0
        return (-1) * (value * math.log(value, 2.0) + (1 - value) * math.log((1 - value), 2.0))



    def collect_data_dt(self, example_file, hypothesis_file):
        """
        Collection of data and calling the required functions
        :param example_file:Training file
        :param hypothesis_file:File to which hypothesis is to be written
        :return:None
        """

        statements, results = self.gather_data(example_file)
        print(len(results))
        attribute1 = []
        attribute2 = []
        attribute3 = []
        attribute4 = []
        attribute5 = []
        attribute6 = []
        attribute7 = []
        attribute8 = []
        attribute9 = []
        attribute10 = []
        attribute11 = []

        # For each line set the values for features for that line
        for line in statements:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.presence_of_van(line))
            attribute5.append(self.presence_of_de_het(line))
            attribute6.append(self.check_for_een(line))
            attribute7.append(self.check_for_en(line))
            attribute8.append(self.check_for_common_dutch_words(line))
            attribute9.append(self.check_for_common_english_words(line))
            attribute10.append(self.presence_of_a_an_the(line))
            attribute11.append(self.check_presence_of_and(line))

        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        attributes.append(attribute8)
        attributes.append(attribute9)
        attributes.append(attribute10)
        attributes.append(attribute11)

        number_lst = []
        for i in range(len(results)):
            number_lst.append(i)

        # To keep track of attributes splitted along a path
        seen = []
        root = helper(attributes,None, results, number_lst, 0, None, None)

        # Calling decision tree function here
        value = self.train_decision_tree(root, attributes, seen, results, number_lst, 0, None)


        # Dumping the hypothesis to a file using pickle
        filehandler = open(hypothesis_file, 'wb')
        pickle.dump(root, filehandler)

    def check_for_0_gain(self,values):
        """
        If 0 gain then return
        :param values:values of gain for the level
        :return:False if the max gain is 0
        """
        if max(values) == 0:
            return False

    def train_decision_tree(self, root, attributes, seen, results, total_results, depth, prevprediction):
        """
        Decides on the best splitting attribute for a given depth , make a node for that and connects it with
        two nodes containing the left and right childs for the given so called root node
        :param root: The node that is being considered right now
        :param attributes:Total set of attributes and their values
        :param seen:Every node that has been seen till now
        :param results:Final results in a list
        :param total_results:Index of examples that are there at this level
        :param depth:Level in consideration
        :param prevprediction:Prediction made before this depth
        :return:None
        """

        # If depth is reached return the plurality of the remaining set
        if depth == len(attributes) - 1:
            counten = 0
            countnl = 0
            for index in total_results:
                if results[index] is 'en':
                    counten = counten + 1
                elif results[index] is 'nl':
                    countnl = countnl + 1
            if counten > countnl:
                root.value = 'en'
                print('en')
            else:
                root.value = 'nl'
                print('nl')

        # If there are no examples left return the prediction made at the previous level
        elif len(total_results) == 0:
            root.value = prevprediction
            print(prevprediction)

        # If there are only positive or only negative examples left return the prediction directly from the plurality
        elif self.number_of_diff_values(results, total_results) == 0:
            root.value = results[total_results[0]]
            print( results[total_results[0]])

        # If all the attributes have been used for splitting along a given path return the prediction of the set of examples
        elif len(attributes) == len(seen):
            counten = 0
            countnl = 0
            for index in total_results:
                if results[index] is 'en':
                    counten = counten + 1
                elif results[index] is 'nl':
                    countnl = countnl + 1
            if counten > countnl:
                root.value = 'en'
            else:
                root.value = 'nl'

        # Find the attribute to split on
        else:
            gain = []
            results_en = 0
            results_nl = 0

            # Take the total number of positive and negative examples at this level
            for index in total_results:
                if results[index] == 'en':
                    results_en = results_en + 1
                else:
                    results_nl = results_nl + 1
            # For each attribute
            for index_attribute in range(len(attributes)):

                # Check if it has already been used for splitting so , no gain in splitting over it again
                if index_attribute in seen:
                    gain.append(0)
                    continue

                # Else see for the best splitting attribute
                else:
                    count_true_en = 0
                    count_true_nl = 0
                    count_false_en = 0
                    count_false_nl = 0

                    for index in total_results:

                        if attributes[index_attribute][index] is True and results[index] == 'en':
                            count_true_en = count_true_en + 1
                        elif attributes[index_attribute][index] is True and results[index] == 'nl':
                            count_true_nl = count_true_nl + 1
                        elif attributes[index_attribute][index] is False and results[index] == 'en':
                            count_false_en = count_false_en + 1
                        elif attributes[index_attribute][index] is False and results[index] == 'nl':
                            count_false_nl = count_false_nl + 1

                    # If only positive or only negative examples remain at a particular point , no point in splitting
                    if (count_true_nl + count_true_en == 0) or (count_false_en + count_false_nl == 0):
                        gain_for_attribute = 0
                        gain.append(gain_for_attribute)
                        continue
                    # Handliing certain outlier conditions
                    if count_true_en == 0:
                        rem_true_value = 0
                        # rem_false_value = 0
                        rem_false_value = (
                                          (count_false_en + count_false_nl) / (results_nl + results_nl)) * self.entropy(
                            count_false_en / (count_false_nl + count_false_en))
                    elif count_false_en == 0:
                        rem_false_value = 0
                        #rem_true_value = 0
                        rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en)) * self.entropy(
                            count_true_en / (count_true_nl + count_true_en))
                    else:
                        rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en)) * self.entropy(
                            count_true_en / (count_true_nl + count_true_en))

                        rem_false_value = (
                                          (count_false_en + count_false_nl) / (results_nl + results_en)) * self.entropy(
                            count_false_en / (count_false_nl + count_false_en))

                    # Find the gain for each attribute
                    gain_for_attribute = self.entropy(results_en / (results_en + results_nl)) - (rem_true_value +
                                                                                                 rem_false_value)
                    gain.append(gain_for_attribute)
           # Check if the max gain is 0 then return back as no more gain possible along this path
            continue_var = self.check_for_0_gain(gain)
            if continue_var is False:
                root.value = prevprediction
                print(root.value)
                return

            # Select the max gain attribute
            max_gain_attribute = gain.index(max(gain))

            seen.append(max_gain_attribute)

            index_True = []
            index_False = []

            # Separate out true and false portion for the found out max gain attribute
            for index in total_results:
                if attributes[max_gain_attribute][index] is True:
                    index_True.append(index)
                else:
                    index_False.append(index)

            # Prediction at this stage
            prediction_at_this_stage = ''

            if results_en > results_nl:
                prediction_at_this_stage = 'en'
            else:
                prediction_at_this_stage = 'nl'

            bool_false = False
            bool_true = True
            root.value = max_gain_attribute

            # Make left portion for the max gain attribute

            left_obj = helper(attributes, None,results, index_True, depth + 1,
                              prediction_at_this_stage, bool_true)
            # Make right portion for the max gain attribute

            right_obj = helper(attributes,None, results, index_False, depth + 1,
                               prediction_at_this_stage, bool_false)
            root.left = left_obj
            root.right = right_obj
            # Recurse left and right portions
            self.train_decision_tree(left_obj,attributes,seen,results,index_True,depth + 1,prediction_at_this_stage)
            self.train_decision_tree(right_obj,attributes,seen,results,index_False,depth + 1,prediction_at_this_stage)

            del seen[-1]



    def gather_data(self, file):
        """
        Gathers data from the train.dat file for training
        :param file:input training file
        :return:list of statements and final predictions
        """

        # Open file
        file_details = open(file, encoding="utf-8-sig")
        all_details = ''
        for file_lines in file_details:
            all_details += file_lines

        # Get all the statements
        statements = all_details.split('|')
        all_data_stripped_space = all_details.split()

        for index in range(len(statements)):
            if index < 1:
                continue
            statements[index] = statements[index][:-4]
        statements = statements[1:]

        # Get all the results

        results = []
        pointer = 0
        for values in all_data_stripped_space:
            if values.startswith('nl|') or values.startswith('en|'):
                results.insert(pointer, values[0:2])
                pointer = pointer + 1

        return statements, results

    def collect_data_ada(self, example_file, hypothesis_file):
        """
        Collection of data for Adaboost , collection of stumps formed
        :param example_file:Training file for training
        :param hypothesis_file:Hypothesis file to write the set of hypothesis
        :return:None
        """
        # Collection of examples from the training file
        statements, results = self.gather_data(example_file)
        weights = [1 / len(statements)] * len(statements)

        # Number of hypothesis
        number_of_decision_stumps = 50

        attribute1 = []
        attribute2 = []
        attribute3 = []
        attribute4 = []
        attribute5 = []
        attribute6 = []
        attribute7 = []
        attribute8 = []
        attribute9 = []
        attribute10 = []
        attribute11 = []

        # For each 15-word line in training set decide on the value of features
        for line in statements:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.presence_of_van(line))
            attribute5.append(self.presence_of_de_het(line))
            attribute6.append(self.check_for_een(line))
            attribute7.append(self.check_for_en(line))
            attribute8.append(self.check_for_common_dutch_words(line))
            attribute9.append(self.check_for_common_english_words(line))
            attribute10.append(self.presence_of_a_an_the(line))
            attribute11.append(self.check_presence_of_and(line))


        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        attributes.append(attribute8)
        attributes.append(attribute9)
        attributes.append(attribute10)
        attributes.append(attribute11)

        number_lst = []
        stump_values = []

        hypot_weights = [1] * number_of_decision_stumps

        # Set contining indices of all the examples
        for i in range(len(results)):
            number_lst.append(i)

        # Initialization of the root


        # Adaboost algorithm for training
        for hypothesis in range(0, number_of_decision_stumps):

            root = helper(attributes, None, results, number_lst, 0, None, None)
            # For every hypothesis index generate a hypotesis to be added
            stump = self.return_stump(0, root, attributes, results, number_lst, weights)
            error = 0
            correct = 0
            incorrect = 0
            for index in range(len(statements)):

                # Check for number of examples that do not match with hypothesis output value and update error value
                if self.prediction(stump, statements[index], attributes, index) != results[index]:
                    error = error + weights[index]
                    incorrect = incorrect + 1

            for index in range(len(statements)):

                # Check for number of examples that do mathc with the hypothesis output value and update weights of examples
                if self.prediction(stump, statements[index], attributes, index) == results[index]:
                    weights[index] = weights[index] * error / (1 - error)
                    correct = correct + 1
            total = 0
            # Calculation for normalization
            for weight in weights:
                total += weight
            for index in range(len(weights)):
                weights[index] = weights[index] / total

            # Updated values for hypothseis weight
            hypot_weights[hypothesis] = math.log(((1 - error) / (error)),2)
            stump_values.append(stump)

        # Dump the set of generated hypothesis
        filehandler = open(hypothesis_file, 'wb')
        pickle.dump((stump_values, hypot_weights), filehandler)

    def prediction(self, stump, statement, attributes, index):
        """
        For predicting the stump and the result it will give
        :param stump:Input decision stump
        :param statement:Input statement
        :param attributes:Set of attributes/features we have decided upon
        :param index:Index of the statement that is inputted
        :return:Return final prediction from the stump
        """
        attribute_value = stump.value
        if attributes[attribute_value][index] is True:
            return stump.left.value
        else:
            return stump.right.value

    def return_stump(self, depth, root, attributes, results, total_results, weights):
        """
        Function returns a decision stump
        :param depth:Depth of the tree we are at
        :param root:
        :param attributes:
        :param results:
        :param total_results:
        :param weights:
        :return:
        """
        gain = []
        results_en = 0
        results_nl = 0
        for index in total_results:
            if results[index] == 'en':
                results_en = results_en + 1 * weights[index]
            else:
                results_nl = results_nl + 1 * weights[index]

        for index_attribute in range(len(attributes)):
            count_true_en = 0
            count_true_nl = 0
            count_false_en = 0
            count_false_nl = 0
            for index in total_results:
                if attributes[index_attribute][index] is True and results[index] == 'en':
                    count_true_en = count_true_en + 1 * weights[index]
                elif attributes[index_attribute][index] is True and results[index] == 'nl':
                    count_true_nl = count_true_nl + 1 * weights[index]
                elif attributes[index_attribute][index] is False and results[index] == 'en':
                    count_false_en = count_false_en + 1 * weights[index]
                elif attributes[index_attribute][index] is False and results[index] == 'nl':
                    count_false_nl = count_false_nl + 1 * weights[index]

            # Handliing certain outlier conditions
            if count_true_en == 0:
                rem_true_value = 0
                rem_false_value = ((count_false_en + count_false_nl) / (results_nl + results_nl)) * self.entropy(
                    count_false_en / (count_false_nl + count_false_en))
            elif count_false_en == 0:
                rem_false_value = 0
                rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en)) * self.entropy(
                    count_true_en / (count_true_nl + count_true_en))
            else:
                rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en)) * self.entropy(
                    count_true_en / (count_true_nl + count_true_en))

                rem_false_value = ((count_false_en + count_false_nl) / (results_nl + results_en)) * self.entropy(
                    count_false_en / (count_false_nl + count_false_en))

            gain_for_attribute = self.entropy(results_en / (results_en + results_nl)) - (rem_true_value +
                                                                                         rem_false_value)
            gain.append(gain_for_attribute)

        max_gain_attribute = gain.index(max(gain))
        root.value = max_gain_attribute
        count_max_true_en = 0
        count_max_true_nl = 0
        count_max_false_en = 0
        count_max_false_nl = 0

        for index in range(len(attributes[max_gain_attribute])):
            if attributes[max_gain_attribute][index] is True:
                if results[index] == 'en':
                    count_max_true_en = count_max_true_en + 1 * weights[index]
                else:
                    count_max_true_nl = count_max_true_nl + 1 * weights[index]
            else:
                if results[index] == 'en':
                    count_max_false_en = count_max_false_en + 1 * weights[index]
                else:
                    count_max_false_nl = count_max_false_nl + 1 * weights[index]

        left_obj = helper(attributes, None, results, None, depth + 1,
                          None, None)
        right_obj = helper(attributes, None, results, None, depth + 1,
                           None, None)
        if count_max_true_en > count_max_true_nl:
            left_obj.value = 'en'
        else:
            left_obj.value = 'nl'
        if count_max_false_en > count_max_false_nl:
            right_obj.value = 'en'
        else:
            right_obj.value = 'nl'

        root.left = left_obj
        root.right = right_obj

        return root

    def predict_ada(self, hypothesis_file, input_test_file):
        """
        Making the prediction using the saved adaboost model
        :param hypothesis_file:File containing the adaboost model
        :param input_test_file:Test file to be tested
        :return:None
        """
        # Loading model from the file
        object = pickle.load(open(hypothesis_file, 'rb'))
        file_open = open(input_test_file)
        sentence_list = []
        counter = 0
        sentence = ''

        # Take out 15-word lines from the test file
        for line in file_open:
            words = line.split()

            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
                    sentence += word
                    sentence_list.append(sentence)
                    sentence = ''
                    counter = 0

        attribute1 = []
        attribute2 = []
        attribute3 = []
        attribute4 = []
        attribute5 = []
        attribute6 = []
        attribute7 = []
        attribute8 = []
        attribute9 = []
        attribute10 = []
        attribute11 = []

        # For every line decide on the value of features
        for line in sentence_list:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.presence_of_van(line))
            attribute5.append(self.presence_of_de_het(line))
            attribute6.append(self.check_for_een(line))
            attribute7.append(self.check_for_en(line))
            attribute8.append(self.check_for_common_dutch_words(line))
            attribute9.append(self.check_for_common_english_words(line))
            attribute10.append(self.presence_of_a_an_the(line))
            attribute11.append(self.check_presence_of_and(line))

        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        attributes.append(attribute8)
        attributes.append(attribute9)
        attributes.append(attribute10)
        attributes.append(attribute11)

        statement_pointer = 0
        hypot_weights = object[1]
        hypot_list = object[0]

        # For every 15-word line make a prediction
        for sentence in sentence_list:
            total_summation = 0
            for index in range(len(object[0])):
                total_summation += self.make_final_prediction(hypot_list[index], sentence, attributes,
                                                              statement_pointer) * hypot_weights[index]

            if total_summation > 0:
                print('en')
            else:
                print('nl')
            statement_pointer += 1

    def make_final_prediction(self, stump, sentence, attributes, index):
        """
        Returns prediction based on the input hypothesis(stump) in consideration
        :param stump:Input hypothesis
        :param sentence:Input sentence
        :param attributes:Total attributes/features we have decided on
        :param index:Index of the input test statement in the test statement list
        :return:
        """
        attribute_value = stump.value
        if attributes[attribute_value][index] is True:
            if stump.left.value == 'en':
                return 1
            else:
                return -1
        else:
            if stump.right.value == 'en':
                return 1
            else:
                return -1

    def check_avg_word_length_greater_than_5(self, statement):
        """
        Check the average word length of the statement
        :param statement: Input statement
        :return: Boolean value representing whether the average word size is greater than 5 or lesser than 5
        """
        words = statement.split()
        total_word_size = 0
        total_words = 0
        for word in words:
            total_word_size = total_word_size + len(word)
            total_words = total_words + 1
        if total_word_size / total_words > 5:
            return True
        else:
            return False

    def containsQ(self, statement):
        """
        Check for occurence of the character Q
        :param statement:Input statement
        :return:Boolean value representing the presence of a character
        """
        if statement.find('Q') < 0 or statement.find('q') < 0:
            return False
        else:
            return True

    def containsX(self, statement):
        """
        Check for occurence of the character Q
        :param statement:Input statement
        :return:Boolean value representing the presence of a character
        """
        if statement.find('x') < 0 or statement.find('X') < 0:
            return False
        else:
            return True

    def check_for_en(self,statement):
        """
        Checking for the presence of the word en in the sentence
        :param statement:Input Statement
        :return:Boolean value representing the presence or absence of the word
        """
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') == 'en':
                return True
        return False

    def check_for_common_dutch_words(self,statement):
        """
        Checking for the presence of common dutch words
        :param statement:Input Statement
        :return:Boolean value representing the presence or absence of the common dutch words
        """
        list = ['naar','be','ik','het','voor','niet','met','hij','zijn','ze','wij','ze','er','hun','zo','over','hem','weten'
                'jouw','dan','ook','onze','deze','ons','meest']
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') in list:
                return True
        return False

    def check_for_common_english_words(self,statement):
        """
        Checking for the presence of common english words
        :param statement: Input statement
        :return: Boolean value representing the presence of common english words
        """
        list = ['to','be','I', 'it','for','not','with','he','his','they','we','she','there', 'their','so', 'about','me',
                'him','know','your','than','then','also','our','these','us','most']
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') in list:
                return True
        return False

    def presence_of_van(self, statement):
        """
        Check if the statement contains the string van
        :param statement:Input statement
        :return:Boolean value representing the presence of the string 'van'
        """
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') == 'van':
                return True
        return False


    def presence_of_de_het(self, statement):
        """
        Check if the statement contains the string de and het
        :param statement:Input statement
        :return:Boolean value representing the presence of the word 'de' or 'het' or both
        """
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') == 'de' or word.lower().replace(',','') =='het':
                return True
        return False

    def presence_of_a_an_the(self,statement):
        """
        Check for the presence of articles a an the
        If they are present , chances are statement is in  english language
        :param statement:
        :return: Boolean value reprenting the presence of articles
        """
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') == 'a' or word.lower().replace(',','') =='an' or word.lower().replace(',','') =='the':
                return True
        return False

    def check_for_een(self,statement):
        """
        Checking for the presence of the word een
        :param statement:Input 15-word statement
        :return:Boolean value representing the presence of the word 'een' in the 15-word sentence
        """
        words = statement.split()
        for word in words:
            if word.lower().replace(',','') == 'een':
                return True
        return False

    def check_presence_of_and(self,sentence):
        """
        Checking presence of 'and' in the sentence
        :param sentence:Input sentence
        :return:Boolean value representing the presence of 'and'
        """
        words = sentence.split()
        for word in words:
            if word.lower().replace(',','') == 'and':
                return True
        return False

def main():
    """
    Main Function
    :return: None
    """
    cl_obj = classifylanguage()
    # Check if right command line arguments given
    try:
        if sys.argv[1] == 'train':
            if sys.argv[4] == 'dt':
                cl_obj.collect_data_dt(sys.argv[2], sys.argv[3])
            else:
                cl_obj.collect_data_ada(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == 'predict':
            if sys.argv[4] =='dt':
                cl_obj.predict_dt(sys.argv[2], sys.argv[3])
            else:
                cl_obj.predict_ada(sys.argv[2], sys.argv[3])
    except:
        print('Syntax :train <examples> <hypothesisOut> <learning-type>')
        print('or')
        print('Syntax :predict <hypothesis> <file> <testing-type(dt or ada)>')

if __name__ == "__main__":
    main()

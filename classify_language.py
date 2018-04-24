class classifylanguage:

    __slots__ = 'train_file'

    def __init__(self):
        print()

    def collect_data_dt(self, example_file, hypothesis_file):
        # Gather data from example file
        statements, results = self.gather_data(example_file)
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

        file = open(example_file)
        for line in statements:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.contains_dutch_dipthongs(line))
            attribute5.append(self.contains_eng_dipthongs(line))
            attribute6.append(self.presence_of_van(line))
            attribute7.append(self.presence_of_de_het(line))
            #attribute8.insert(self.check_average_e_used_greater_than_100(values))
            #attribute9.insert(self.check_property9(values))
            #attribute10.insert(self.check_property10(values))
        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)

        number_lst = []
        for i in range(len(results)):
            number_lst.append(i)

        seen = []
        queue = Queue()
        # Setting root for decision tree
        root = helper(attributes, seen, results, number_lst, 0, None, None)

        value = self.train_decision_tree(root,attributes, seen, results, number_lst, 0, None, None, queue)
        # BFS for creating the remaining decision tree
        while not queue.empty():
            queue_obj = queue.get()
            if type(queue_obj) != str:
                predict = self.train_decision_tree(queue_obj,queue_obj.attributes, queue_obj.seen, queue_obj.results,
                                                   queue_obj.total_results, queue_obj.depth,
                                                   queue_obj.prediction_at_this_stage, queue_obj.bool,
                                                   queue)
            else:
                print(queue_obj)
        queue2 = Queue()
        queue2.put(root)
        print('\n')
        self.traverse_tree(queue2)
        # Saving the decision tree object in the hypothis file using pickle
        filehandler = open(hypothesis_file, 'wb')
        pickle.dump(root, filehandler)
        
    def predict_dt(self,hypothesis,file):
        object = pickle.load(open(hypothesis, 'rb'))
        file_open =  open(file)
        sentence_list = []
        counter = 0
        sentence = ''
        for line in file_open:
            words = line.split()


            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
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

        file_open.seek(0)

        for line in sentence_list:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.contains_dutch_dipthongs(line))
            attribute5.append(self.contains_eng_dipthongs(line))
            attribute6.append(self.presence_of_van(line))
            attribute7.append(self.presence_of_de_het(line))
        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        #attributes.append(attribute8)

        statement = 0
        file_open.seek(0)
        # Check whehter sentence_list or file_open
        for sentence in sentence_list:
            object_temp = object
            while type(object_temp.value) != str:
                value = attributes[object_temp.value][statement]
                if value == 'True':
                    object_temp = object_temp.left
                else:
                    object_temp = object_temp.right
            print(object_temp.value)
           
    def train_decision_tree(self,root,attributes,seen, results,total_results, depth, prevprediction, bool,queue):
        # If depth reached returned the plurality of the remaining samples
        if depth == len(attributes) - 1:
            counten = 0
            countnl = 0
            for index in total_results:
                if results[index] is 'en':
                    counten = counten + 1
                elif results[index] is 'nl':
                    countnl = countnl + 1
            if counten > countnl:
                queue.put('en')
                root.value = 'en'
            else:
                queue.put('nl')
                root.value = 'nl'
        # If no examples left return the previous predictions
        elif len(total_results) == 0:
            queue.put(prevprediction)
            root.value = prevprediction
        # If all examples of the same type return the type
        elif self.number_of_diff_values(results, total_results) == 0:
            queue.put(results[total_results[0]])
            root.value = results[total_results[0]]
        # If all the attributes have been used in a portion of the tree 
        # We dont use same attributes in the same subtree
        elif len(attributes) == len(seen):
            counten = 0
            countnl = 0
            for index in total_results:
                if results[index] is 'en':
                    counten = counten + 1
                elif results[index] is 'nl':
                    countnl = countnl + 1
            if counten > countnl:
                queue.put('en')
                root.value = 'en'
            else:
                queue.put('nl')
                root.value = 'nl'
        else:
        # Calculation of best attribute to split on based on information gain and entropy
            gain = []
            results_en = 0
            results_nl = 0
            for index in total_results:
                if results[index] == 'en':
                    results_en = results_en + 1
                else:
                    results_nl = results_nl + 1
            for index_attribute in range(len(attributes)):
                if index_attribute in seen:
                    gain.append(-100)
                    continue
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

                    if(count_true_nl + count_true_en == 0) or (count_false_en + count_false_nl == 0):
                        gain_for_attribute = 0
                        gain.append(gain_for_attribute)
                        continue

                    # Handliing certain outlier conditions
                    if count_true_en == 0:
                        rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en))
                        rem_false_value = ((count_false_en + count_false_nl) / (results_nl + results_nl)) * self.entropy(
                            count_false_en / (count_false_nl + count_false_en))
                    elif count_false_en == 0:
                        rem_false_value = ((count_false_en + count_false_nl)/ (results_nl + results_en))
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
            #print('Splitting attribute',max_gain_attribute,'on depth:', depth)

            seen.append(max_gain_attribute)

            index_True = []
            index_False = []
            # Sampling samples of maximum gain attribute
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
            # Samples corresponding to true value of maximum gain attribute
            left_obj = helper(attributes, seen, results, index_True, depth + 1,
                                                          prediction_at_this_stage, bool_true)
            # Samples corresponding to false values of maximum gain attribute
            right_obj = helper(attributes, seen, results, index_False, depth + 1,
                                                          prediction_at_this_stage, bool_false)
            root.left = left_obj
            root.right = right_obj
            queue.put(left_obj)
            queue.put(right_obj)
            return max_gain_attribute
        
    def gather_data(self, file):
        """
        Gathers data from the train.dat file for training
        :param file:input training file
        :return:list of statements and final predictions
        """
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
        print(statements[:])

        # Get all the results

        results = []
        pointer = 0
        for values in all_data_stripped_space:
            if values.startswith('nl|') or values.startswith('en|'):
                results.insert(pointer, values[0:2])
                pointer = pointer + 1
        print(results)

        return statements, results

    def predict_ada(self,hypothesis_file, input_test_file):
        object = pickle.load(open(hypothesis_file, 'rb'))
        file_open =  open(input_test_file)
        sentence_list = []
        counter = 0
        sentence = ''
        for line in file_open:
            words = line.split()


            for word in words:
                if counter != 14:
                    sentence += word + ' '
                    counter += 1
                else:
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

        file_open.seek(0)
        for line in sentence_list:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.contains_dutch_dipthongs(line))
            attribute5.append(self.contains_eng_dipthongs(line))
            attribute6.append(self.presence_of_van(line))
            attribute7.append(self.presence_of_de_het(line))
        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)
        #attributes.append(attribute8

        file_open.seek(0)
        statement_pointer = 0
        hypot_weights = object[1]
        hypot_list = object[0]
        for sentence in sentence_list:
            total_summation = 0
            for index in range(len(object[0])):
                total_summation += self.make_final_prediction(hypot_list[index],sentence,attributes,statement_pointer) * hypot_weights[index]

            if total_summation > 0:
                print('en')
            else:
                print('nl')
            statement_pointer += 1
            
    def collect_data_ada(self,example_file,hypothesis_file):
        statements, results = self.gather_data(example_file)
        weights = [1/len(statements)] * len(statements)

        # Number of hypothesis
        number_of_decision_stumps = 10
        number_of_examples_correctly_classified = 0
        number_of_examples_incorrectly_classified = 0

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


        for line in statements:
            attribute1.append(self.containsQ(line))
            attribute2.append(self.containsX(line))
            attribute3.append(self.check_avg_word_length_greater_than_5(line))
            attribute4.append(self.contains_dutch_dipthongs(line))
            attribute5.append(self.contains_eng_dipthongs(line))
            attribute6.append(self.presence_of_van(line))
            attribute7.append(self.presence_of_de_het(line))

        attributes = []
        attributes.append(attribute1)
        attributes.append(attribute2)
        attributes.append(attribute3)
        attributes.append(attribute4)
        attributes.append(attribute5)
        attributes.append(attribute6)
        attributes.append(attribute7)

        queue = Queue()
        number_lst = []
        stump_values = []
        hypot_weights = [0.000000000000000000000000001] * number_of_decision_stumps

        for i in range(len(results)):
            number_lst.append(i)

        root = helper(attributes,None,results, number_lst, 0, None, None)

        for hypothesis in range(1,number_of_decision_stumps):
            stump = self.return_stump(0, root, attributes, results,number_lst, weights , queue)
            error = 0.000000000000000000000000000000001
            for index in range(len(statements)):
                if self.prediction(stump,statements[index],attributes,index) != results[index]:
                    error = error + weights[index]
            for index in range(len(statements)):
                if self.prediction(stump, statements[index], attributes, index) == results[index]:
                     weights[index] = weights[index] * ((error)/(1 - error))
            # ASK ABOUT NORMALIZING WEIGHTS FROM SOMEONE
            hypot_weights[hypothesis] = math.log(((1 - error)/(error)),2.0)
            stump_values.append(stump)

        filehandler = open(hypothesis_file, 'wb')
        pickle.dump((stump_values,hypot_weights), filehandler)


    def prediction(self,stump,statement,attributes,index):
        attribute_value = stump.value
        if attributes[attribute_value][index] is True:
            return stump.left.value
        else:
            return stump.right.value

    def return_stump(self,depth ,root, attributes, results,total_results, weights , queue):
        gain = []
        results_en = 0
        results_nl = 0
        for index in total_results:
            if results[index] == 'en':
                results_en = results_en + 1*weights[index]
            else:
                results_nl = results_nl + 1*weights[index]
        for index_attribute in range(len(attributes)):
                count_true_en = 0
                count_true_nl = 0
                count_false_en = 0
                count_false_nl = 0
                for index in total_results:
                    if attributes[index_attribute][index] is True and results[index] == 'en':
                        count_true_en = count_true_en + 1*weights[index]
                    elif attributes[index_attribute][index] is True and results[index] == 'nl':
                        count_true_nl = count_true_nl + 1*weights[index]
                    elif attributes[index_attribute][index] is False and results[index] == 'en':
                        count_false_en = count_false_en + 1*weights[index]
                    elif attributes[index_attribute][index] is False and results[index] == 'nl':
                        count_false_nl = count_false_nl + 1*weights[index]

                # Handliing certain outlier conditions
                if count_true_en == 0:
                    rem_true_value = ((count_true_en + count_true_nl) / (results_nl + results_en))
                    rem_false_value = ((count_false_en + count_false_nl) / (results_nl + results_nl)) * self.entropy(
                        count_false_en / (count_false_nl + count_false_en))
                elif count_false_en == 0:
                    rem_false_value = ((count_false_en + count_false_nl) / (results_nl + results_en))
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
                    count_max_true_en = count_max_true_en + 1*weights[index]
                else:
                    count_max_true_nl = count_max_true_nl + 1*weights[index]
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
            if word.lower() == 'a' or word.lower() =='an' or word.lower() =='the':
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
            else:
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
            if sys.argv[4] == 'dt':
                cl_obj.predict_dt(sys.argv[2], sys.argv[3])
            else:
                cl_obj.predict_ada(sys.argv[2], sys.argv[3])
    except:
        print('Syntax :train <examples> <hypothesisOut> <learning-type>')
        print('or')
        print('Syntax :predict <hypothesis> <file> <testing-type(dt or ada)>')

if __name__=="__main__":
    main()

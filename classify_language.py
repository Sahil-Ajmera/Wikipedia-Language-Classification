class classifylanguage:

    __slots__ = 'train_file'

    def __init__(self):
        print()

    def train(self, file):
        statements, results = self.gather_data(file)
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

        for values in statements:
            attribute1.insert(self.containsQ(values))
            attribute2.insert(self.containsX(values))
            attribute3.insert(self.check_avg_word_length_greater_than_5(values))
     
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
        if total_word_size/total_words > 5:
            return True
        else:
            return False
        
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

    
def main():
    cl_obj = classifylanguage()
    cl_obj.train('train.dat')

if __name__=="__main__":
    main()

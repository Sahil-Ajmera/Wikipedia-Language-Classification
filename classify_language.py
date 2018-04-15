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

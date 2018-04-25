




def main():
    # sentence = input()
    # words = sentence.split()
    # sentences = []
    # counter = 0
    # sample = ''
    # for word in words:
    #     if counter != 15:
    #         sample = sample + word + ' '
    #         counter = counter + 1
    #     else:
    #         counter = 0
    #         sample = sample + ' ' +'nl|'
    #         sentences.append(sample)
    #         sample = ''
    # for sentence in sentences:
    #     print(sentence)

    input_file = input()
    file_open = open(input_file,encoding="utf-8-sig")

    for lines in file_open:
        for words in lines.split():
            words.replace(",","")
        print(lines)
if __name__== "__main__":
    main()
import pandas as pd
import re, os

class data_tool(object):

    def __init__(self, train_path, test_path, truncated_length):
        self.train_path = train_path
        self.test_path = test_path

        print("load data...")
        # training set
        # self.train = pd.read_table(train_path, sep="\t")
        # self.train_x = [i[:truncated_length] for i in self.train['Phrase']]
        self.train = pd.read_csv(train_path, names=['1', '2', '3'])
        self.train_x = [' '.join(self.train.iloc[i, [1, 2]])[:truncated_length] for i in range(self.train.shape[0])]

        # self.train_y = [[0] * 4 for i in range(self.train.shape[0])]
        # _ = [self.train_y[i].insert(j, int(j)) for i, j in enumerate(self.train['Sentiment'])]
        self.train_y = [[0] * 3 for i in self.train.iloc[:, 0]]
        _ = [self.train_y[i].insert(j-1, 1) for i, j in enumerate(self.train.iloc[:, 0])]


        # test set
        #self.test = pd.read_table(test_path, sep='\t')
        #self.test_x = [i[:truncated_length] for i in self.test['Phrase']]

        self.test = pd.read_csv(test_path, names=['1', '2', '3'])
        self.test_x = [' '.join(self.test.iloc[i, [1, 2]])[:truncated_length] for i in range(self.test.shape[0])]
        print("data loaded!")

        self.test_y = [[0] * 3 for i in self.test.iloc[:, 0]]
        _ = [self.test_y[i].insert(j - 1, 1) for i, j in enumerate(self.test.iloc[:, 0])]
        self.test_y = np.array(self.test_y)

        # character_dict
        self.char_dict = self.character_corpus()

        # word_vector
        self.one_hot_word_vector = self.to_one_hot(self.char_dict, truncated_length)

        # form data
        self.train_x = np.array([self.text2index(j, self.char_dict, truncated_length) for j in self.train_x])
        self.test_x = np.array([self.text2index(j, self.char_dict, truncated_length) for j in self.test_x])

        self.train_y = np.array(self.train_y)
        self.test_y = np.array(self.test_y)

    def clean(self, string):
        return re.sub("[a-zA-Z]\\+[a-zA-Z]", '\n', string)

    def character_corpus(self):
        char_dict = {char: index + 1 for index, char
                     in enumerate("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'\"/\\|_@#$%^&*~`+-=<>()[]{}")}
        char_dict['\n'] = 69
        return char_dict

    def to_one_hot(self, char_dict, truncated_length):
        tmp = np.zeros([truncated_length, char_dict.keys().__len__()])
        for i, j in enumerate(char_dict.values()):
            tmp[i, j-1] = 1
        return np.concatenate([np.zeros([truncated_length, 1]), tmp], axis=1)

    def text2index(self, text, vocab_dict, truncated_length):
        """
        tokenization
        """
        text = self.clean(text)
        tmp = [0] * truncated_length
        for i in range(1, len(text)+1):
            tmp[i-1] = vocab_dict.get(text[-i].lower(), 0)
        return tmp

    # generate batches of data to train
    def generate_batches(self, data_x, data_y, epoch_size, batch_size, shuffle=False):
        data_size = len(data_x)
        num_batches = data_size // batch_size + 1

        for i in range(epoch_size):
            if shuffle:
                np.random.seed(1000)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data_x, shuffle_data_y = data_x[shuffle_indices], data_y[shuffle_indices]
            else:
                shuffle_data_x, shuffle_data_y = data_x, data_y

            for j in range(num_batches):
                start_index = j * batch_size
                end_index = min((j+1) * batch_size, data_size)
                batch_x = shuffle_data_x[start_index: end_index]
                batch_y = shuffle_data_y[start_index: end_index]
                yield batch_x, batch_y

    def save_data(self, result):
        test_data = pd.read_csv(test_path, names=['1', '2', '3'])
        test_data['4'] = [i+1 for i in result.reshape(-1).tolist()]
        test_data = test_data.loc[:, ['1', '4']]
        print((test_data['1'] == test_data['4']).mean())
        test_data.to_csv("data/sample_submission.csv", index=False)
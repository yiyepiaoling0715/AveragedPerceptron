import pickle
import collections
from averaged_perceptron import AveragedPerceptron
import random
import logging
logging.basicConfig(level=logging.INFO)

def _pc(c,n):
    return float(round(c/n,6))*100

class PerceptronTagger(object):
    _START = ['-START2-','-START-']
    _END = ['-END-','-END2-']

    def __init__(self,bool_load):
        self.classes = set()
        self.tagdict = dict()
        self.model = AveragedPerceptron()
        if bool_load:
            self.load(PICKLE)
    def _get_features(self, i, word, context, prev, prev2):
        features = collections.defaultdict(int)
        i += len(self._START)

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        add('bias')   ##意义在于哪？？？
        add('i suffix ', word[-3:])
        try:
            add('i prev1 ', word[0])
        except IndexError as e:
            print(e.args)
        add('i-1 tag ', prev)
        add('i-2 tag ', prev2)
        add('i tag+i-2 tag ', prev, prev2)  # flag 有问题
        add('i word ', context[i])
        add('i-1 tag+i word ', prev, context[i])
        add('i-1 word ', context[i - 1])
        add('i-1 suffix ', context[i - 1][-3:])
        add('i-2 word ', context[i - 2])
        add('i+1 word ', context[i + 1])
        add('i+1 suffix ', context[i + 1][-3:])
        add('i+2 word ', context[i + 2])
        return features

    def _normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word.isdigit():
            return '!DIGIT'
        else:
            return word.lower()

    def _make_tagdict(self, sentences):
        freq_thres = 20
        ambiguity_thres = 0.97
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        for words,tags in sentences:
            for tag, word in zip(tags, words):
                counts[word][tag] += 1
                self.classes.add(tag)

        for word, tags in counts.items():
            n = sum(tags.values())
            tag,mode=max(tags.items(),key=lambda x:x[1])
            # for tag, num in tags.items():
            if n > freq_thres and float(mode) / n > ambiguity_thres:
                self.tagdict[word] = tag

    def load(self, loc):
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError as e:
            msg = ("Missing trontagger.pickle file.")
            raise IOError(msg)
        self.model.weights,self.classes,self.tagdict  = w_td_c
        self.model.set_classes(self.classes)

    def tag(self,corpus):
        s_split=lambda x:x.split('\n')
        w_split=lambda x:x.split()
        def split_corpus(corpus):
            for sentence in s_split(corpus):
                yield w_split(sentence)
        tokens=list()
        prev,prev2=self._START
        for sentence in split_corpus(corpus):
            context=self._START+[self._normalize(word) for word in sentence]+self._END
            for index,word in enumerate(sentence):
                # word_normalize=self._normalize(word)
                word_normalize=word
                guess=self.tagdict.get(word_normalize)
                if not guess:
                    features=self._get_features(index,word,context,prev,prev2)
                    guess=self.model.predict(features)
                tokens.append((word,guess))
                prev2=prev
                prev=guess
        return  tokens
    def train(self, sentences, save_loc=None, nr_iter=5):
        logging.info('Start training.....')
        self._make_tagdict(sentences)
        # print('self.tagdict==>\n',self.tagdict)
        self.model.set_classes(self.classes)
        for iter_n in range(nr_iter):
            c,n=0,0
            for words,tags in sentences:
                prev,prev2=self._START
                # context=self._START+[self._normalize(word) for word in words]+self._END
                context=self._START+[word for word in words]+self._END
                for index,word in enumerate(words):
                    features=self._get_features(index,word,context,prev,prev2)
                    guess=self.tagdict.get(word)
                    if not guess:
                        guess=self.model.predict(features)
                        self.model.update(guess,tags[index],features)
                    c+=guess==tags[index]
                    n+=1
                    prev2=prev
                    prev=guess
            # logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
            logging.info('Iter {0}: {1}/{2}={3}'.format(iter_n,c,n,_pc(c,n)))
            random.shuffle(sentences)
        self.model.average_weights()
        if save_loc:
            pickle.dump([self.model.weights,self.model.get_classes,self.tagdict],open(save_loc,'wb'))
        else:
            pass
        return None

if __name__=='__main__':
    pt=PerceptronTagger(False)
    PICKLE = "data/trontagger-0.1.0.pickle"
    train_filepath='data/train.txt'
    test_filepath='data/test.txt'
    nr_iter=5
    try:
        pt.load(PICKLE)
        logging.info('Start Testing....')
        fr=open(test_filepath,'r')
        sentence=[[],[]]
        right,total=0,0
        for line in fr:
            words = line.split()
            # param=(words[0],words[1])
            # sentence.append(param)
            if len(words) != 2:
                continue
            sentence[0].append(words[0])
            sentence[1].append(words[1])
            if words[0] == '.':
                text=''
                for i, word in enumerate(sentence[0]):
                    text += word
                    if i < len(sentence[0]):
                        text += ' '
                tokens=pt.tag(text)
                for index,(word_iter,guess_iter) in enumerate(tokens):
                    try:
                        if sentence[1][index]==guess_iter.strip('\r\n '):
                            right+=1
                    except IndexError as e:
                        print(e.args)
                    total+=1
                sentence = [[], []]
            else:
                pass
        logging.info("Precision : %f", right / total)

    except IOError as e:
        training_data=list()
        fr=open(train_filepath,'r')
        sentence=[[],[]]
        logging.info('Reading Corpus .....')
        for line in fr:
            words=line.split('\t')
            # param=(words[0],words[1])
            # sentence.append(param)
            sentence[0].append(words[0])
            sentence[1].append(words[1])
            if words[0]=='.':
                training_data.append(sentence)
                sentence = [[], []]
            else:
                pass
        logging.info('training corpus size {}'.format(len(training_data)))
        pt.train(training_data,PICKLE,nr_iter)
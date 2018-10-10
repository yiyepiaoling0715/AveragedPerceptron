import collections
import pickle
class AveragedPerceptron(object):
    def __init__(self):
        self.weights=dict()
        self._classes=dict()
        self._tstaps=collections.defaultdict(int)
        self._totals=collections.defaultdict(int)
        self.i=0
    def set_classes(self,classes_in):
        self._classes=classes_in
    @property
    def get_classes(self):
        return self._classes
    def predict(self,features):
        scores=collections.defaultdict(int)
        for feat,value in features.items():
            if feat not in self.weights or value==0:
                continue
            for label,weight in self.weights[feat].items():
                scores[label]+=value*weight
        return max(self._classes,key=lambda x:(scores[x],x))

    def update(self,guess,truth,features):
        def update_feat(f,c,w,v):
            param=(f,c)
            self._totals[param]+=(self.i-self._tstaps[param])*w
            self._tstaps[param]=self.i
            self.weights[f][c]=w+v
        self.i+=1
        if guess==truth:
            return True
        for feature in features:
            weights=self.weights.setdefault(feature,{})
            # value=self.classes.get(feature,0.0)
            update_feat(feature,truth,weights.get(truth,0.0),1.0)
            update_feat(feature,guess,weights.get(guess,0.0),-1.0)

    def average_weights(self):
        for feature,weights in self.weights.items():
            average_feature=collections.defaultdict(int)
            for label,weight in weights.items():
                param=(feature,label)
                weight=(self.i-self._tstaps[param])*weight
                self._totals[param]+=weight
                average_feature[label]=round(self._totals[param]/self.i,6)
            self.weights[feature]=average_feature
    def save(self,path):
        return  pickle.dump(dict(self.weights),open(path,'wb'))
    def load(self,path):
        self.weights=pickle.load(open(path,'rb'))
def train():
    pass
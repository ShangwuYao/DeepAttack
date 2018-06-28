#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:09:17 2018

@author: mancx111
"""
import numpy as np
import json
import argparse
from keras.models import load_model


#parser = argparse.ArgumentParser()
#parser.add_argument("--model",help="model path",default='.')
#parser.add_argument("--index",help="index path",default='./imagenet_class_index.json')
#
#
#
#args = parser.parse_args()



'''
The class takes model's path and its json file's path as input
'''
class Model_Evaluator(object):
    def __init__(self,model_path,json_path):
        super(Model_Evaluator,self).__init__()
        self.set_path=['CLEAN','FGSM','Mi-FGSM','I-FGSM']
        
        
        #Just for alpha testing,need change after deployment
        self.models=[model_path]
        
        self.class_index=json.load(open(json_path))
        self.class_set=set([self.class_index[x][0] for x in self.class_index])  ##need change after deployment
    #Private functions that only be called by init
    def load_set(self,set_):
        my_dict={}
        inputs=np.load(open('./image_data_final/evaluate_input_'+set_+'.npy'))
        labels=np.load(open('./image_data_final/evaluate_label_'+set_+'.npy'))
        for key,input_ in zip(labels,inputs):
            if key not in my_dict:
                my_dict[key]=[]
            my_dict[key].append(input_)
        return my_dict
            
            
    
    #Private functions that only will be called by evaluate
    def decode_predictions(self,predict):
        argmax=np.argmax(predict,axis=1)
        confidence=np.max(predict,axis=1).sum()
        decode=[]
        for label in argmax:
            decode.append(self.class_index[str(label)][0]) #need change after deployment
        return np.array(decode),confidence
        
    def evaluate(self):
        '''
        The output of this model should be accuracy
        '''
        def calculate_acc(data,model):
            '''
            data is a dictionary where y is the key and X are the value
            '''
            right,total,confidence=0,0,0.0
            for key in data:
                if len(data[key]) ==0 or key not in self.class_set:
                    continue
                y=key
    
                batch=np.vstack(data[y])-127.5
                predict,con=self.decode_predictions(model.predict(batch,batch_size=10))
                right+=len(predict[predict==y])
                total+=len(data[y])
                confidence+=con
            return float(right*100)/total,confidence*100/total
            #print clean_predict.shape
        
        
        
        result=None
        #The deployment should only have one model in self.models
        for model_path in self.models:
            degrade,score_list=0.0,[]
            result={'robustness':None,'rating':None,'details':[],'graph_link':None,'suggestion':None}
            for path in self.set_path:
                inputs=self.load_set(path)
                score={}
                acc,confidence=calculate_acc(inputs,load_model(model_path)) ##Need change after deployment
                score['attack_method']=path.upper()
                score['accuracy']=str(acc)+'%'
                score['confidence']=str(confidence)+'%'
                result['details'].append(score)
                score_list.append(acc)
                
                if len(score_list)>1:
                    degrade+=score_list[-1]
            result['robustness']=str(100*(score_list[0]-degrade/(len(score_list)-1))/score_list[0])
      
        return result
        
            
                
###Your work
##specify what is the input here

#result={}
#try:
#    model=Model_Evaluator(args.model,args.index)
#    result=model.evaluate()
#except Exception as exc:
#    result['message']=exc.__str__()
#
#output=json.dumps(result)
    

##What is the output of the evaluate
        
    
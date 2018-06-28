import boto3
import json
import time
import urllib
import os
import zipfile
from db import db, Submission
from evaluation import Model_Evaluator


s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
response_model = s3_client.head_object(Bucket='advex', Key='vgg16.h5')
response_index = s3_client.head_object(Bucket='advex', Key='imagenet_class_index.json')
bucket = s3.Bucket('advex')

sqs = boto3.client('sqs')
resp = sqs.get_queue_url(QueueName='advex')
queue_url = resp['QueueUrl']

urllib.urlretrieve("https://s3.amazonaws.com/advex/evaluation_data/image_data_final.zip", filename="image_data_final.zip")
os.mkdir('./image_data_final',0755)
zip_ref = zipfile.ZipFile('image_data_final.zip', 'r')
zip_ref.extractall('./image_data_final')
zip_ref.close()

# SAMPLE_FEEDBACK = {
# 	"robustness": "9",
# 	"rating": "Good",
# 	"details": {
# 		"original_accuracy": "98.55%",
# 		"attack_results": [
# 			{
# 				"attack_method": "FGSM",
# 				"accuracy": "80.05%",
# 				"confidence": "95%"
# 			},
# 			{
# 				"attack_method": "Basic Iterative Method",
# 				"accuracy": "92.10%",
# 				"confidence": "91%"
# 			},
# 			{
# 				"attack_method": "Carlini Wagner",
# 				"accuracy": "94.10%",
# 				"confidence": "93%"
# 			},
# 			{
# 				"attack_method": "Momentum Iterative Method",
# 				"accuracy": "94.10%",
# 				"confidence": "93.7%"
# 			},
# 			{
# 				"attack_method": "DeepFool",
# 				"accuracy": "90.10%",
# 				"confidence": "89%"
# 			}
# 		]
# 	},
# 	"suggestion": "Your model can be made more robust by training it with some of the adversarial examples which you can download for free from your dashboard."
# }


def write_feedback(submission_id, feedback):
	print('Writing feedback.')
	submission = Submission.query.get(submission_id)
	submission.feedback = feedback
	db.session.commit()


def evaluate_job(job):
    print('Evaluating model.')
    feedback={}
    submission_id = job['submission_id']
    model_file = job['s3_model_key']
    index_file = job['s3_index_key']
    
    #Check 1: File extension
    if not model_file.endswith('.h5'):
        if not index_file.endswith('.json'):
            feedback = {"error": "Model file has to have .h5 as its extension. Index file has to have .json as its extension"}
        else:
            feedback = {"error": "Model file has to have .h5 as its extension."}
    else:
        if not index_file.endswith('.json'):
            feedback = {"error": "Index file has to have .json as its extension"}       
            
    #bucket.download_file(model_file, model_file)
    #bucket.download_file(index_file, index_file)
    
    model_size=response_model['ContentLength']
    index_size=response_index['ContentLength']
    #Check 2: File Size Check
    if not feedback:
        if model_size > 1073741824: # 1 GiB
            if index_size > 102400:
                feedback = {"error": ".h5 file can't be bigger than 1GB and .json file can't be bigger than 100KB."}
            else:
                feedback = {"error": ".h5 file can't be bigger than 1GB."}
        else:
            if index_size > 102400:
                feedback = {"error": ".json file can't be bigger than 100KB."}
    
    if not feedback:
        #The model file and index file are perfectly fine.

        result={}
        try:
            model=Model_Evaluator(model_file,index_file)
            result=model.evaluate()
        except Exception as exc:
            result['message']=exc.__str__()
        print result
        feedback=json.dumps(result)
        #Check 3: DDOS
            
    write_feedback(submission_id, feedback)
    
def main():
	while True:
		resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
		if 'Messages' not in resp:
			print('No messages received, sleep for 10s.')
			time.sleep(10)
			continue

		print('Message received.')
		message = resp['Messages'][0]
		receipt_handle = message['ReceiptHandle']
		job = json.loads(message['Body'])

		# Process job
		evaluate_job(job)

		# Delete message
		resp = sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=receipt_handle)


if __name__ == '__main__':
	main()
